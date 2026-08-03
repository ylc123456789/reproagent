"""LLM interface for the agent-loop controller.

A single system prompt drives the agent.  All previous stage-specific prompts
(plan_environment, plan_probe, plan_experiment, revise_after_failure,
review_experiment_plan, final_review) are removed — the controller prompt
covers everything.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime

from .models import EnvironmentInfo, RepoContext, ReproTask

SYSTEM_PROMPT = """\
You are a machine-learning reproducibility engineer. Your task: reproduce a paper experiment inside a prepared conda environment.

## How to work

You have tools. On each turn, return a JSON action. The system executes it and shows you the result. Then you decide the next action. Repeat until you call finish.

## Tools

### run_commands
Execute one or more shell commands in the conda environment.
Commands already run inside the repository root — no `cd` needed.
Format:
{"thinking": "why", "action": "run_commands", "stage_hint": "environment|probe|experiment", "commands": ["cmd1", "cmd2"]}

### audit_env
Check installed packages, Python version, torch version, and whether CUDA/GPU is available.
Use this when you are unsure about the current environment state.
Format:
{"thinking": "why", "action": "audit_env"}

### call_coding_agent
Ask a coding agent to modify the repository code. Use this when a required metric is not printed by the script, or when a small config/logging change is needed. Do NOT use this for dependency installation — use run_commands for that.
Format:
{"thinking": "why", "action": "call_coding_agent", "coding_goal": "one-line goal for the coding agent", "coding_issues": ["specific issue 1", "specific issue 2"]}

### finish
End the task. Your final message will become the reproduction report, so make it thorough.
Format:
{"thinking": "why", "action": "finish", "finish_status": "completed|completed_with_failures|failed", "finish_summary": "your final report in Markdown"}

## Rules

### Safety (MUST follow — these will be enforced by the runner)
- Never use sudo, rm -rf, | bash, shutdown, or reboot.
- Never use parent-directory traversal (../ outside the repo).
- Do not use cd, tee, shell log redirection (> / >> / 2>&1 / | tee).
- Do not use conda activate or conda create — the environment is already prepared.

### Environment setup
- ALWAYS run pip install / package setup BEFORE any import or version check.
  A failed import does NOT mean setup failed — it means you tried to import before installing.
- When an NVIDIA GPU is visible, prioritize GPU-capable PyTorch/TensorFlow/JAX builds
  compatible with the reported driver/CUDA version.
- Follow the mirror policy from the context when choosing package indexes.
- If a command fails, read the error and fix it. If a dependency is missing, install it.
- If pip install fails with a specific version, try without the version pin or from a different index.

### Experiment
- Probe the script interface (--help, head, grep) before running training.
- If the goal requires a metric that the script does not print, call call_coding_agent
  to add logging. Do NOT skip the metric or assume it works.
- Prefer bounded/small runs first (few epochs) to validate the pipeline,
  then scale up if needed.
- Use `time` prefix to capture runtime when the goal requires it.

### Reporting
- When you call finish, your finish_summary must be a complete Markdown report
  with these sections:
  ## Summary — one paragraph, what happened
  ## Metrics — table: each goal requirement, [MET]/[NOT MET]/[UNCERTAIN], actual value,
    and which log file the evidence comes from
  ## Deviations — how the experiment differed from the paper/goal setup
  ## Data Index — every log file path and what it contains
  ## Next Steps — what a human should do next
"""


# ── context builder ──────────────────────────────────────────────

def build_context_block(task: ReproTask, repo_context: RepoContext, environment: EnvironmentInfo) -> str:
    """Build the initial user message with all context the agent needs."""
    env_text = f"Conda env: {environment.env_name}"
    if environment.created:
        env_text += " (freshly created — contains only Python, no packages installed yet)"
    else:
        env_text += " (reused — may already have packages)"

    return f"""## Task

Paper: {task.paper_url}
Repo: {task.repo_url} (cloned at {repo_context.repo_path}, commit {repo_context.commit_hash or 'unknown'})
Experiment goal: {task.experiment_goal}
Timeout: {task.timeout_seconds}s per command batch
Max steps: {task.max_steps}

## Environment

{env_text}
{_mirror_block(task)}

## Hardware

{repo_context.hardware_text}

## File Tree

{repo_context.file_tree}

## README / docs

{repo_context.readme_text[:16000]}

---

Begin. Think step by step, one action at a time. What is your first action?"""


def _mirror_block(task: ReproTask) -> str:
    profile = task.mirror_profile
    if profile == "none":
        return "Mirror policy: none. Use official package indexes or the repo instructions."
    strict = "strict" if task.mirror_strict else "preferred"
    lines = [f"Mirror policy: {profile} ({strict})."]
    if profile in ("cn", "autodl"):
        lines.append("For pip packages, prefer: -i https://mirrors.aliyun.com/pypi/simple")
        lines.append("Avoid --index-url https://download.pytorch.org/whl/ — it overrides domestic mirrors.")
    if profile == "autodl":
        lines.append("Only use aliyun PyTorch wheel find-links when a +cuXXX local-version wheel is required and available.")
    if task.mirror_strict:
        lines.append("Strict mode: if a package is unavailable from the mirror, report it instead of falling back to official indexes.")
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────

def call_llm(task: ReproTask, messages: list[dict]) -> str:
    """Call the LLM and return the response text."""
    if task.mock_llm:
        return _mock_response(messages)
    return _openai_compatible(task, messages)


def _openai_compatible(task: ReproTask, messages: list[dict]) -> str:
    api_key = os.environ.get(task.api_key_env)
    if not api_key:
        raise RuntimeError(f"{task.api_key_env} is not set. Use --mock-llm for local testing.")
    model = task.model or "gpt-4.1-mini"
    body = {"model": model, "messages": messages, "temperature": 0.2}
    url = _chat_completions_url(task.api_base)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _chat_completions_url(api_base: str) -> str:
    """Build the chat completions endpoint URL."""
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _mock_response(messages: list[dict]) -> str:
    """Deterministic mock for tests."""
    last = messages[-1]["content"] if messages else ""
    # initial context → run commands; after results → finish
    if "Begin." in last:
        return '{"thinking": "mock: probe the repo", "action": "run_commands", "stage_hint": "probe", "commands": ["head -20 README.md"]}'
    if "Start." in last:
        return '{"thinking": "mock: probe the repo", "action": "run_commands", "stage_hint": "probe", "commands": ["head -20 README.md"]}'
    if "Result" in last:
        return '{"thinking": "mock: done", "action": "finish", "finish_status": "completed", "finish_summary": "Mock run completed."}'
    return '{"thinking": "mock: done", "action": "finish", "finish_status": "completed", "finish_summary": "Mock run completed."}'
