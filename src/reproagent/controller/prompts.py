"""System prompt, context builders, and prompt formatting helpers.

All prompt construction logic lives here.  The controller imports
SYSTEM_PROMPT, build_initial_context, and build_turn_prompt from this
module; the LLM API layer is in llm.py.
"""

from __future__ import annotations

from ..context.policy import ContextPolicy
from ..models import AgentObservation, AgentState, EnvironmentInfo, RepoContext, ReproTask
from ..repository.context import workspace_mode

SYSTEM_PROMPT = """\
You are an experiment operator for machine-learning research repositories. You work inside a prepared conda environment: probe the repository, install dependencies, run experiments, and collect metric evidence from log files. When a paper is provided, your goal is reproducing its experiment; otherwise your goal is exactly the experiment goal given to you.

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
- Follow the mirror policy exactly — use domestic mirrors for all pip installs when configured.
- When an NVIDIA GPU is visible, prioritize GPU-capable PyTorch/TensorFlow/JAX builds
  compatible with the reported driver/CUDA version.
- If a command fails, read the error and fix it. If a dependency is missing, install it.
- If pip install fails with a specific version, try without the version pin or from a different index.
- After installing major dependencies (torch, tensorflow, etc.), re-run audit_env so you
  and any future call_coding_agent have accurate environment information.
- Experiment commands are refused until audit_env reports success. If the audit
  fails, repair the environment (pip installs) and re-run audit_env.

### Experiment
- Probe the script interface (--help, head, grep) before running training.
- After the experiment finishes, extract metrics from the experiment log files
  and proceed to finish. If metrics are not in the first portion of a log file,
  try reading from the end (tail) — training logs often have header material
  followed by epoch output.
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
  ## Deviations — how the experiment differed from the goal (and from the paper setup when reproducing)
  ## Data Index — every log file path and what it contains
  ## Next Steps — what a human should do next
"""


# ── context builders ─────────────────────────────────────────────

def build_initial_context(task: ReproTask, repo_context: RepoContext, environment: EnvironmentInfo,
                          policy: ContextPolicy | None = None,
                          dataset_links: list[dict] | None = None) -> str:
    """Build the very first user message — only called once at the start."""
    from ..runtime.dataset_cache import render_dataset_block
    env_text = _env_line(environment)
    readme_limit = policy.readme_chars if policy else 16000
    dataset_block = render_dataset_block(task, repo_context.repo_path, dataset_links or [])
    return f"""## Task

Paper: {task.paper_url or 'not provided'}
{_source_line(task)}
Workspace: {repo_context.repo_path} (mode: {workspace_mode(task)}, commit {repo_context.commit_hash or 'unknown'})
Experiment goal: {task.experiment_goal}
Timeout: {task.timeout_seconds}s per command batch
Max steps: {task.max_steps}
{dataset_block}{_input_artifacts_block(task)}

## Environment

{env_text}
{_mirror_block(task)}

## Hardware

{repo_context.hardware_text}

## File Tree

{repo_context.file_tree}

## README / docs

{repo_context.readme_text[:readme_limit]}

---

Begin. Think step by step, one action at a time. What is your first action?"""


def build_turn_prompt(state: AgentState, policy: ContextPolicy) -> str:
    """Build a fresh prompt for the NEXT agent turn from structured state.

    Each turn the prompt is rebuilt from scratch using the current state —
    old step output is compacted, file reads are cached, and the most
    recent step is shown in full.
    """
    parts: list[str] = []

    # --- task ---
    parts.append("## Task")
    parts.append(f"Goal: {state.task.experiment_goal}")
    parts.append(f"Paper: {state.task.paper_url}")
    parts.append(f"Repo: {state.task.repo_url}")
    input_block = _input_artifacts_block(state.task)
    if input_block:
        parts.append(input_block)
    if state.repo_context is not None:
        parts.append(
            f"Commands run with cwd = {state.repo_context.repo_path} (absolute). "
            "Relative paths in scripts resolve against this directory, "
            "not the script's own location."
        )
    remaining = state.task.max_steps - len(state.steps)
    parts.append(f"Timeout: {state.task.timeout_seconds}s | Steps used: {len(state.steps)}/{state.task.max_steps} (max)")

    # --- dataset cache (absolute-path mapping, system-resolved) ---
    from ..runtime.dataset_cache import render_dataset_block
    repo_path = state.repo_context.repo_path if state.repo_context else None
    dataset_block = render_dataset_block(state.task, repo_path, state.dataset_links)
    if dataset_block:
        parts.append(dataset_block)
    if remaining <= 4:
        parts.append(f"Only {remaining} step(s) remain. Prioritize finishing.")

    # --- environment ---
    if state.environment:
        parts.append(f"\n## Environment\n{_env_line(state.environment)}\n{_mirror_block(state.task)}")
    if state.last_audit:
        parts.append(f"\n## Latest Audit\n{_compact_audit(state.last_audit)}")

    # --- file cache ---
    if state.file_cache:
        headers: list[str] = []
        for path, text in list(state.file_cache.items())[-policy.file_cache_count:]:
            tail = text[-policy.file_cache_chars:]
            headers.append(f"{path} ({len(text)} chars):\n```\n{tail}\n```")
        parts.append(f"\n## Recent File Reads\n\n" + "\n\n".join(headers))

    # --- compacted steps ---
    history = state.steps
    if len(history) > 1:
        compacted = history[:-1][-policy.step_history:]
        if compacted:
            items: list[str] = []
            for step in compacted:
                items.append(_compact_step_line(step, policy))
            parts.append(f"\n## Previous Steps\n\n" + "\n".join(items))

    # --- last result (full) ---
    if history:
        parts.append(f"\n## Last Result (step {len(history)})\n{_format_step_full(history[-1], policy)}")

    parts.append("\nWhat is your next action? Return JSON.")
    return "\n".join(parts)


# ── helpers ──────────────────────────────────────────────────────

def _input_artifacts_block(task: ReproTask) -> str:
    """Structured upstream artifacts block — empty when none given."""
    if not task.input_artifacts:
        return ""
    lines = [
        "\n## Input Artifacts",
        "",
        "Artifacts produced by upstream tasks — read these files directly "
        "instead of re-deriving paths:",
    ]
    for item in task.input_artifacts:
        path = str(item.get("path", ""))
        description = str(item.get("description", ""))
        lines.append(f"- {path}" + (f" — {description}" if description else ""))
    return "\n".join(lines)


def _source_line(task: ReproTask) -> str:
    """One-line description of where the workspace came from."""
    if task.repo_url:
        return f"Source: {task.repo_url} (cloned)"
    if task.copy_from:
        return f"Source: local worktree copy of {task.copy_from}"
    if task.external_repo_path:
        return f"Source: {task.external_repo_path} (shared — operated in place)"
    return "Source: existing workspace repository (resume)"


SETUP_ONLY_DIRECTIVE = """

## Setup-only task

This is an environment-provisioning task. Install dependencies
(environment.yml/requirements.txt/pyproject.toml as needed, aligned with the
mirror policy and GPU capability), then run audit_env, and finish with a
summary of the prepared environment.

The runner enforces a deterministic command whitelist (independent of
stage_hint): only package installation (pip / conda install / conda env
update), inspection (ls/cat/grep/head/tail/find/rg/sed/wc/pwd/which),
--help probes, python -m py_compile, and setup chores (git/mkdir/cp/mv/
ln/wget/curl/echo) are allowed. Compound shell commands (&&, ||, ;, |,
$(), newlines) are rejected — exactly one command per list item. Direct
script or module execution (python *.py, ./run.sh, bash x.sh, make ...)
and arbitrary python -c inline programs are blocked — experiments cannot
run here. Import/version probing is done via the audit_env tool.

A successful audit_env is REQUIRED before finish; without it the run ends
failed."""


def _env_line(env: EnvironmentInfo) -> str:
    if env.created:
        return f"Conda env: {env.env_name} (freshly created — empty, only Python)"
    return f"Conda env: {env.env_name} (reused — may have packages)"


def _cache_line(task: ReproTask) -> str:
    """Deprecated: superseded by dataset_cache.render_dataset_block."""
    if task.dataset_cache_dir:
        return f"\nDataset cache: {task.dataset_cache_dir}\n"
    return ""


def _mirror_block(task: ReproTask) -> str:
    """Render the mirror policy block with profile-specific guidance."""
    profile = task.mirror_profile
    if profile == "none":
        return "Mirror policy: none."
    strict = "strict" if task.mirror_strict else "preferred"
    lines = [
        f"Mirror policy: {profile} ({strict}).",
        "For pip: use -i https://mirrors.aliyun.com/pypi/simple",
        "Avoid --index-url https://download.pytorch.org/whl/ — overrides domestic mirrors.",
        "Install GPU framework BEFORE pip install -e .",
    ]
    if profile == "autodl":
        lines.append("Prefer plain pip pins (torch==2.6.0 torchvision==0.21.0). Only use -f aliyun pytorch-wheels for +cuXXX wheels.")
    if task.mirror_strict:
        lines.append("Strict: report missing packages instead of falling back to official indexes.")
    return "\n".join(lines)


def _compact_audit(audit) -> str:
    """Format audit details as compact bullet lines."""
    if not audit.details:
        return "No audit data."
    return "\n".join(f"- {d}" for d in audit.details)


def _compact_step_line(step: AgentObservation, policy: ContextPolicy) -> str:
    """Single-line compact summary of a step."""
    tag = step.stage_hint or step.action
    if step.error:
        return f"- Step {step.step} {step.action}({tag}): ERROR — {step.error[:200]}"
    if step.command_results:
        codes = ", ".join(f"{'OK' if r.exit_code == 0 else 'FAIL'}" for r in step.command_results)
        snippets = " | ".join(_command_snippet(r, policy) for r in step.command_results)
        return f"- Step {step.step} {step.action}({tag}): [{codes}] {snippets}"
    if step.audit:
        return f"- Step {step.step} audit_env: {'OK' if step.audit.success else 'FAILED'}"
    if step.coding_result:
        return f"- Step {step.step} call_coding_agent: {step.coding_result.status}"
    return f"- Step {step.step} {step.action}({tag})"


def _command_snippet(result, policy: ContextPolicy) -> str:
    """Brief command summary with tail of output."""
    cmd_short = result.command[:80]
    tail = ""
    for path in (result.stderr_path, result.stdout_path):
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                tail = text.strip()[-policy.observation_tail:]
                break
    if tail:
        return f"{cmd_short}: {tail}"
    return f"{cmd_short} (exit={result.exit_code})"


def _format_step_full(step: AgentObservation, policy: ContextPolicy) -> str:
    """Full output for the most recent step."""
    if step.error:
        return f"Action: {step.action}({step.stage_hint})\nError: {step.error}"
    if step.command_results:
        lines = [f"Action: {step.action}({step.stage_hint})"]
        for r in step.command_results:
            tag = "OK" if r.exit_code == 0 else "FAIL"
            lines.append(f"$ {r.command}")
            lines.append(f"exit={r.exit_code} duration={r.duration_seconds}s [{tag}]")
            for label, path in [("stdout", r.stdout_path), ("stderr", r.stderr_path)]:
                if path.exists():
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        lines.append(f"--- {label} ({len(text)} bytes) ---")
                        lines.append(text[-policy.observation_tail * 3:])
        return "\n".join(lines)
    if step.audit:
        return f"audit_env: {'PASSED' if step.audit.success else 'FAILED'}\n" + "\n".join(f"- {d}" for d in (step.audit.details or []))
    if step.coding_result:
        cr = step.coding_result
        return f"call_coding_agent: {cr.status}\nSummary: {cr.summary}\nChanged: {', '.join(cr.changed_files) or 'none'}"
    return f"Action: {step.action}({step.stage_hint})"
