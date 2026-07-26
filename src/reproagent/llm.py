"""LLM planning and review."""
from __future__ import annotations

import json
import os
import urllib.request

from .models import CommandPlan, ReproState

SYSTEM = """You are a careful machine-learning reproducibility engineer.
The final goal is faithful reproduction of the paper's reported experiments and results, not merely a smoke test.
Return strict JSON only for command plans. Prefer safe inspection, evaluation, demo, or test commands before full training, but configure the environment for the final reproduction target.
Environment-stage commands must only install dependencies or run quick import/version/device checks; do not run repository demos, examples, training scripts, or long evaluations during environment setup.
If an NVIDIA GPU is visible, prioritize a GPU-capable ML environment and choose dependency builds compatible with the reported driver/CUDA capability. Do not blindly install the newest PyTorch/JAX/TensorFlow build.
Do not suggest destructive commands. If data or checkpoints are missing, say so.
Important: the system already runs every command inside a prepared conda environment. Do not use `conda activate`, `conda create`, or `conda run` in your commands.
"""


def plan_environment(state: ReproState) -> CommandPlan:
    prompt = _base_context(state) + """

Plan dependency setup commands needed inside the already-created conda environment for the final goal: faithful reproduction of the paper's experiments/results.
Return JSON with fields: stage, summary, commands, assumptions, stop_reason.
Use stage='environment'. Prefer project-provided files such as requirements.txt, pyproject.toml, setup.py, or environment.yml when appropriate, but do not blindly accept an unconstrained latest GPU framework package if it conflicts with the detected hardware.
If an NVIDIA GPU is visible and the project uses PyTorch/JAX/TensorFlow, configure a GPU-capable build compatible with the reported driver/CUDA capability. For PyTorch, inspect the repo pins first; if the repo only says something broad like torch>=x, choose an official build compatible with this machine instead of defaulting to the newest CUDA build.
For older PyTorch builds, avoid incompatible NumPy 2.x ABI issues; pin numpy<2 when the selected torch build or logs indicate NumPy 1.x compatibility.
Do not create or activate conda environments. Commands already run inside the prepared conda environment.
Do not run repo examples, demos, train scripts, MNIST scripts, or long evaluations in the environment stage. Use only quick checks such as `python -c "import torch; print(torch.cuda.is_available())"` or a tiny inline tensor operation expected to finish in seconds. Put real reproduction/demo/evaluation commands in the experiment stage after audit passes.
If using `python -c`, do not define a `def` function on a semicolon-separated one-liner; use a lambda or a short import/device check instead.
If no setup is needed, commands may be empty.
"""
    return _complete_plan(state, prompt, stage="environment")


def plan_experiment(state: ReproState) -> CommandPlan:
    prompt = _base_context(state) + _recent_logs(state) + """

Plan the next experiment/demo/evaluation commands to try.
Return JSON with fields: stage, summary, commands, assumptions, stop_reason.
Use stage='experiment'. Prefer bounded commands that can produce a metric or verify the repo runs with the configured hardware.
For the default run, choose a short reproduction smoke/evaluation expected to finish in a few minutes. Prefer, in order: (1) a tiny inline GPU workload that imports the project and performs one minimal operation, (2) one small targeted test file, (3) one documented small demo with bounded runtime.
Avoid aggregate test suites such as tests/run_all.py, bare pytest over the whole repo, dataset downloads, MNIST/full training, multi-epoch training, or long examples unless there is no smaller valid reproduction step.
If full paper reproduction requires expensive training or datasets, do not start it by default; describe the required command, data, expected runtime, and assumptions in stop_reason/assumptions so the report records the next step.
Do not use conda activate; commands already run inside the prepared conda environment.
"""
    return _complete_plan(state, prompt, stage="experiment")


def revise_after_failure(state: ReproState, stage: str) -> CommandPlan:
    prompt = _base_context(state) + _recent_logs(state) + f"""

The previous {stage} attempt failed or the environment audit required repair. Diagnose the logs and propose a revised {stage} command plan.
Return JSON with fields: stage, summary, commands, assumptions, stop_reason.
If the audit says GPU repair is required, fix the installed ML framework build so GPU is available on this hardware, unless the repo clearly does not use that framework. Prefer explicit uninstall/reinstall commands for incompatible packages when needed.
If the audit says NumPy ABI repair is required, pin or downgrade NumPy, commonly `pip install "numpy<2"`, then rerun a quick import/device check.
For environment-stage revisions, do not run repository demos/examples/training. Validate with quick import/version/device checks only; real demos belong in the experiment stage after audit passes.
For experiment-stage revisions, prefer shorter bounded checks after a timeout, such as a tiny inline operation or a single targeted test file. Do not escalate to aggregate test suites, dataset downloads, or training jobs unless explicitly necessary.
If using `python -c`, do not define a `def` function on a semicolon-separated one-liner; use a lambda or a short import/device check instead.
Do not use conda activate; commands already run inside the prepared conda environment.
"""
    return _complete_plan(state, prompt, stage=stage)


def final_review(state: ReproState) -> str:
    if state.task.mock_llm:
        return _mock_final_review(state)
    prompt = _base_context(state) + _recent_logs(state) + """

Write a concise reproduction result summary. Explain what worked, what failed, whether any metrics were found, whether the run used GPU or CPU fallback, and what human input is needed next. Return plain Markdown.
"""
    return _openai_compatible_text(state, prompt)


def _complete_plan(state: ReproState, prompt: str, stage: str) -> CommandPlan:
    if state.task.mock_llm:
        return _mock_plan(stage)
    text = _openai_compatible_text(state, prompt)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM did not return JSON: {text[:500]}") from exc
    returned_stage = data.get("stage", stage)
    if returned_stage not in {"environment", "experiment"}:
        returned_stage = stage
    return CommandPlan(
        stage=returned_stage,
        summary=str(data.get("summary", "")),
        commands=_as_list(data.get("commands", [])),
        assumptions=_as_list(data.get("assumptions", [])),
        stop_reason=data.get("stop_reason"),
    )


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _openai_compatible_text(state: ReproState, prompt: str) -> str:
    api_key = os.environ.get(state.task.api_key_env)
    if not api_key:
        raise RuntimeError(f"{state.task.api_key_env} is not set. Use --mock-llm for local testing.")
    model = state.task.model or "gpt-4.1-mini"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    url = _chat_completions_url(state.task.api_base)
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
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _base_context(state: ReproState) -> str:
    ctx = state.repo_context
    if ctx is None:
        return "No repo context available."
    env = state.environment
    env_text = f"Conda env: {env.env_name}" if env else "Conda env: not prepared yet"
    return f"""
Paper URL: {state.task.paper_url}
Repo URL: {state.task.repo_url}
Repo path: {ctx.repo_path}
Commit: {ctx.commit_hash}
{env_text}

Hardware context:
{ctx.hardware_text}

File tree:
{ctx.file_tree}

README/docs excerpt:
{ctx.readme_text[:16000]}
"""


def _recent_logs(state: ReproState, max_chars: int = 12000) -> str:
    chunks: list[str] = []
    if state.environment_audit:
        chunks.append("\n## latest environment audit\n")
        chunks.append(f"summary={state.environment_audit.summary}\n")
        chunks.append(f"requires_repair={state.environment_audit.requires_repair}\n")
        if state.environment_audit.details:
            details = "\n".join(f"- {x}" for x in state.environment_audit.details)
            chunks.append(f"details:\n{details}\n")
        for label, path in [("audit stdout", state.environment_audit.stdout_path), ("audit stderr", state.environment_audit.stderr_path)]:
            if path and path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")
                chunks.append(f"--- {label} ---\n{text[-3000:]}\n")
    attempts = state.environment_attempts[-2:] + state.experiment_attempts[-2:]
    for attempt in attempts:
        chunks.append(f"\n## {attempt.stage} attempt {attempt.attempt}: {attempt.plan.summary}\n")
        for result in attempt.results[-3:]:
            chunks.append(f"$ {result.command}\nexit={result.exit_code}\n")
            for label, path in [("stdout", result.stdout_path), ("stderr", result.stderr_path)]:
                if path.exists():
                    text = path.read_text(encoding="utf-8", errors="replace")
                    chunks.append(f"--- {label} ---\n{text[-3000:]}\n")
    return "\n".join(chunks)[-max_chars:]

def _mock_plan(stage: str) -> CommandPlan:
    if stage == "environment":
        return CommandPlan(stage="environment", summary="Mock environment: no setup commands.", commands=[])
    return CommandPlan(stage="experiment", summary="Mock experiment: inspect Python entry points.", commands=["python3 --version"])


def _mock_final_review(state: ReproState) -> str:
    return "Mock LLM review: the run completed in mock mode. Inspect logs and state.json for command results."
