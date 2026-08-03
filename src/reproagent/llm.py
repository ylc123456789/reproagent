"""LLM planning and review."""
from __future__ import annotations

import json
import re
import os
from datetime import datetime
import urllib.request

from .models import CommandPlan, ReproState
from .text import normalize_plan_text, normalize_text, normalize_text_list

FEASIBILITIES = {"ready_to_run", "needs_config", "needs_patch", "blocked", "unsafe_or_too_expensive"}

SYSTEM = """You are a careful machine-learning reproducibility engineer.
The final goal is faithful reproduction of the paper's reported experiments and results, not merely a smoke test.
Return strict JSON only for command plans. Prefer safe inspection, evaluation, demo, or test commands before full training, but configure the environment for the final reproduction target.
Command-plan JSON fields are: stage, summary, commands, assumptions, feasibility, expected_runtime, needs_user_input, stop_reason. Use feasibility values: ready_to_run, needs_config, needs_patch, blocked, unsafe_or_too_expensive.
Environment-stage commands must only install dependencies or run quick import/version/device checks; do not run repository demos, examples, training scripts, or long evaluations during environment setup.
If an NVIDIA GPU is visible, prioritize a GPU-capable ML environment and choose dependency builds compatible with the reported driver/CUDA capability. Do not blindly install the newest PyTorch/JAX/TensorFlow build.
Do not suggest destructive commands. If data or checkpoints are missing, say so.
Important: the system already runs every command inside the repository root and inside a prepared conda environment. Do not use `cd`, `tee`, shell output redirection for logs, `conda activate`, `conda create`, or `conda run` in your commands. The runner captures stdout/stderr and writes logs.
"""


def plan_environment(state: ReproState) -> CommandPlan:
    """Ask the LLM for dependency setup commands."""
    prompt = _base_context(state) + """

Plan dependency setup commands needed inside the conda environment for the experiment goal and final reproduction target.
Return JSON with fields: stage, summary, commands, assumptions, stop_reason.
Use stage='environment'. The conda environment status is shown above — if freshly created, it contains only Python; you must install every dependency from scratch. Always run pip install / package setup commands BEFORE any import or version checks. A failed import check does not mean the environment setup failed; it means you tried to import before installing.
Prefer project-provided files such as requirements.txt, pyproject.toml, setup.py, or environment.yml when appropriate, but do not blindly accept an unconstrained latest GPU framework package if it conflicts with the detected hardware.
If an NVIDIA GPU is visible and the project uses PyTorch/JAX/TensorFlow, configure a GPU-capable build compatible with the reported driver/CUDA capability. For PyTorch, inspect the repo pins first; if the repo only says something broad like torch>=x, choose a build compatible with this machine instead of defaulting to the newest CUDA build. Follow the mirror policy from the run context when choosing package indexes or find-links.
For older PyTorch builds, avoid incompatible NumPy 2.x ABI issues; pin numpy<2 when the selected torch build or logs indicate NumPy 1.x compatibility.
Quote shell-sensitive pip version specifiers such as `numpy<2`, `torch>=2`, or `package[extra]` in commands, for example `pip install "numpy<2" scipy`.
Do not create or activate conda environments. Commands already run inside the prepared conda environment.
Do not run repo examples, demos, train scripts, MNIST scripts, or long evaluations in the environment stage. Use only quick checks such as `python -c "import torch; print(torch.cuda.is_available())"` or a tiny inline tensor operation expected to finish in seconds. Put real reproduction/demo/evaluation commands in the experiment stage after audit passes.
If using `python -c`, do not define a `def` function on a semicolon-separated one-liner; use a lambda or a short import/device check instead.
If no setup is needed, commands may be empty.
"""
    return _complete_plan(state, prompt, stage="environment")


def plan_probe(state: ReproState) -> CommandPlan:
    """Ask the LLM for safe repository probe commands."""
    prompt = _base_context(state) + _recent_logs(state) + """

Plan safe probe commands to discover the experiment interface before any real training/evaluation.
Return JSON with fields: stage, summary, commands, assumptions, feasibility, expected_runtime, needs_user_input, stop_reason.
Use stage='probe'. Commands must only inspect files, print help, list configs, or run tiny inline checks. Good examples: `python train.py --help`, `python examples/odenet_mnist.py --help`, `find . -maxdepth 3 -iname '*.yaml'`, `sed -n '1,160p' configs/foo.yaml`.
Do not run training, evaluation, demos, tests over datasets, downloads, or scripts without a help/config/listing flag in the probe stage.
The purpose is to identify entry points, configurable parameters, config-file style, metric outputs, GPU flags, and constraints such as hidden_size divisibility or checkpoint compatibility.
"""
    return _complete_plan(state, prompt, stage="probe")


def plan_experiment(state: ReproState) -> CommandPlan:
    """Ask the LLM for the formal experiment command plan."""
    prompt = _base_context(state) + _recent_logs(state) + """

Plan the final experiment/demo/evaluation commands to execute for the experiment goal, using the probe logs above.
Return JSON with fields: stage, summary, commands, assumptions, feasibility, expected_runtime, needs_user_input, stop_reason.
Use stage='experiment'. The experiment goal is the contract for this run: choose commands that directly attempt it, or explain clearly in assumptions/stop_reason why the goal is blocked, too expensive, or requires config/code changes first.
Do not assume default script parameters satisfy the goal. If the goal says bounded, short, one epoch, GPU, specific dataset, or specific metric, translate that into explicit CLI args or a generated config path whenever the repo interface supports it.
Prefer repo-provided scripts and documented arguments when available. If the repo uses YAML/JSON config files, prefer creating or referencing a run-specific config in the workspace rather than editing upstream defaults. If a goal requires modifying original project code, set feasibility='needs_patch' and explain the minimal patch needed instead of silently changing behavior.
Check parameter constraints before proposing hyperparameter changes. For architecture-coupled settings such as attention heads, hidden sizes, checkpoint shapes, image sizes, or class counts, state the constraint and avoid incompatible changes.
If training or dataset downloads are required by the goal, they are allowed; keep commands within the user-provided timeout budget and state expected runtime, data, GPU use, and metrics in assumptions.
Prefer commands that produce measurable evidence such as accuracy, loss, generated artifacts, or saved logs. Do not silently substitute an unrelated demo for the requested goal. If the goal asks for a metric that the discovered script does not print, set feasibility='needs_patch' or explicitly state the metric is unavailable from the unmodified repo; do not claim it can be extracted from logs.
Do not use cd, tee, shell log redirection, or conda activate; commands already run inside the repository root and prepared conda environment, and the runner captures logs.
"""
    return _complete_plan(state, prompt, stage="experiment")


def revise_after_failure(state: ReproState, stage: str) -> CommandPlan:
    """Ask the LLM to revise a stage plan after failed execution or validation."""
    previous_issues: list[str] = []
    if stage == "experiment":
        attempts = state.experiment_attempts
    elif stage == "environment":
        attempts = state.environment_attempts
    else:
        attempts = state.probe_attempts
    if attempts:
        last = attempts[-1]
        if last.plan.needs_user_input:
            previous_issues = last.plan.needs_user_input

    feedback_block = ""
    if previous_issues:
        feedback_block = "\nIssues from the previous attempt that MUST be fixed:\n"
        feedback_block += "\n".join(f"- {item}" for item in previous_issues)

    prompt = _base_context(state) + _recent_logs(state) + feedback_block + f"""

The previous {stage} attempt failed or the environment audit required repair. Diagnose the logs and propose a revised {stage} command plan.
Return JSON with fields: stage, summary, commands, assumptions, stop_reason.
If the audit says GPU repair is required, fix the installed ML framework build so GPU is available on this hardware, unless the repo clearly does not use that framework. Prefer explicit uninstall/reinstall commands for incompatible packages when needed.
If the audit says NumPy ABI repair is required, pin or downgrade NumPy, commonly `pip install "numpy<2"`, then rerun a quick import/device check.
For environment-stage revisions: first check whether the environment is already correctly set up by looking at the audit results and command logs. If imports, GPU, and all required packages are working, return an empty command plan — the environment stage is done. If something is broken, fix only that specific issue with targeted pip install/uninstall commands. Never run repository demos, examples, training scripts, or evaluation commands in the environment stage — even as a "quick test". Real experiments belong in the experiment stage after audit passes.
For experiment-stage revisions, stay anchored to the experiment goal. If previous validation flagged issues (listed above), fix each one explicitly in the new plan. If the exact goal appears too expensive or impossible within the timeout, set feasibility='blocked' or feasibility='unsafe_or_too_expensive' and explain the smallest goal-relevant diagnostic or required human decision in assumptions/stop_reason.
If using `python -c`, do not define a `def` function on a semicolon-separated one-liner; use a lambda or a short import/device check instead.
Do not use cd, tee, shell log redirection, or conda activate; commands already run inside the repository root and prepared conda environment, and the runner captures logs.
"""
    return _complete_plan(state, prompt, stage=stage)


def review_experiment_plan(state: ReproState, plan: CommandPlan) -> dict:
    """Ask the LLM to review the experiment plan against the goal and return structured feedback.

    Returns a dict with keys: ready (bool), issues (list[str]), feasibility (str).
    This single call replaces both the old hard-rule validation and semantic review.
    """
    if state.task.mock_llm:
        return {"ready": True, "issues": [], "feasibility": "ready_to_run"}

    recent = _recent_logs(state)
    prompt = _base_context(state) + recent + f"""

You are reviewing a proposed experiment plan. Your job is to catch problems BEFORE
execution, so the plan can be fixed. Return strict JSON only:

{{"ready": true or false, "issues": ["issue 1", "issue 2"], "feasibility": "ready_to_run" or "needs_config" or "needs_patch"}}

Experiment goal:
{state.task.experiment_goal}

Plan summary:
{plan.summary}

Plan commands:
{chr(10).join(f"- {command}" for command in plan.commands) or "- none"}

Plan assumptions:
{chr(10).join(f"- {item}" for item in plan.assumptions) or "- none"}

Check EVERY item below. For each check that fails, add a specific, actionable issue
to the issues list. Quote the exact command or assumption that is wrong.

At the end, classify every requirement you found in the goal as [MET], [NOT MET],
or [UNCERTAIN]. Include this classification at the top of the issues list.

## Goal Alignment

1. Does the plan directly attempt the experiment goal? If it only does inspection
   (grep/cat/head/--help), that is NOT sufficient — report it.
2. "bounded" / "short" / "few epoch" → the plan must explicitly set a SMALL
   epoch/step count (--nepochs 10 or fewer). Using the script's default value
   (e.g. --nepochs 160) is NOT bounded. Report the issue.
3. "GPU" → the command must have a GPU flag, or probe evidence must show the
   script defaults to CUDA on its own.
4. If the goal requires a specific metric (e.g. "report training loss", "report
   accuracy"), check the probe evidence: does the script actually print/log that
   metric? Do NOT assume it does. If probe evidence confirms the script does NOT
   output this metric, you MUST set feasibility to "needs_patch" — no CLI flag
   or plan revision can make the script print something it does not print.
5. "runtime" → the plan must capture elapsed time somehow (time command wrapper,
   timer printed in script output, etc).
6. If the goal mentions a specific entry point (a .py file), verify that file
   appears in the commands.

## Command Format

7. Experiment commands must NOT include --help or -h (that belongs in probe stage).
8. Commands must NOT use cd, tee, | tee, 2>&1, > or >> for log redirection
   (the runner already captures stdout/stderr).
9. Commands must NOT use conda activate or conda run (the runner wraps commands).
10. Every command must be safe: no sudo, no rm -rf, no shutdown/reboot, no curl|bash,
    no parent-directory traversal.

## CLI Flag Correctness

11. Compare every --flag in the commands against the probe --help output. If a flag
    does not appear in the help text, report it and suggest the correct alternative.
12. If a flag needs a value but is missing one, report it.

## Feasibility

- Use "ready_to_run" when all checks pass and every goal requirement is [MET].
- Use "needs_config" when issues can be fixed by revising the plan (format errors,
  wrong flags, missing budget, missing GPU evidence, wrong epoch count, etc.).
- Use "needs_patch" when the project code must be modified to satisfy the goal
  (e.g. a required metric is not printed by the script and no CLI flag can add it).
  If probe evidence confirms a required metric is missing from the script output,
  needs_patch is the ONLY correct choice — do not downgrade to needs_config.
- The system will retry on both "needs_config" and "needs_patch", so do not be
  afraid to report issues. Be strict — a false pass is worse than a false flag.

If ready is true, issues must be empty and feasibility must be "ready_to_run".
"""
    text = _call_llm_text(state, prompt, trace_label="experiment_plan_review")
    data = _loads_plan_json(text)
    issues = normalize_text_list(_as_list(data.get("issues", []), drop_false=True))
    feasibility = data.get("feasibility", "needs_config")
    if feasibility not in FEASIBILITIES:
        feasibility = "needs_config"
    ready = bool(data.get("ready", False))
    if ready and issues:
        ready = False
        if feasibility == "ready_to_run":
            feasibility = "needs_config"
    return {"ready": ready, "issues": issues, "feasibility": feasibility}


def apply_review_to_plan(plan: CommandPlan, review: dict) -> CommandPlan:
    """Apply a review result to a command plan, updating feasibility and issues."""
    if review.get("ready"):
        return plan.model_copy(update={"feasibility": "ready_to_run", "needs_user_input": []})
    messages = review.get("issues", [])
    existing = [item for item in (plan.needs_user_input or []) if item not in messages]
    feasibility = review.get("feasibility") or "needs_config"
    return plan.model_copy(update={
        "feasibility": feasibility,
        "needs_user_input": existing + messages,
    })


def final_review(state: ReproState) -> str:
    """Write the final LLM review for a run."""
    if state.task.mock_llm:
        return _mock_final_review(state)
    prompt = _base_context(state) + _recent_logs(state) + """

Write a reproduction result report in Markdown with the sections below.
Return plain Markdown only — no JSON wrapper.

## Required sections

### Summary
One paragraph: what was the goal, what ran, did it succeed or partially succeed,
did the run use GPU or CPU, were any code patches applied.

### Metrics
A table or list covering EVERY requirement in the experiment goal. For each one,
state the actual value obtained (or "not available"), which log file the value
came from, and a [MET] / [NOT MET] / [UNCERTAIN] label. If a metric was computed
by the script but not printed to any output, mark it [NOT MET] and explain why.
Quote the exact log line or file path as evidence. Example format:

| Requirement | Status | Value | Evidence |
|-------------|--------|-------|----------|
| test accuracy | [MET] | 99.05% | logs/experiment_02_02.stderr L10 |
| training loss | [NOT MET] | — | Script computes loss for backprop but does not log it |

### Deviations
List every way the experiment differed from the paper or goal requirements
(epoch count, flags, hardware, solver settings, etc.). Be explicit.

### Data Index
A bullet list of EVERY log file produced by this run with its path relative to
the workspace and a one-line description of what it contains. Group by stage
(environment, probe, experiment, patches). Example:

- logs/experiment_02_02.stderr — per-epoch training output with accuracy and timing
- logs/experiment_02_02.stdout — experiment stdout (empty; script writes to stderr)
- patches/coding_agent_01/diff.patch — code changes applied (if any)

### Next Steps
What a human should do next: rerun with different flags, apply a patch for a
missing metric, run the full training schedule, etc.
"""
    return normalize_text(_call_llm_text(state, prompt, trace_label="final_review"))


def _complete_plan(state: ReproState, prompt: str, stage: str) -> CommandPlan:
    """Call the LLM and normalize a command plan."""
    if state.task.mock_llm:
        return _mock_plan(stage)
    text = _call_llm_text(state, prompt, trace_label=f"{stage}_plan")
    data = _loads_plan_json(text)
    returned_stage = data.get("stage", stage)
    if returned_stage not in {"environment", "probe", "experiment"}:
        returned_stage = stage
    feasibility = data.get("feasibility")
    if feasibility not in FEASIBILITIES:
        feasibility = None
    return CommandPlan(
        stage=returned_stage,
        summary=normalize_text(str(data.get("summary", ""))),
        commands=normalize_text_list(_as_list(data.get("commands", []))),
        assumptions=normalize_text_list(_as_list(data.get("assumptions", []))),
        feasibility=feasibility,
        expected_runtime=normalize_plan_text(data.get("expected_runtime")),
        needs_user_input=normalize_text_list(_as_list(data.get("needs_user_input", []), drop_false=True)),
        stop_reason=normalize_plan_text(data.get("stop_reason")),
    )



def _loads_plan_json(text: str) -> dict:
    """Parse JSON returned by the LLM."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        repaired = _escape_invalid_json_backslashes(text)
        if repaired != text:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"LLM did not return JSON: {text[:500]}") from first_error


def _escape_invalid_json_backslashes(text: str) -> str:
    """Repair invalid JSON backslash escapes."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

def _as_list(value, drop_false: bool = False) -> list[str]:
    """Normalize a value into a list of strings."""
    if value is None or (drop_false and value is False):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _call_llm_text(state: ReproState, prompt: str, trace_label: str) -> str:
    """Call the configured LLM while preserving legacy test doubles."""
    try:
        return _openai_compatible_text(state, prompt, trace_label=trace_label)
    except TypeError as exc:
        if "trace_label" not in str(exc):
            raise
        return _openai_compatible_text(state, prompt)


def _openai_compatible_text(state: ReproState, prompt: str, trace_label: str | None = None) -> str:
    """Call an OpenAI-compatible chat completion API."""
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
    text = data["choices"][0]["message"]["content"].strip()
    if trace_label:
        _write_llm_trace(state, trace_label, prompt, text)
    return text


def _write_llm_trace(state: ReproState, trace_label: str, prompt: str, response: str) -> None:
    """Persist raw LLM prompt/response pairs for debugging."""
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_label).strip("_") or "llm"
    logs_dir = state.task.workspace_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    prefix = logs_dir / f"llm_{stamp}_{safe_label}"
    (prefix.with_suffix(".prompt.txt")).write_text(prompt, encoding="utf-8")
    (prefix.with_suffix(".response.txt")).write_text(response, encoding="utf-8")


def _chat_completions_url(api_base: str) -> str:
    """Build the chat completions endpoint URL."""
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _base_context(state: ReproState) -> str:
    """Render shared context for LLM prompts."""
    ctx = state.repo_context
    if ctx is None:
        return "No repo context available."
    env = state.environment
    if env:
        env_text = f"Conda env: {env.env_name}"
        if env.created:
            env_text += " (freshly created — contains only Python, no packages installed yet)"
        else:
            env_text += " (reused from a previous run — may already have packages installed)"
    else:
        env_text = "Conda env: not prepared yet"
    return f"""
Paper URL: {state.task.paper_url}
Repo URL: {state.task.repo_url}
Experiment goal: {state.task.experiment_goal or 'No explicit experiment goal provided.'}
Timeout budget per command: {state.task.timeout_seconds}s
Repo path: {ctx.repo_path}
Commit: {ctx.commit_hash}
{env_text}
{_mirror_context(state)}

Hardware context:
{ctx.hardware_text}

File tree:
{ctx.file_tree}

README/docs excerpt:
{ctx.readme_text[:16000]}
"""


def _mirror_context(state: ReproState) -> str:
    """Render dependency mirror guidance for prompts."""
    profile = state.task.mirror_profile
    if profile == "none":
        return "Mirror policy: none. Use the repository instructions, the current pip/conda configuration, or official package indexes as appropriate."

    strict = "strict" if state.task.mirror_strict else "preferred"
    lines = [
        f"Mirror policy: {profile} ({strict}). Prefer the configured mirror profile for dependency downloads.",
        "For ordinary pip packages, prefer the server's configured pip mirror; if adding an explicit pip index, prefer `-i https://mirrors.aliyun.com/pypi/simple`.",
        "Avoid `--index-url https://download.pytorch.org/whl/...` because it overrides domestic/server pip mirrors and often downloads slowly from international PyTorch hosts.",
        "Install the intended GPU ML framework before `pip install -e .` when the repo has broad dependencies like torch>=x, so editable install does not pull an incompatible/latest framework build.",
    ]
    if profile == "autodl":
        lines += [
            "AutoDL official guidance: first prefer an AutoDL image that already includes the required PyTorch/TensorFlow version. If installing PyTorch manually, prefer plain pip pins such as `pip install torch==2.6.0 torchvision==0.21.0` and remove PyTorch official `-f`/`--index-url` options so installation uses the domestic pip source configured on the instance.",
            "Only use Aliyun PyTorch wheel find-links such as `-f https://mirrors.aliyun.com/pytorch-wheels/cu124/` when an explicit local-version wheel like `torch==2.6.0+cu124` is required and available from that mirror.",
            "Keep conda envs/caches on the data disk when already configured on AutoDL-style servers.",
        ]
    elif profile == "cn":
        lines.append("For explicit PyTorch local-version CUDA wheels, prefer a domestic find-links mirror such as `-f https://mirrors.aliyun.com/pytorch-wheels/cu124/` when the matching CUDA wheel page exists.")
    if state.task.mirror_strict:
        lines.append("Strict mirror mode: if a required package or GPU wheel is unavailable from the preferred mirror/profile, set feasibility='needs_config' or 'blocked' and explain the missing mirror instead of silently falling back to official international indexes.")
    return "\n".join(lines)

def _recent_logs(state: ReproState, max_chars: int = 12000) -> str:
    """Render recent logs for LLM prompts."""
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
    if state.coding_agent_results:
        latest = state.coding_agent_results[-1]
        chunks.append("\n## latest CodingAgent result\n")
        chunks.append(f"status={latest.status}\n")
        if latest.summary:
            chunks.append(f"summary={latest.summary}\n")
        if latest.changed_files:
            chunks.append("changed_files:\n" + "\n".join(f"- {path}" for path in latest.changed_files) + "\n")

    attempts = state.environment_attempts[-2:] + state.probe_attempts[-2:] + state.experiment_attempts[-2:]
    for attempt in attempts:
        chunks.append(f"\n## {attempt.stage} attempt {attempt.attempt}: {attempt.plan.summary}\n")
        chunks.append(f"feasibility={attempt.plan.feasibility or 'unknown'}\n")
        if attempt.plan.commands:
            chunks.append("planned commands:\n" + "\n".join(f"- {command}" for command in attempt.plan.commands) + "\n")
        if attempt.plan.needs_user_input:
            chunks.append("validation/user issues:\n" + "\n".join(f"- {item}" for item in attempt.plan.needs_user_input) + "\n")
        if attempt.plan.stop_reason:
            chunks.append(f"stop_reason={attempt.plan.stop_reason}\n")
        for result in attempt.results[-3:]:
            chunks.append(f"$ {result.command}\nexit={result.exit_code}\n")
            for label, path in [("stdout", result.stdout_path), ("stderr", result.stderr_path)]:
                if path.exists():
                    text = path.read_text(encoding="utf-8", errors="replace")
                    chunks.append(f"--- {label} ---\n{text[-3000:]}\n")
    return "\n".join(chunks)[-max_chars:]

def _mock_plan(stage: str) -> CommandPlan:
    """Return a deterministic mock command plan."""
    if stage == "environment":
        return CommandPlan(stage="environment", summary="Mock environment: no setup commands.", commands=[])
    if stage == "probe":
        return CommandPlan(stage="probe", summary="Mock probe: inspect Python entry points.", commands=["python3 -c \"import sys; print(sys.version)\""], feasibility="ready_to_run")
    return CommandPlan(stage="experiment", summary="Mock experiment: inspect Python entry points.", commands=["python3 --version"], feasibility="ready_to_run", expected_runtime="seconds")


def _mock_final_review(state: ReproState) -> str:
    """Return a deterministic mock final review."""
    return "Mock LLM review: the run completed in mock mode. Inspect logs and state.json for command results."
