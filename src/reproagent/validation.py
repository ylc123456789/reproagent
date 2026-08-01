"""Lightweight final experiment-plan validation."""
from __future__ import annotations

import re

from .models import CommandPlan, ReproState

GUESS_MARKERS = ("assume", "likely", "typical", "probably", "we assume", "may print")
LOG_REDIRECTION_MARKERS = (" tee ", "| tee", "2>&1", " >", " >>")
LOSS_OUTPUT_MARKERS = ("logger.info", "print(", "logging.info")
EXPERIMENT_ENTRYPOINT_RE = r"[A-Za-z0-9_./-]+\.py"


def validate_experiment_plan(state: ReproState, plan: CommandPlan) -> CommandPlan:
    """Return a copy of plan annotated with obvious validation issues.

    This is intentionally conservative: it does not try to prove a plan correct.
    It catches common high-signal mismatches between the user goal, probe
    evidence, and final command plan.
    """
    issues: list[str] = []
    goal = state.task.experiment_goal.lower()
    command_text = "\n".join(plan.commands).lower()
    plan_text = "\n".join([plan.summary, *plan.assumptions, plan.stop_reason or ""]).lower()
    probe_text = _probe_text(state).lower()
    coding_agent_text = _coding_agent_text(state).lower()
    evidence_text = "\n".join([probe_text, coding_agent_text])

    if _mentions_any(goal, ("bounded", "short", "small", "one epoch", "few epoch")) and not _mentions_any(
        command_text,
        ("--epoch", "--epochs", "--nepochs", "max_epoch", "max_epochs", "max_steps", "num_train_epochs", "iters", "iterations"),
    ):
        issues.append("Goal asks for a bounded run, but the final commands do not set an explicit training budget such as epochs/steps.")

    if "gpu" in goal and not _has_gpu_execution_evidence(command_text, plan_text, evidence_text):
        issues.append("Goal asks for GPU execution, but the final plan does not show command-level GPU use or probe evidence that the entry point defaults to CUDA/GPU.")

    if "loss" in goal and _loss_logging_is_uncertain(evidence_text):
        issues.append("Goal asks for training loss, but probe evidence does not show that the unmodified script prints/logs loss; mark this as needs_patch or explicitly report the metric as unavailable.")

    if _mentions_any(plan_text, GUESS_MARKERS):
        issues.append("Plan uses guess language such as assume/likely/typical; replace guesses with probe evidence or mark the uncertain requirement as blocked/needs_patch.")

    if _mentions_any(command_text, LOG_REDIRECTION_MARKERS) or command_text.startswith("cd ") or "\ncd " in command_text:
        issues.append("Final commands should not use cd, tee, or shell log redirection; the runner already sets cwd and captures stdout/stderr.")

    if _experiment_commands_are_only_setup_or_inspection(plan.commands):
        issues.append("Experiment plan only contains dependency/setup or inspection commands; final experiment commands must directly run training, evaluation, demo, or another goal-directed experiment entry point.")

    missing_entrypoint = _missing_goal_entrypoint(goal, command_text)
    if missing_entrypoint:
        issues.append(f"Experiment goal names `{missing_entrypoint}`, but final commands do not run that entry point.")

    if not issues:
        return plan

    existing_inputs = [item for item in plan.needs_user_input if item not in issues]
    stop_reason = (plan.stop_reason or "").strip()
    validation_reason = "Plan validation issues: " + " ".join(issues)
    return plan.model_copy(update={
        "feasibility": _downgraded_feasibility(plan, issues),
        "needs_user_input": existing_inputs + issues,
        "stop_reason": f"{stop_reason}\n{validation_reason}".strip(),
    })


def annotate_plan_with_validation_issues(plan: CommandPlan, issues: list[str]) -> CommandPlan:
    """Apply validation issues from a secondary reviewer such as an LLM."""
    if not issues:
        return plan
    existing_inputs = [item for item in plan.needs_user_input if item not in issues]
    stop_reason = (plan.stop_reason or "").strip()
    validation_reason = "Semantic validation issues: " + " ".join(issues)
    return plan.model_copy(update={
        "feasibility": _downgraded_feasibility(plan, issues),
        "needs_user_input": existing_inputs + issues,
        "stop_reason": f"{stop_reason}\n{validation_reason}".strip(),
    })


def _has_gpu_execution_evidence(command_text: str, plan_text: str, probe_text: str) -> bool:
    """Return whether has gpu execution evidence."""
    evidence = "\n".join([command_text, plan_text, probe_text])
    if _mentions_any(evidence, ("--gpu", "--device", "cuda", "torch.cuda", "device = torch.device", "cuda:", "gpu gpu")):
        return True
    if "default=0" in evidence and "gpu" in evidence and "torch.cuda.is_available" in evidence:
        return True
    return False


def _probe_text(state: ReproState) -> str:
    """Collect probe evidence for validation."""
    chunks: list[str] = []
    for attempt in state.probe_attempts:
        chunks.append(attempt.plan.summary)
        chunks.extend(attempt.plan.assumptions)
        for result in attempt.results:
            chunks.append(result.command)
            for path in (result.stdout_path, result.stderr_path):
                if path.exists():
                    chunks.append(path.read_text(encoding="utf-8", errors="replace")[-5000:])
    return "\n".join(chunks)


def _coding_agent_text(state: ReproState) -> str:
    """Collect CodingAgent evidence for validation."""
    chunks: list[str] = []
    for result in state.coding_agent_results:
        chunks.append(result.status)
        chunks.append(result.summary)
        chunks.extend(result.changed_files)
        chunks.extend(result.residual_risks)
        for path in (result.diff_path, result.report_path):
            if path and path.exists():
                chunks.append(path.read_text(encoding="utf-8", errors="replace")[-5000:])
    return "\n".join(chunks)


def _loss_logging_is_uncertain(probe_text: str) -> bool:
    """Return whether loss logging evidence is uncertain."""
    lines = probe_text.splitlines()
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "loss" in lowered and _mentions_any(lowered, LOSS_OUTPUT_MARKERS):
            return False

        if _mentions_any(lowered, LOSS_OUTPUT_MARKERS):
            following_block = "\n".join(lines[index:index + 6]).lower()
            if "loss" in following_block:
                return False

    lowered_text = probe_text.lower()
    if "loss_meter" in lowered_text and _mentions_any(lowered_text, LOSS_OUTPUT_MARKERS):
        return False
    return True


def _experiment_commands_are_only_setup_or_inspection(commands: list[str]) -> bool:
    """Return whether experiment commands are only setup or inspection."""
    meaningful = [command.strip() for command in commands if command.strip()]
    if not meaningful:
        return False
    return all(_is_setup_or_inspection_command(command) for command in meaningful)


def _is_setup_or_inspection_command(command: str) -> bool:
    """Return whether a command is setup or inspection only."""
    lowered = command.strip().lower()
    if lowered.startswith(("pip install", "python -m pip install", "python3 -m pip install", "conda install", "mamba install")):
        return True
    if lowered.startswith(("python -c", "python3 -c")):
        inspection_markers = ("import ", "print(", "__version__", "cuda.is_available", "device_count")
        return any(marker in lowered for marker in inspection_markers)
    if "--help" in lowered or lowered.endswith(" -h") or " -h " in lowered:
        return True
    if lowered.startswith(("ls", "find", "rg", "grep", "sed", "cat", "head", "tail", "wc", "pwd")):
        return True
    if lowered.startswith(("python -m py_compile", "python3 -m py_compile")):
        return True
    return False


def _missing_goal_entrypoint(goal: str, command_text: str) -> str | None:
    """Find a goal entry point missing from final commands."""
    for match in re.finditer(EXPERIMENT_ENTRYPOINT_RE, goal):
        entrypoint = match.group(0).strip("`'\".,;:)")
        if entrypoint and entrypoint.lower() not in command_text:
            return entrypoint
    return None


def _downgraded_feasibility(plan: CommandPlan, issues: list[str]) -> str:
    """Choose feasibility after validation issues."""
    lowered = " ".join(issues).lower()
    if "loss" in lowered:
        return "needs_patch"
    if "cd, tee" in lowered or "shell log redirection" in lowered:
        return "blocked"
    return "needs_config"


def _mentions_any(text: str, markers: tuple[str, ...]) -> bool:
    """Return whether text contains any marker."""
    return any(marker in text for marker in markers)
