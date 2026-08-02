"""Lightweight final experiment-plan validation."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Literal

from .models import CommandPlan, ReproState

FeasibilityRank = Literal["ready_to_run", "needs_config", "needs_patch", "blocked", "unsafe_or_too_expensive"]

GUESS_MARKERS = ("assume", "likely", "typical", "probably", "we assume", "may print")
LOG_REDIRECTION_MARKERS = (" tee ", "| tee", "2>&1", " >", " >>")
LOSS_OUTPUT_MARKERS = ("logger.info", "print(", "logging.info")
EXPERIMENT_ENTRYPOINT_RE = r"[A-Za-z0-9_./-]+\.py"
FEASIBILITY_PRIORITY: dict[str, int] = {
    "ready_to_run": 0,
    "needs_config": 1,
    "needs_patch": 2,
    "blocked": 3,
    "unsafe_or_too_expensive": 4,
}


@dataclass(frozen=True)
class ValidationIssue:
    """Structured validation issue with a target feasibility."""

    code: str
    message: str
    feasibility: FeasibilityRank


def validate_experiment_plan(state: ReproState, plan: CommandPlan) -> CommandPlan:
    """Return a copy of plan annotated with obvious validation issues.

    This is intentionally conservative: it does not try to prove a plan correct.
    It catches common high-signal mismatches between the user goal, probe
    evidence, and final command plan.
    """
    return apply_validation_issues(plan, collect_experiment_validation_issues(state, plan), label="Plan validation issues")


def collect_experiment_validation_issues(state: ReproState, plan: CommandPlan) -> list[ValidationIssue]:
    """Collect hard-rule experiment-plan issues without mutating the plan."""
    issues: list[ValidationIssue] = []
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
        issues.append(ValidationIssue(
            "missing_training_budget",
            "Goal asks for a bounded run, but the final commands do not set an explicit training budget such as epochs/steps.",
            "needs_config",
        ))

    if "gpu" in goal and not _has_gpu_execution_evidence(command_text, plan_text, evidence_text):
        issues.append(ValidationIssue(
            "missing_gpu_evidence",
            "Goal asks for GPU execution, but the final plan does not show command-level GPU use or probe evidence that the entry point defaults to CUDA/GPU.",
            "needs_config",
        ))

    if "loss" in goal and (_loss_logging_is_uncertain(evidence_text) or _plan_admits_missing_required_loss(plan_text)):
        issues.append(ValidationIssue(
            "missing_required_loss_output",
            "Goal asks for training loss, but evidence does not show the final run will print/log loss; mark this as needs_patch instead of treating missing loss as an acceptable deviation.",
            "needs_patch",
        ))

    if _mentions_any(plan_text, GUESS_MARKERS):
        issues.append(ValidationIssue(
            "guess_language",
            "Plan uses guess language such as assume/likely/typical; replace guesses with probe evidence or mark the uncertain requirement as blocked/needs_patch.",
            "needs_config",
        ))

    if _mentions_any(command_text, LOG_REDIRECTION_MARKERS) or command_text.startswith("cd ") or "\ncd " in command_text:
        issues.append(ValidationIssue(
            "unsafe_shell_logging",
            "Final commands should not use cd, tee, or shell log redirection; the runner already sets cwd and captures stdout/stderr.",
            "blocked",
        ))

    if _experiment_commands_are_only_setup_or_inspection(plan.commands):
        issues.append(ValidationIssue(
            "setup_only_experiment",
            "Experiment plan only contains dependency/setup or inspection commands; final experiment commands must directly run training, evaluation, demo, or another goal-directed experiment entry point.",
            "needs_config",
        ))

    if _experiment_commands_include_help(plan.commands):
        issues.append(ValidationIssue(
            "help_in_experiment",
            "Experiment commands include --help/-h inspection; help commands belong in the probe stage, not the formal experiment stage.",
            "needs_config",
        ))

    missing_entrypoint = _missing_goal_entrypoint(goal, command_text)
    if missing_entrypoint:
        issues.append(ValidationIssue(
            "missing_goal_entrypoint",
            f"Experiment goal names `{missing_entrypoint}`, but final commands do not run that entry point.",
            "needs_config",
        ))

    issues.extend(_cli_argument_issues(plan.commands, probe_text))
    return _dedupe_issues(issues)


def annotate_plan_with_validation_issues(plan: CommandPlan, issues: list[str]) -> CommandPlan:
    """Apply validation issues from a secondary reviewer such as an LLM."""
    structured = [ValidationIssue("semantic_review", issue, _semantic_issue_feasibility(issue)) for issue in issues]
    return apply_validation_issues(plan, structured, label="Semantic validation issues")


def _semantic_issue_feasibility(issue: str) -> FeasibilityRank:
    """Infer feasibility from an LLM semantic review issue."""
    lowered = issue.lower()
    if any(marker in lowered for marker in ("unsafe", "too expensive")):
        return "unsafe_or_too_expensive"
    if any(marker in lowered for marker in ("blocked", "impossible", "missing data", "missing checkpoint")):
        return "blocked"
    if any(marker in lowered for marker in ("patch", "code change", "modify the repo", "repo-local")):
        return "needs_patch"
    return "needs_config"


def apply_validation_issues(plan: CommandPlan, issues: list[ValidationIssue], label: str) -> CommandPlan:
    """Apply structured validation issues to a command plan."""
    if not issues:
        return plan
    messages = [issue.message for issue in issues]
    existing_inputs = [item for item in plan.needs_user_input if item not in messages]
    stop_reason = (plan.stop_reason or "").strip()
    validation_reason = f"{label}: " + " ".join(messages)
    return plan.model_copy(update={
        "feasibility": _merged_feasibility(plan.feasibility, issues),
        "needs_user_input": existing_inputs + messages,
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


def _loss_logging_is_uncertain(evidence_text: str) -> bool:
    """Return whether loss logging evidence is uncertain."""
    for block in _output_call_blocks(evidence_text):
        if "loss" in block.lower():
            return False
    return True


def _output_call_blocks(text: str) -> list[str]:
    """Return logger/print call blocks, without unrelated following source lines."""
    lines = text.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        if not _mentions_any(lowered, LOSS_OUTPUT_MARKERS):
            continue
        block = [line]
        balance = line.count("(") - line.count(")")
        for following in lines[index + 1:index + 8]:
            if balance <= 0:
                break
            block.append(following)
            balance += following.count("(") - following.count(")")
        blocks.append("\n".join(block))
    return blocks


def _plan_admits_missing_required_loss(plan_text: str) -> bool:
    """Return whether the plan admits required loss cannot be reported."""
    missing_markers = (
        "loss cannot be reported",
        "cannot report loss",
        "cannot be reported",
        "loss is unavailable",
        "metric is unavailable",
        "not log training loss",
        "does not log training loss",
        "does not print training loss",
        "training loss cannot",
    )
    return "loss" in plan_text and _mentions_any(plan_text, missing_markers)


def _experiment_commands_are_only_setup_or_inspection(commands: list[str]) -> bool:
    """Return whether experiment commands are only setup or inspection."""
    meaningful = [command.strip() for command in commands if command.strip()]
    if not meaningful:
        return False
    return all(_is_setup_or_inspection_command(command) for command in meaningful)


def _experiment_commands_include_help(commands: list[str]) -> bool:
    """Return whether an experiment plan still contains probe-style help commands."""
    return any(_command_has_help(command) for command in commands)


def _is_setup_or_inspection_command(command: str) -> bool:
    """Return whether a command is setup or inspection only."""
    lowered = command.strip().lower()
    if lowered.startswith(("pip install", "python -m pip install", "python3 -m pip install", "conda install", "mamba install")):
        return True
    if lowered.startswith(("python -c", "python3 -c")):
        inspection_markers = ("import ", "print(", "__version__", "cuda.is_available", "device_count")
        return any(marker in lowered for marker in inspection_markers)
    if _command_has_help(lowered):
        return True
    if lowered.startswith(("ls", "find", "rg", "grep", "sed", "cat", "head", "tail", "wc", "pwd")):
        return True
    if lowered.startswith(("python -m py_compile", "python3 -m py_compile")):
        return True
    return False


def _command_has_help(command: str) -> bool:
    """Return whether a command requests help output."""
    lowered = command.strip().lower()
    return "--help" in lowered or lowered.endswith(" -h") or " -h " in lowered


def _cli_argument_issues(commands: list[str], probe_text: str) -> list[ValidationIssue]:
    """Validate command-line flags against discovered --help output."""
    options = _help_options(probe_text)
    if not options:
        return []
    issues: list[ValidationIssue] = []
    for command in commands:
        if _is_setup_or_inspection_command(command):
            continue
        for flag, value in _command_flags(command):
            if flag not in options:
                close = _closest_option(flag, options)
                hint = f" Did you mean `{close}`?" if close else ""
                issues.append(ValidationIssue(
                    "unknown_cli_flag",
                    f"Experiment command uses `{flag}`, but probe help output does not list that option.{hint}",
                    "needs_config",
                ))
            elif options[flag] and value is None:
                issues.append(ValidationIssue(
                    "missing_cli_flag_value",
                    f"Experiment command uses `{flag}` without the value required by probe help output.",
                    "needs_config",
                ))
    return issues


def _help_options(probe_text: str) -> dict[str, bool]:
    """Extract option names and whether they require values from help output."""
    options: dict[str, bool] = {}
    for raw_line in probe_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("--"):
            continue
        match = re.match(r"(?P<flag>--[A-Za-z0-9_-]+)(?P<rest>.*)", line)
        if not match:
            continue
        flag = match.group("flag")
        rest = match.group("rest").strip()
        options[flag] = bool(rest and not rest.startswith("show "))
    return options


def _command_flags(command: str) -> list[tuple[str, str | None]]:
    """Return long flags and attached or following values from one command."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    flags: list[tuple[str, str | None]] = []
    for index, token in enumerate(tokens):
        if not token.startswith("--") or token == "--":
            continue
        if "=" in token:
            flag, value = token.split("=", 1)
        else:
            flag = token
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            value = None if next_token is None or next_token.startswith("-") else next_token
        flags.append((flag, value))
    return flags


def _closest_option(flag: str, options: dict[str, bool]) -> str | None:
    """Return a simple underscore/dash spelling correction if available."""
    normalized = flag.replace("-", "_")
    for option in options:
        if option.replace("-", "_") == normalized:
            return option
    return None


def _missing_goal_entrypoint(goal: str, command_text: str) -> str | None:
    """Find a goal entry point missing from final commands."""
    for match in re.finditer(EXPERIMENT_ENTRYPOINT_RE, goal):
        entrypoint = match.group(0).strip("`'\".,;:)")
        if entrypoint and entrypoint.lower() not in command_text:
            return entrypoint
    return None


def _merged_feasibility(current: str | None, issues: list[ValidationIssue]) -> str:
    """Choose the most severe feasibility among existing and validation issues."""
    candidates = [current or "ready_to_run", *[issue.feasibility for issue in issues]]
    return max(candidates, key=lambda item: FEASIBILITY_PRIORITY.get(item, 0))


def _dedupe_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    """Deduplicate validation issues by code and message."""
    seen: set[tuple[str, str]] = set()
    deduped: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.code, issue.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _mentions_any(text: str, markers: tuple[str, ...]) -> bool:
    """Return whether text contains any marker."""
    return any(marker in text for marker in markers)
