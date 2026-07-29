"""Lightweight final experiment-plan validation."""
from __future__ import annotations

from .models import CommandPlan, ReproState

GUESS_MARKERS = ("assume", "likely", "typical", "probably", "we assume", "may print")
LOG_REDIRECTION_MARKERS = (" tee ", "| tee", "2>&1", " >", " >>")
LOSS_EVIDENCE_MARKERS = ("loss", "criterion", "crossentropy", "cross_entropy")
LOSS_OUTPUT_MARKERS = ("logger.info", "print(", "logging.info")


def validate_experiment_plan(state: ReproState, plan: CommandPlan) -> CommandPlan:
    """Return a copy of plan annotated with obvious validation issues.

    This is intentionally conservative: it does not try to prove a plan correct.
    It only catches common high-signal mismatches between the user goal, probe
    evidence, and final command plan.
    """
    issues: list[str] = []
    goal = state.task.experiment_goal.lower()
    command_text = "\n".join(plan.commands).lower()
    plan_text = "\n".join([plan.summary, *plan.assumptions, plan.stop_reason or ""]).lower()
    probe_text = _probe_text(state).lower()

    if _mentions_any(goal, ("bounded", "short", "small", "one epoch", "few epoch")) and not _mentions_any(
        command_text,
        ("--epoch", "--epochs", "--nepochs", "max_epoch", "max_epochs", "max_steps", "num_train_epochs", "iters", "iterations"),
    ):
        issues.append("Goal asks for a bounded run, but the final commands do not set an explicit training budget such as epochs/steps.")

    if "gpu" in goal and not _has_gpu_execution_evidence(command_text, plan_text, probe_text):
        issues.append("Goal asks for GPU execution, but the final plan does not show command-level GPU use or probe evidence that the entry point defaults to CUDA/GPU.")

    if "loss" in goal and _loss_logging_is_uncertain(probe_text):
        issues.append("Goal asks for training loss, but probe evidence does not show that the unmodified script prints/logs loss; mark this as needs_patch or explicitly report the metric as unavailable.")

    if _mentions_any(plan_text, GUESS_MARKERS):
        issues.append("Plan uses guess language such as assume/likely/typical; replace guesses with probe evidence or mark the uncertain requirement as blocked/needs_patch.")

    if _mentions_any(command_text, LOG_REDIRECTION_MARKERS) or command_text.startswith("cd ") or "\ncd " in command_text:
        issues.append("Final commands should not use cd, tee, or shell log redirection; the runner already sets cwd and captures stdout/stderr.")

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



def _has_gpu_execution_evidence(command_text: str, plan_text: str, probe_text: str) -> bool:
    evidence = "\n".join([command_text, plan_text, probe_text])
    if _mentions_any(evidence, ("--gpu", "--device", "cuda", "torch.cuda", "device = torch.device", "cuda:", "gpu gpu")):
        return True
    if "default=0" in evidence and "gpu" in evidence and "torch.cuda.is_available" in evidence:
        return True
    return False

def _probe_text(state: ReproState) -> str:
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


def _loss_logging_is_uncertain(probe_text: str) -> bool:
    for line in probe_text.splitlines():
        lowered = line.lower()
        if "loss" in lowered and _mentions_any(lowered, LOSS_OUTPUT_MARKERS):
            return False
    return True


def _downgraded_feasibility(plan: CommandPlan, issues: list[str]) -> str:
    if any("loss" in issue.lower() or "guess" in issue.lower() for issue in issues):
        return "needs_patch"
    return "blocked"


def _mentions_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)
