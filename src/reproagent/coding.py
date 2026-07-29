"""Adapter for the vendored generic coding_agent package."""
from __future__ import annotations

from coding_agent import CodeTaskSpec, run_code_task

from .models import CodingAgentResult, CommandPlan, ReproState


def run_coding_agent_for_patch(state: ReproState, plan: CommandPlan) -> CodingAgentResult:
    if state.repo_context is None:
        raise RuntimeError("repo context is required before running CodingAgent")
    output_dir = state.task.workspace_dir / "patches" / f"coding_agent_{len(state.coding_agent_results) + 1:02d}"
    verify_commands = _verification_commands(state, plan)
    report = run_code_task(CodeTaskSpec(
        repo_path=state.repo_context.repo_path,
        task_goal=_task_goal(state, plan),
        constraints=_constraints(),
        verify_commands=verify_commands,
        max_steps=state.task.max_coding_agent_steps,
        timeout_seconds=state.task.timeout_seconds,
        api_base=state.task.api_base,
        api_key_env=state.task.api_key_env,
        model=state.task.model or "gpt-4.1",
        output_dir=output_dir,
    ))
    return CodingAgentResult(
        status=report.status,
        summary=report.summary,
        changed_files=report.changed_files,
        diff_path=report.diff_path,
        report_path=output_dir / "patch_report.md",
        output_dir=output_dir,
        verification_commands=[result.command for result in report.verification_results] or verify_commands,
        residual_risks=report.residual_risks,
    )


def _task_goal(state: ReproState, plan: CommandPlan) -> str:
    issues = "\n".join(f"- {item}" for item in plan.needs_user_input)
    commands = "\n".join(f"- {command}" for command in plan.commands)
    return f"""Modify the repository minimally so the experiment goal can be attempted without changing research semantics.

Experiment goal:
{state.task.experiment_goal}

Current plan summary:
{plan.summary}

Validation issues to resolve:
{issues or '- none recorded'}

Planned experiment commands before patch:
{commands or '- none'}

Prefer the smallest code/config change that resolves the validation issues. If a requested metric is not logged, add logging rather than changing training behavior. If the goal is bounded and the repo lacks suitable controls, add a minimal CLI/config control instead of editing default full-experiment behavior.
"""


def _constraints() -> list[str]:
    return [
        "Do not change model architecture unless explicitly required by the task.",
        "Do not change optimizer, loss function, dataset split, or evaluation metric unless explicitly required by the task.",
        "Prefer logging/configuration changes over algorithmic changes.",
        "Keep patches minimal and easy to review.",
        "Do not write outside the repository worktree.",
        "Do not remove files, datasets, checkpoints, or caches.",
    ]


def _verification_commands(state: ReproState, plan: CommandPlan) -> list[str]:
    commands: list[str] = []
    for attempt in state.probe_attempts:
        for result in attempt.results:
            command = result.command
            if "--help" in command and command not in commands:
                commands.append(command)
    for command in plan.commands:
        lowered = command.lower()
        if ("--debug" in lowered or "--help" in lowered) and command not in commands:
            commands.append(command)
    return commands[:3]
