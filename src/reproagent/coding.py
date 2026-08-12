"""Repo-local patch orchestration through the CodingAgent integration."""
from __future__ import annotations

import shlex

from .runtime.environment import find_conda
from .integrations.codingagent import run_code_task as run_codingagent_code_task
from .models import CodingAgentResult, CommandPlan, ReproState


def run_coding_agent_for_patch(state: ReproState, plan: CommandPlan) -> CodingAgentResult:
    """Run the external CodingAgent to produce and verify a repository patch."""
    if state.repo_context is None:
        raise RuntimeError("repo context is required before running CodingAgent")
    output_dir = state.task.workspace_dir / "patches" / f"coding_agent_{len(state.coding_agent_results) + 1:02d}"
    env_summary = _environment_summary(state)
    verify_commands = _verification_commands(state, plan)
    report = run_codingagent_code_task(
        codingagent_path=state.task.codingagent_path,
        repo_path=state.repo_context.repo_path,
        task_goal=_task_goal(state, plan, env_summary),
        constraints=_constraints(env_summary),
        verify_commands=verify_commands,
        max_steps=state.task.max_coding_agent_steps,
        timeout_seconds=state.task.timeout_seconds,
        api_base=state.task.api_base,
        api_key_env=state.task.api_key_env,
        model=state.task.model or "gpt-4.1",
        output_dir=output_dir,
    )
    return CodingAgentResult(
        status=report.status,
        summary=report.summary,
        changed_files=report.changed_files,
        diff_path=report.diff_path,
        report_path=output_dir / "patch_report.md",
        output_dir=output_dir,
        environment_summary=env_summary,
        verification_commands=[result.command for result in report.verification_results] or verify_commands,
        residual_risks=report.residual_risks,
    )


def _task_goal(state: ReproState, plan: CommandPlan, env_summary: str) -> str:
    """Build the CodingAgent patch goal text."""
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

Execution environment:
{env_summary}

Your verification commands are already wrapped by reproagent to run inside that environment. If an import, dependency, CUDA, or package-version error appears, report it as an environment issue for reproagent instead of installing packages.

Prefer the smallest code/config change that resolves the validation issues. If a requested metric is not logged, add logging rather than changing training behavior. If the goal is bounded and the repo lacks suitable controls, add a minimal CLI/config control instead of editing default full-experiment behavior.
"""


def _constraints(env_summary: str) -> list[str]:
    """Build patch constraints for CodingAgent."""
    return [
        "Do not change model architecture unless explicitly required by the task.",
        "Do not change optimizer, loss function, dataset split, or evaluation metric unless explicitly required by the task.",
        "Prefer logging/configuration changes over algorithmic changes.",
        "Keep patches minimal and easy to review.",
        "Do not write outside the repository worktree.",
        "Do not remove files, datasets, checkpoints, or caches.",
        "Do not install or upgrade dependencies; reproagent has already prepared the conda environment.",
        "Use the provided verification commands and existing environment instead of running pip/conda/apt installs.",
        "Environment context: " + env_summary.replace(chr(10), " / "),
    ]


def _environment_summary(state: ReproState) -> str:
    """Summarize the prepared runtime environment."""
    if state.environment is None:
        return "No reproagent-managed environment is available; use only repo-local inspection and report environment blockers."
    audit_details = ""
    if state.environment_audit and state.environment_audit.details:
        audit_details = chr(10) + "- Audit details: " + " / ".join(state.environment_audit.details)
    codingagent_path = str(state.task.codingagent_path) if state.task.codingagent_path else "importable default"
    return (
        f"- reproagent has already prepared the conda environment {state.environment.env_name!r} for this repository.\n"
        f"- Environment backend: {state.environment.backend}. Created this run: {state.environment.created}.\n"
        f"- CodingAgent source: {codingagent_path}.\n"
        f"- Verification commands are executed via conda run -n {state.environment.env_name} bash -c <command>.\n"
        "- Do not install, upgrade, or remove dependencies from CodingAgent. Dependency/environment repair belongs to reproagent environment stage.\n"
        "- Your responsibility is repo-local code/config edits and verification inside the prepared environment."
        f"{audit_details}"
    )


def _verification_commands(state: ReproState, plan: CommandPlan) -> list[str]:
    """Choose verification commands for the patch attempt."""
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
    commands = commands[:3]
    if state.environment is None:
        return commands
    return [_wrap_verify_command(command, state.environment.env_name) for command in commands]


def _wrap_verify_command(command: str, env_name: str) -> str:
    """Wrap a verification command in the prepared conda environment."""
    conda = str(find_conda())
    return " ".join([
        shlex.quote(conda),
        "run",
        "-n",
        shlex.quote(env_name),
        "bash",
        "-c",
        shlex.quote(command),
    ])
