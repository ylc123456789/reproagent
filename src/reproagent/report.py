"""Write state.json and result.md."""
from __future__ import annotations
from .models import ReproState

from .text import normalize_text


def save_state(state: ReproState):
    """Persist the current workflow state to state.json."""
    state.task.workspace_dir.mkdir(parents=True, exist_ok=True)
    path = state.task.workspace_dir / "state.json"
    path.write_text(normalize_text(state.model_dump_json(indent=2)), encoding="utf-8")
    return path


def write_result(state: ReproState):
    """Write the human-readable reproduction result report."""
    path = state.task.workspace_dir / "result.md"
    state.result_path = path
    lines = [
        "# Reproduction Result", "",
        f"Task ID: `{state.task.task_id}`",
        f"Status: `{state.status}`", "",
        "## Inputs", "",
        f"- Paper: {state.task.paper_url}",
        f"- Repo: {state.task.repo_url}",
        f"- Backend: {state.task.backend}",
        f"- Experiment goal: {state.task.experiment_goal or 'not specified'}",
        f"- CodingAgent path: `{state.task.codingagent_path or 'importable default'}`",
        "",
    ]
    if state.repo_context:
        lines += [
            "## Context", "",
            f"- Repo path: `{state.repo_context.repo_path}`",
            f"- Commit: `{state.repo_context.commit_hash or 'unknown'}`",
            f"- Context summary: `{state.repo_context.summary_path}`", "",
            "### Hardware", "",
            "```text",
            state.repo_context.hardware_text,
            "```", "",
        ]
    if state.environment:
        lines += [
            "## Environment", "",
            f"- Backend: `{state.environment.backend}`",
            f"- Conda env: `{state.environment.env_name}`",
            f"- Created this run: `{state.environment.created}`",
            f"- Used environment.yml: `{state.environment.used_environment_yml}`",
            f"- Setup command: `{state.environment.setup_command or 'reused existing env'}`",
            f"- Setup stdout: `{state.environment.setup_stdout_path}`",
            f"- Setup stderr: `{state.environment.setup_stderr_path}`", "",
        ]
    if state.environment_audit:
        mark = "ISSUES"
        if state.environment_audit.requires_repair:
            mark = "REPAIR_REQUIRED"
        elif state.environment_audit.success:
            mark = "WARNING" if state.environment_audit.has_warnings else "OK"
        lines += [
            "## Environment Audit", "",
            f"Status: `{mark}`",
            state.environment_audit.summary, "",
        ]
        if state.environment_audit.details:
            lines += ["Details:"] + [f"- {x}" for x in state.environment_audit.details] + [""]
        lines += [
            f"- Audit stdout: `{state.environment_audit.stdout_path}`",
            f"- Audit stderr: `{state.environment_audit.stderr_path}`", "",
        ]
    lines += _stage_lines("Environment Attempts", state.environment_attempts)
    lines += _stage_lines("Probe Attempts", state.probe_attempts)
    if state.planned_experiment:
        lines += _planned_experiment_lines(state.planned_experiment)
    lines += _coding_agent_lines(state.coding_agent_results)
    lines += _stage_lines("Experiment Attempts", state.experiment_attempts)
    lines += ["## Final Summary", "", state.final_summary or "No final summary.", ""]
    path.write_text(_clean_text("\n".join(lines)), encoding="utf-8")
    save_state(state)
    return path


def _stage_lines(title: str, attempts) -> list[str]:
    """Render report lines for stage attempts."""
    lines = [f"## {title}", ""]
    if not attempts:
        return lines + ["No attempts recorded.", ""]
    for attempt in attempts:
        lines += [f"### Attempt {attempt.attempt}", "", f"Plan: {attempt.plan.summary}", ""]
        if attempt.plan.feasibility:
            lines += [f"Feasibility: `{attempt.plan.feasibility}`", ""]
        if attempt.plan.expected_runtime:
            lines += [f"Expected runtime: {attempt.plan.expected_runtime}", ""]
        if attempt.plan.needs_user_input:
            lines += ["Needs user input:"] + [f"- {x}" for x in attempt.plan.needs_user_input] + [""]
        if attempt.plan.assumptions:
            lines += ["Assumptions:"] + [f"- {x}" for x in attempt.plan.assumptions] + [""]
        if not attempt.results:
            lines += ["No commands executed.", ""]
        for result in attempt.results:
            mark = "OK" if result.success else "FAIL"
            backend = " ".join(result.backend_command) if result.backend_command else "blocked before backend execution"
            lines += [
                f"- `{result.command}` -> {mark} exit={result.exit_code}, {result.duration_seconds}s",
                f"  - backend command: `{backend}`",
                f"  - stdout: `{result.stdout_path}`",
                f"  - stderr: `{result.stderr_path}`",
            ]
        lines.append("")
    return lines



def _coding_agent_lines(results) -> list[str]:
    """Render report lines for CodingAgent attempts."""
    lines = ["## Coding Agent", ""]
    if not results:
        return lines + ["No coding agent runs recorded.", ""]
    for index, result in enumerate(results, start=1):
        lines += ["### Run " + str(index), "", "Status: " + result.status, "", result.summary or "No summary.", ""]
        if result.changed_files:
            lines += ["Changed files:"] + ["- " + path for path in result.changed_files] + [""]
        if result.environment_summary:
            lines += ["Environment:", result.environment_summary, ""]
        if result.diff_path:
            lines += ["- Diff: " + str(result.diff_path)]
        if result.report_path:
            lines += ["- Report: " + str(result.report_path)]
        if result.output_dir:
            lines += ["- Output dir: " + str(result.output_dir)]
        if result.verification_commands:
            lines += ["", "Verification commands:"] + ["- " + command for command in result.verification_commands]
        if result.residual_risks:
            lines += ["", "Residual risks:"] + ["- " + risk for risk in result.residual_risks]
        lines.append("")
    return lines


def _planned_experiment_lines(plan) -> list[str]:
    """Render report lines for an unexecuted experiment plan."""
    lines = ["## Planned Experiment", "", f"Plan: {plan.summary}", ""]
    if plan.feasibility:
        lines += [f"Feasibility: `{plan.feasibility}`", ""]
    if plan.expected_runtime:
        lines += [f"Expected runtime: {plan.expected_runtime}", ""]
    if plan.needs_user_input:
        lines += ["Needs user input:"] + [f"- {x}" for x in plan.needs_user_input] + [""]
    if plan.assumptions:
        lines += ["Assumptions:"] + [f"- {x}" for x in plan.assumptions] + [""]
    if plan.commands:
        lines += ["Commands:"] + [f"{i}. `{command}`" for i, command in enumerate(plan.commands, start=1)] + [""]
    else:
        lines += ["No commands planned.", ""]
    if plan.stop_reason:
        lines += [f"Stop reason: {plan.stop_reason}", ""]
    return lines

def _clean_text(text: str) -> str:
    """Normalize optional report text."""
    return normalize_text(text)
