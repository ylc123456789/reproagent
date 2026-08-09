"""Write state.json and result.md."""
from __future__ import annotations

from pathlib import Path

from .models import AgentState, ReproAgentVersion, ReproState
from .text import normalize_text


def save_state(state: ReproState):
    """Persist the current workflow state to state.json."""
    state.task.workspace_dir.mkdir(parents=True, exist_ok=True)
    path = state.task.workspace_dir / "state.json"
    path.write_text(normalize_text(state.model_dump_json(indent=2)), encoding="utf-8")
    return path


def write_result(state: ReproState):
    """Legacy report writer — kept for test and infrastructure compatibility."""
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
        f"- Repo cache dir: `{state.task.repo_cache_dir or 'not configured'}`",
        f"- Experiment goal: {state.task.experiment_goal or 'not specified'}",
        f"- CodingAgent path: `{state.task.codingagent_path or 'importable default'}`",
        "",
    ]
    if state.reproagent_version:
        version = state.reproagent_version
        lines += [
            "## ReproAgent Version", "",
            f"- Source path: `{version.source_path}`",
            f"- Git remote: `{version.git_remote or 'unknown'}`",
            f"- Git branch: `{version.git_branch or 'unknown'}`",
            f"- Git commit: `{version.git_commit or 'unknown'}`",
            f"- Git dirty: `{version.git_dirty}`",
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
    if state.coding_agent_results:
        lines += _coding_agent_lines(state.coding_agent_results)
    lines += ["## Final Summary", "", state.final_summary or "No final summary.", ""]
    path.write_text(_clean_text("\n".join(lines)), encoding="utf-8")
    save_state(state)
    return path


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


def _clean_text(text: str) -> str:
    """Normalize optional report text."""
    return normalize_text(text)


def write_agent_result(state: AgentState, version: ReproAgentVersion | None = None) -> Path:
    """Write result.md + state.json for an agent-loop run."""
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
        f"- Repo cache dir: `{state.task.repo_cache_dir or 'not configured'}`",
        f"- Experiment goal: {state.task.experiment_goal or 'not specified'}",
        f"- CodingAgent path: `{state.task.codingagent_path or 'importable default'}`",
        f"- Max steps: {state.task.max_steps}",
        "",
    ]

    if version:
        lines += [
            "## ReproAgent Version", "",
            f"- Source path: `{version.source_path}`",
            f"- Git remote: `{version.git_remote or 'unknown'}`",
            f"- Git branch: `{version.git_branch or 'unknown'}`",
            f"- Git commit: `{version.git_commit or 'unknown'}`",
            f"- Git dirty: `{version.git_dirty}`",
            "",
        ]

    if state.repo_context:
        lines += [
            "## Context", "",
            f"- Repo path: `{state.repo_context.repo_path}`",
            f"- Commit: `{state.repo_context.commit_hash or 'unknown'}`", "",
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
            "",
        ]

    lines += _coding_agent_lines(state.coding_results)

    lines += [
        "## Agent Steps", "",
    ]
    for step in state.steps:
        lines.append(f"### Step {step.step}: {step.action} ({step.stage_hint})")
        if step.error:
            lines.append(f"Error: {step.error}")
        if step.command_results:
            for r in step.command_results:
                tag = "OK" if r.exit_code == 0 else "FAIL"
                lines.append(f"- `{r.command}` -> {tag} exit={r.exit_code}, {r.duration_seconds}s")
                lines.append(f"  - stdout: `{r.stdout_path}`")
                lines.append(f"  - stderr: `{r.stderr_path}`")
        if step.audit:
            lines.append(f"Audit: {'OK' if step.audit.success else 'FAILED'}")
        lines.append("")

    lines += [
        "## Final Summary", "",
        state.final_summary or "No final summary.", "",
    ]

    path.write_text(_clean_text("\n".join(lines)), encoding="utf-8")
    _save_agent_state(state)
    state.produced_files["result"] = path
    state.produced_files["state"] = state.task.workspace_dir / "state.json"
    logs_dir = state.task.workspace_dir / "logs"
    if logs_dir.is_dir():
        state.produced_files["logs"] = logs_dir
    return path


def _save_agent_state(state: AgentState):
    """Persist agent state to state.json."""
    state.task.workspace_dir.mkdir(parents=True, exist_ok=True)
    spath = state.task.workspace_dir / "state.json"
    spath.write_text(normalize_text(state.model_dump_json(indent=2)), encoding="utf-8")
