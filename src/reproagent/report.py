"""Write state.json and result.md."""
from __future__ import annotations
from .models import ReproState

MOJIBAKE_REPLACEMENTS = {
    "鈥檚": "'s",
    "鈥檛": "'t",
    "鈥檙": "'r",
    "鈥檝": "'v",
    "鈥檓": "'m",
    "鈥檒": "'l",
    "鈥檇": "'d",
    "鈥?": "-",
    "鈥": "-",
    "憇": "s",
    "憆": "r",
    "慳": "a",
    "慛": "N",
}


def save_state(state: ReproState):
    state.task.workspace_dir.mkdir(parents=True, exist_ok=True)
    path = state.task.workspace_dir / "state.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_result(state: ReproState):
    path = state.task.workspace_dir / "result.md"
    state.result_path = path
    lines = [
        "# Reproduction Result", "",
        f"Task ID: `{state.task.task_id}`",
        f"Status: `{state.status}`", "",
        "## Inputs", "",
        f"- Paper: {state.task.paper_url}",
        f"- Repo: {state.task.repo_url}",
        f"- Backend: {state.task.backend}", "",
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
    lines += _stage_lines("Experiment Attempts", state.experiment_attempts)
    lines += ["## Final Summary", "", state.final_summary or "No final summary.", ""]
    path.write_text(_clean_text("\n".join(lines)), encoding="utf-8")
    save_state(state)
    return path


def _stage_lines(title: str, attempts) -> list[str]:
    lines = [f"## {title}", ""]
    if not attempts:
        return lines + ["No attempts recorded.", ""]
    for attempt in attempts:
        lines += [f"### Attempt {attempt.attempt}", "", f"Plan: {attempt.plan.summary}", ""]
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


def _clean_text(text: str) -> str:
    cleaned = text
    for bad, replacement in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, replacement)
    return cleaned
