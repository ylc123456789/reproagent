"""Write human-readable and structured reproduction results."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .models import AgentState, ReproAgentVersion
from .text import normalize_text


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
    """Write result.md, result.json, frozen evidence, and state.json."""
    path = state.task.workspace_dir / "result.md"
    state.result_path = path
    evidence, evidence_warnings = _freeze_evidence(state)
    structured = {
        "schema": "repro_result_v1",
        "task_id": state.task.task_id,
        "status": state.status,
        "summary": state.final_summary,
        "metrics": state.structured_result.get("metrics", {}),
        "parameters": state.structured_result.get("parameters", {}),
        "deviations": state.structured_result.get("deviations", []),
        "evidence": evidence,
        "warnings": evidence_warnings,
    }

    attempt_label = f" (retry/resume)" if state.attempt_count > 1 else ""
    lines = [
        "# Reproduction Result", "",
        f"Task ID: `{state.task.task_id}`",
        f"Attempt: {state.attempt_count}{attempt_label}",
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

    # Recovered warnings: failed commands that were later retried/resolved
    recovered = _recovered_warnings(state)
    if recovered:
        lines += ["## Recovered Warnings", ""] + [f"- {w}" for w in recovered] + [""]

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
    if evidence:
        lines += ["## Frozen Evidence", ""]
        lines += [f"- `{item['path']}` (sha256 `{item['sha256']}`)" for item in evidence]
        lines.append("")
    if evidence_warnings:
        lines += ["## Evidence Warnings", ""]
        lines += [f"- {warning}" for warning in evidence_warnings]
        lines.append("")

    path.write_text(_clean_text("\n".join(lines)), encoding="utf-8")
    result_json = state.task.workspace_dir / "result.json"
    result_json.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    state.produced_files["result"] = path
    state.produced_files["result_json"] = result_json
    state.produced_files["state"] = state.task.workspace_dir / "state.json"
    logs_dir = state.task.workspace_dir / "logs"
    if logs_dir.is_dir():
        state.produced_files["logs"] = logs_dir
    _save_agent_state(state)
    return path


def _freeze_evidence(state: AgentState) -> tuple[list[dict], list[str]]:
    """Copy only explicitly declared result files into the task workspace."""
    declared = state.structured_result.get("evidence_files", [])
    if not isinstance(declared, list):
        return [], ["evidence_files must be a list"]

    workspace = state.task.workspace_dir.resolve()
    roots: list[tuple[str, Path]] = []
    if state.repo_context is not None:
        roots.append(("repo", state.repo_context.repo_path.resolve()))
    roots.append(("workspace", workspace))
    frozen: list[dict] = []
    warnings: list[str] = []
    seen: set[Path] = set()

    for raw_path in declared:
        source = Path(str(raw_path))
        if not source.is_absolute():
            base = state.repo_context.repo_path if state.repo_context else workspace
            source = base / source
        try:
            source = source.resolve(strict=True)
        except OSError:
            warnings.append(f"Evidence file does not exist: {raw_path}")
            continue
        if not source.is_file():
            warnings.append(f"Evidence path is not a file: {raw_path}")
            continue
        if source in seen:
            continue
        seen.add(source)

        location = next(
            ((label, source.relative_to(root)) for label, root in roots
             if source == root or root in source.parents),
            None,
        )
        if location is None:
            warnings.append(f"Evidence file is outside the repository/workspace: {raw_path}")
            continue
        label, relative = location
        destination = workspace / "evidence" / label / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copy2(source, destination)
        frozen.append({
            "source": str(raw_path),
            "path": str(destination.relative_to(workspace)),
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        })
    return frozen, warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recovered_warnings(state: AgentState) -> list[str]:
    """Scan steps for failed commands that were resolved by later recovery."""
    warnings: list[str] = []
    for step in state.steps:
        for cr in step.command_results:
            if cr.exit_code not in (0, -2):  # -2 is safety-blocked, expected
                warnings.append(
                    f"Step {step.step}: `{cr.command[:80]}` (exit={cr.exit_code}) — "
                    f"recovered by subsequent step"
                )
    for step in state.steps:
        if step.error and "parse_error" in step.action:
            warnings.append(f"Step {step.step}: parse_error — LLM response was not valid JSON, retried")
        if step.error and "llm_error" in step.action:
            warnings.append(f"Step {step.step}: llm_error — API call failed, retried")
    return warnings[:20]


def _save_agent_state(state: AgentState):
    """Persist agent state to state.json."""
    state.task.workspace_dir.mkdir(parents=True, exist_ok=True)
    spath = state.task.workspace_dir / "state.json"
    spath.write_text(normalize_text(state.model_dump_json(indent=2)), encoding="utf-8")
