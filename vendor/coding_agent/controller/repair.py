"""Repair failed patches and structured edits."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from ..apply import PatchApplyError, extract_patch_paths, normalize_patch_text
from ..edits import StructuredEditError, find_all
from ..llm import LLMClient
from ..models import CodeTaskSpec, ControllerAction
from ..safety import SafetyError, ensure_path_allowed

class PatchRepairResponse(BaseModel):
    """Structured response returned by patch repair prompts."""
    action: str = "apply_patch"
    patch: str | None = None
    path: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    anchor_text: str | None = None
    insert_text: str | None = None
    occurrence_index: int | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("notes", mode="before")
    @classmethod
    def coerce_notes(cls, value):
        """Normalize repair notes to a list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


def repair_structured_edit(
    spec: CodeTaskSpec,
    failed_action: ControllerAction,
    error: str,
    output_dir: Path,
    step: int,
    attempt: int,
    client: LLMClient,
) -> ControllerAction:
    """Ask the model to repair a failed structured edit."""
    match_context = _structured_edit_match_context(spec, failed_action)
    _save_structured_edit_context(output_dir, step, attempt, match_context)
    system = (
        "You repair failed deterministic structured edits for a coding agent. Return one JSON action only. "
        "Prefer a longer exact old_text or anchor_text copied from match_context so the edit matches exactly once. "
        "If a repeated match is truly intentional and the surrounding context proves which one is correct, you may set "
        "occurrence_index to the correct 1-based occurrence. Do not switch to unified diff."
    )
    user = {
        "task_goal": spec.task_goal,
        "constraints": spec.constraints,
        "failed_action": failed_action.model_dump(),
        "error": error,
        "match_context": match_context,
        "schema": {
            "action": "replace_text|insert_before|insert_after",
            "reasoning": "brief reason",
            "path": "relative path",
            "old_text": "exact old text for replace_text",
            "new_text": "replacement text for replace_text",
            "anchor_text": "exact unique anchor text for insert_before/insert_after",
            "insert_text": "text to insert for insert_before/insert_after",
            "occurrence_index": "optional 1-based occurrence index",
        },
    }
    repaired = ControllerAction.model_validate(client.complete_json(system, json.dumps(user, indent=2)))
    if repaired.action not in {"replace_text", "insert_before", "insert_after"}:
        raise StructuredEditError(f"structured edit repair returned unsupported action: {repaired.action}")
    _save_structured_edit_response(output_dir, step, attempt, repaired)
    return repaired


def _structured_edit_match_context(spec: CodeTaskSpec, action: ControllerAction) -> dict[str, object]:
    """Build match context for structured edit repair."""
    if not action.path:
        return {"error": "structured edit has no path"}
    try:
        path = ensure_path_allowed(spec.repo_path, action.path, spec.allowed_paths or None)
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {"path": action.path, "error": str(exc)}

    needle = action.old_text if action.action == "replace_text" else action.anchor_text
    if not needle:
        return {"path": action.path, "error": "structured edit has no old_text/anchor_text"}

    matches = []
    for occurrence, index in enumerate(find_all(text, needle), start=1):
        matches.append(_match_window(text, index, len(needle), occurrence))
    return {
        "path": action.path,
        "action": action.action,
        "match_text": needle,
        "match_count": len(matches),
        "matches": matches[:20],
    }


def _match_window(text: str, index: int, length: int, occurrence: int) -> dict[str, object]:
    """Return line and context window for a text match."""
    line_start = text.count("\n", 0, index) + 1
    line_end = line_start + text[index : index + length].count("\n")
    before_start = _nth_previous_line_start(text, index, 4)
    after_end = _nth_next_line_end(text, index + length, 4)
    return {
        "occurrence_index": occurrence,
        "start_line": line_start,
        "end_line": line_end,
        "context": text[before_start:after_end],
    }


def _nth_previous_line_start(text: str, index: int, line_count: int) -> int:
    """Find the start index several lines before a point."""
    cursor = index
    for _ in range(line_count):
        previous = text.rfind("\n", 0, max(cursor - 1, 0))
        if previous == -1:
            return 0
        cursor = previous
    return cursor + 1


def _nth_next_line_end(text: str, index: int, line_count: int) -> int:
    """Find the end index several lines after a point."""
    cursor = index
    for _ in range(line_count):
        next_newline = text.find("\n", cursor)
        if next_newline == -1:
            return len(text)
        cursor = next_newline + 1
    return cursor


def _save_structured_edit_failure(
    spec: CodeTaskSpec,
    output_dir: Path,
    step: int,
    attempt: int,
    action: ControllerAction,
    error: str,
) -> None:
    """Persist a failed structured edit action."""
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {"action": action.model_dump(), "error": error, "allowed_paths": spec.allowed_paths}
    (logs / f"failed_structured_edit_{step:02d}_{attempt:02d}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _save_structured_edit_context(output_dir: Path, step: int, attempt: int, context: dict[str, object]) -> None:
    """Persist context used for structured edit repair."""
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"structured_edit_context_{step:02d}_{attempt:02d}.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )


def _save_structured_edit_response(output_dir: Path, step: int, attempt: int, action: ControllerAction) -> None:
    """Persist the repaired structured edit action."""
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"structured_edit_response_{step:02d}_{attempt:02d}.json").write_text(
        action.model_dump_json(indent=2), encoding="utf-8"
    )

def repair_patch(
    spec: CodeTaskSpec,
    failed_patch: str,
    stderr: str,
    output_dir: Path,
    step: int,
    attempt: int,
    client: LLMClient,
) -> PatchRepairResponse:
    """Ask the model to repair a failed patch."""
    paths = extract_patch_paths(failed_patch)
    file_context = []
    for rel in paths[:6]:
        try:
            path = ensure_path_allowed(spec.repo_path, rel, spec.allowed_paths or None)
            file_context.append({"path": rel, "text": path.read_text(encoding="utf-8", errors="ignore")[:24_000]})
        except Exception as exc:
            file_context.append({"path": rel, "error": str(exc)})
    _save_repair_context(output_dir, step, attempt, file_context)

    system = (
        "You repair failed edits for a coding agent. Prefer structured edit actions over another unified diff. "
        "For a small local edit, return action replace_text, insert_before, or insert_after with exact text copied from "
        "current_file_context. Use apply_patch only when a structured edit is unsuitable. Return only JSON."
    )
    user = {
        "task_goal": spec.task_goal,
        "constraints": spec.constraints,
        "allowed_paths": spec.allowed_paths,
        "git_apply_check_or_apply_error": stderr,
        "failed_patch": failed_patch,
        "current_file_context": file_context,
        "schema": {
            "action": "replace_text|insert_before|insert_after|apply_patch",
            "path": "relative path for structured edit",
            "old_text": "exact old text for replace_text",
            "new_text": "replacement text for replace_text",
            "anchor_text": "exact anchor text for insert_before/insert_after",
            "insert_text": "text to insert for insert_before/insert_after",
            "occurrence_index": "optional 1-based index when a repeated anchor is intentional",
            "patch": "valid unified diff only if action is apply_patch",
            "notes": ["repair note"],
        },
    }
    response = PatchRepairResponse.model_validate(client.complete_json(system, json.dumps(user, indent=2)))
    _save_repair_response(output_dir, step, attempt, response)
    return response


def _save_failed_patch(output_dir: Path, step: int, attempt: int, patch: str, stderr: str) -> None:
    """Persist a failed patch and error text."""
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = f"failed_patch_{step:02d}_{attempt:02d}"
    (logs / f"{stem}.patch").write_text(patch, encoding="utf-8")
    (logs / f"{stem}.stderr").write_text(stderr, encoding="utf-8")


def _save_repair_context(output_dir: Path, step: int, attempt: int, file_context: list[dict[str, str]]) -> None:
    """Persist file context used for patch repair."""
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"repair_context_{step:02d}_{attempt:02d}.json"
    path.write_text(json.dumps(file_context, indent=2), encoding="utf-8")


def _save_repair_response(output_dir: Path, step: int, attempt: int, response: PatchRepairResponse) -> None:
    """Persist the model patch repair response."""
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"repair_response_{step:02d}_{attempt:02d}.json"
    path.write_text(response.model_dump_json(indent=2), encoding="utf-8")


