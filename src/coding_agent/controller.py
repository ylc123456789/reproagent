from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .apply import PatchApplyError, apply_patch_text, current_diff, extract_patch_paths, normalize_patch_text
from .context import build_repo_context
from .edits import StructuredEditError, find_all, insert_after_anchor, insert_before_anchor, replace_text_once
from .llm import LLMClient
from .models import AgentState, CodeTaskSpec, ControllerAction, PatchReport, StepRecord
from .report import prepare_output_dir, write_diff, write_initial_diff, write_patch_report, write_state
from .reviewer import review_outcome
from .runner import run_verify_commands
from .safety import SafetyError, ensure_path_allowed


ACTION_SCHEMA = {
    "action": "list_tree|read_file|search|replace_text|insert_before|insert_after|apply_patch|run_command|finish|ask_user",
    "reasoning": "brief reason for this next action",
    "path": "relative file path for read_file or structured edits, optional",
    "query": "search query for search, optional",
    "command": "verification command for run_command, optional",
    "patch": "unified diff for apply_patch, optional",
    "old_text": "exact text copied from the current file for replace_text",
    "new_text": "replacement text for replace_text",
    "anchor_text": "exact unique anchor copied from the current file for insert_before/insert_after; prefer several adjacent lines over a short common line",
    "insert_text": "text to insert before or after anchor_text",
    "occurrence_index": "optional 1-based match index only when a repeated anchor is intentional and read_file/search context proves the target occurrence",
    "status": "completed|failed|blocked|needs_user_input for finish/ask_user",
    "summary": "final or user-facing summary, optional",
    "residual_risks": ["risk strings for finish/ask_user"],
}


class PatchRepairResponse(BaseModel):
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
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


def run_step_controller(spec: CodeTaskSpec) -> PatchReport:
    output_dir = prepare_output_dir(spec)
    log_dir = output_dir / "logs"
    state = AgentState(task=spec)
    context = build_repo_context(spec)
    write_initial_diff(context.initial_diff, output_dir)
    client = LLMClient(api_base=spec.api_base, api_key_env=spec.api_key_env, model=spec.model)

    changed_files: list[str] = []
    verification_results = []
    final_error = ""

    for step in range(1, spec.max_steps + 1):
        try:
            action = choose_next_action(spec, state, context, client)
            (log_dir / f"action_{step:02d}.json").write_text(action.model_dump_json(indent=2), encoding="utf-8")
            record = execute_action(spec, action, output_dir, step, client)
            state.steps.append(record)
            changed_files.extend(path for path in record.changed_files if path not in changed_files)
            verification_results.extend(record.verification_results)
            write_state(state, output_dir)

            if action.action in {"replace_text", "insert_before", "insert_after", "apply_patch", "run_command"}:
                context = build_repo_context(spec)

            if action.action == "finish":
                auto_verification = _run_missing_finish_verification(spec, state.steps, output_dir, step)
                if auto_verification:
                    record.verification_results.extend(auto_verification)
                    record.observation = _append_observation(
                        record.observation,
                        f"Auto-ran {len(auto_verification)} verification command(s) before finish.",
                    )
                    verification_results.extend(auto_verification)
                diff_path = write_diff(current_diff(spec.repo_path), output_dir)
                report = PatchReport(
                    status=_final_status(action.status, changed_files, verification_results, bool(spec.verify_commands)),
                    changed_files=changed_files,
                    diff_path=diff_path,
                    verification_results=verification_results,
                    summary=action.summary or "Controller finished the coding task.",
                    residual_risks=action.residual_risks,
                )
                state.report = report
                write_patch_report(spec, report, output_dir)
                write_state(state, output_dir)
                return report

            if action.action == "ask_user":
                report = PatchReport(
                    status="needs_user_input",
                    changed_files=changed_files,
                    diff_path=write_diff(current_diff(spec.repo_path), output_dir),
                    verification_results=verification_results,
                    summary=action.summary or "Controller needs user input before continuing.",
                    residual_risks=action.residual_risks,
                )
                state.report = report
                write_patch_report(spec, report, output_dir)
                write_state(state, output_dir)
                return report
        except Exception as exc:
            final_error = str(exc)
            error_action = ControllerAction(action="finish", reasoning="An unrecoverable controller error occurred.")
            state.steps.append(StepRecord(step=step, action=error_action, observation="", error=final_error))
            write_state(state, output_dir)
            break

    diff_path = write_diff(current_diff(spec.repo_path), output_dir)
    if changed_files and verification_results:
        report = review_outcome(spec, changed_files, diff_path, verification_results, [final_error] if final_error else [])
    else:
        report = PatchReport(
            status="failed",
            changed_files=changed_files,
            diff_path=diff_path,
            verification_results=verification_results,
            summary=f"Controller stopped before completion: {final_error or 'max_steps reached'}",
            residual_risks=[final_error] if final_error else ["max_steps reached"],
        )
    state.report = report
    write_patch_report(spec, report, output_dir)
    write_state(state, output_dir)
    return report


def choose_next_action(spec: CodeTaskSpec, state: AgentState, context, client: LLMClient) -> ControllerAction:
    system = (
        "You are a coding agent controller inspired by modern agentic coding tools. "
        "Choose exactly one next action from the allowed action set. "
        "After reading a file, prefer structured edit actions (replace_text, insert_before, insert_after) for small local edits. "
        "Use exact old_text or anchor_text copied from the current file. For inserts, prefer a unique multi-line "
        "anchor that includes nearby context instead of a short common line. Use apply_patch only for changes that are not suitable "
        "for exact structured edits. Use finish only after the diff and verification evidence satisfy the task, or when failure is clear. "
        "Never silently change research semantics such as model architecture, loss, optimizer, dataset split, or metric. "
        "Return only JSON matching the schema."
    )
    user = {
        "task_goal": spec.task_goal,
        "constraints": spec.constraints,
        "verify_commands": spec.verify_commands,
        "allowed_paths": spec.allowed_paths,
        "repo_tree": context.tree[:300],
        "snippets": [snippet.model_dump() for snippet in context.snippets[:8]],
        "current_diff_tail": current_diff(spec.repo_path)[-8000:],
        "steps": [_compact_step(step) for step in state.steps[-10:]],
        "available_actions": ACTION_SCHEMA,
    }
    return ControllerAction.model_validate(client.complete_json(system, json.dumps(user, indent=2)))


def execute_action(spec: CodeTaskSpec, action: ControllerAction, output_dir: Path, step: int, client: LLMClient) -> StepRecord:
    if action.action == "list_tree":
        context = build_repo_context(spec)
        return StepRecord(step=step, action=action, observation="\n".join(context.tree[:300]))
    if action.action == "read_file":
        if not action.path:
            raise ValueError("read_file requires path")
        path = ensure_path_allowed(spec.repo_path, action.path, spec.allowed_paths or None)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return StepRecord(step=step, action=action, observation=text[:20_000])
    if action.action == "search":
        if not action.query:
            raise ValueError("search requires query")
        return StepRecord(step=step, action=action, observation=_search_repo(spec.repo_path, action.query))
    if action.action in {"replace_text", "insert_before", "insert_after"}:
        changed, observation = _execute_structured_edit_with_repair(spec, action, output_dir, step, client)
        return StepRecord(step=step, action=action, observation=observation, changed_files=changed)
    if action.action == "apply_patch":
        if not action.patch:
            raise ValueError("apply_patch requires patch")
        changed, observation = _apply_patch_with_repair(spec, action.patch, output_dir, step, client)
        return StepRecord(step=step, action=action, observation=observation, changed_files=changed)
    if action.action == "run_command":
        command = action.command or (spec.verify_commands[0] if spec.verify_commands else None)
        if not command:
            raise ValueError("run_command requires command")
        results = run_verify_commands(spec.repo_path, [command], output_dir / "logs" / f"step_{step:02d}", spec.timeout_seconds)
        result = results[0]
        stdout_tail = result.stdout_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        stderr_tail = result.stderr_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        return StepRecord(
            step=step,
            action=action,
            observation=(
                f"returncode={result.returncode} timed_out={result.timed_out}\n"
                f"stdout_tail:\n{stdout_tail}\n"
                f"stderr_tail:\n{stderr_tail}"
            ),
            verification_results=results,
        )
    if action.action in {"finish", "ask_user"}:
        return StepRecord(step=step, action=action, observation=action.summary or action.reasoning)
    raise ValueError(f"unsupported action: {action.action}")



def _run_missing_finish_verification(
    spec: CodeTaskSpec,
    steps: list[StepRecord],
    output_dir: Path,
    finish_step: int,
):
    if not spec.verify_commands:
        return []
    last_change_step = max((step.step for step in steps if step.changed_files), default=0)
    if last_change_step == 0:
        return []
    last_verify_step = max((step.step for step in steps if step.verification_results), default=0)
    if last_verify_step >= last_change_step:
        return []
    return run_verify_commands(
        spec.repo_path,
        spec.verify_commands,
        output_dir / "logs" / f"step_{finish_step:02d}_finish_verify",
        spec.timeout_seconds,
    )


def _append_observation(existing: str, addition: str) -> str:
    return f"{existing}\n{addition}" if existing else addition


def _execute_structured_edit_with_repair(
    spec: CodeTaskSpec,
    action: ControllerAction,
    output_dir: Path,
    step: int,
    client: LLMClient,
) -> tuple[list[str], str]:
    errors: list[str] = []
    current = action
    max_attempts = spec.patch_repair_attempts + 1
    for attempt in range(1, max_attempts + 1):
        try:
            changed, observation = _execute_structured_edit(spec, current, output_dir)
            suffix = "" if attempt == 1 else f" after {attempt - 1} structured repair attempt(s)"
            return changed, observation + suffix
        except StructuredEditError as exc:
            error = str(exc)
            errors.append(error)
            _save_structured_edit_failure(spec, output_dir, step, attempt, current, error)
            if attempt >= max_attempts:
                raise StructuredEditError(
                    f"structured edit failed after {spec.patch_repair_attempts} repair attempt(s):\n" + "\n---\n".join(errors)
                ) from exc
            current = repair_structured_edit(spec, current, error, output_dir, step, attempt, client)
    raise StructuredEditError("structured edit repair loop exited unexpectedly")


def repair_structured_edit(
    spec: CodeTaskSpec,
    failed_action: ControllerAction,
    error: str,
    output_dir: Path,
    step: int,
    attempt: int,
    client: LLMClient,
) -> ControllerAction:
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
    cursor = index
    for _ in range(line_count):
        previous = text.rfind("\n", 0, max(cursor - 1, 0))
        if previous == -1:
            return 0
        cursor = previous
    return cursor + 1


def _nth_next_line_end(text: str, index: int, line_count: int) -> int:
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
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {"action": action.model_dump(), "error": error, "allowed_paths": spec.allowed_paths}
    (logs / f"failed_structured_edit_{step:02d}_{attempt:02d}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _save_structured_edit_context(output_dir: Path, step: int, attempt: int, context: dict[str, object]) -> None:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"structured_edit_context_{step:02d}_{attempt:02d}.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )


def _save_structured_edit_response(output_dir: Path, step: int, attempt: int, action: ControllerAction) -> None:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"structured_edit_response_{step:02d}_{attempt:02d}.json").write_text(
        action.model_dump_json(indent=2), encoding="utf-8"
    )

def _execute_structured_edit(spec: CodeTaskSpec, action: ControllerAction, output_dir: Path) -> tuple[list[str], str]:
    if not action.path:
        raise ValueError(f"{action.action} requires path")
    if action.action == "replace_text":
        if action.old_text is None or action.new_text is None:
            raise ValueError("replace_text requires old_text and new_text")
        changed = replace_text_once(spec.repo_path, action.path, action.old_text, action.new_text, spec.allowed_paths, action.occurrence_index)
    elif action.action == "insert_before":
        if action.anchor_text is None or action.insert_text is None:
            raise ValueError("insert_before requires anchor_text and insert_text")
        changed = insert_before_anchor(spec.repo_path, action.path, action.anchor_text, action.insert_text, spec.allowed_paths, action.occurrence_index)
    elif action.action == "insert_after":
        if action.anchor_text is None or action.insert_text is None:
            raise ValueError("insert_after requires anchor_text and insert_text")
        changed = insert_after_anchor(spec.repo_path, action.path, action.anchor_text, action.insert_text, spec.allowed_paths, action.occurrence_index)
    else:
        raise ValueError(f"unsupported structured edit: {action.action}")
    diff_path = write_diff(current_diff(spec.repo_path), output_dir)
    return [changed], f"Structured edit {action.action} applied to {changed}. Current diff written to {diff_path}."


def _apply_patch_with_repair(
    spec: CodeTaskSpec,
    patch_text: str,
    output_dir: Path,
    step: int,
    client: LLMClient,
) -> tuple[list[str], str]:
    patch = normalize_patch_text(patch_text)
    errors: list[str] = []
    repair_notes: list[str] = []
    max_attempts = spec.patch_repair_attempts + 1

    for attempt in range(1, max_attempts + 1):
        try:
            changed = apply_patch_text(spec.repo_path, patch, spec.allowed_paths)
            diff_path = write_diff(current_diff(spec.repo_path), output_dir)
            suffix = "" if attempt == 1 else f" after {attempt - 1} repair attempt(s)"
            notes = f" Repair notes: {'; '.join(repair_notes)}" if repair_notes else ""
            return changed, f"Patch applied{suffix}. Current diff written to {diff_path}.{notes}"
        except (PatchApplyError, SafetyError) as exc:
            stderr = getattr(exc, "stderr", "") or str(exc)
            errors.append(stderr)
            _save_failed_patch(output_dir, step, attempt, patch, stderr)
            if attempt >= max_attempts:
                joined = "\n---\n".join(errors)
                raise PatchApplyError(
                    f"patch failed validation/application after {spec.patch_repair_attempts} repair attempt(s):\n{joined}",
                    joined,
                ) from exc
            repaired = repair_patch(spec, patch, stderr, output_dir, step, attempt, client)
            repair_notes.extend(repaired.notes)
            if repaired.action in {"replace_text", "insert_before", "insert_after"}:
                action = ControllerAction(
                    action=repaired.action,
                    reasoning="Repair failed unified diff by using a deterministic structured edit.",
                    path=repaired.path,
                    old_text=repaired.old_text,
                    new_text=repaired.new_text,
                    anchor_text=repaired.anchor_text,
                    insert_text=repaired.insert_text,
                    occurrence_index=repaired.occurrence_index,
                )
                changed, observation = _execute_structured_edit(spec, action, output_dir)
                notes = f" Repair notes: {'; '.join(repair_notes)}" if repair_notes else ""
                return changed, f"Patch converted to structured edit after {attempt} failed diff attempt(s). {observation}{notes}"
            if repaired.action != "apply_patch" or not repaired.patch:
                raise PatchApplyError(f"patch repair returned unsupported action or empty patch: {repaired.action}")
            patch = normalize_patch_text(repaired.patch)

    raise PatchApplyError("patch repair loop exited unexpectedly")


def repair_patch(
    spec: CodeTaskSpec,
    failed_patch: str,
    stderr: str,
    output_dir: Path,
    step: int,
    attempt: int,
    client: LLMClient,
) -> PatchRepairResponse:
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
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = f"failed_patch_{step:02d}_{attempt:02d}"
    (logs / f"{stem}.patch").write_text(patch, encoding="utf-8")
    (logs / f"{stem}.stderr").write_text(stderr, encoding="utf-8")


def _save_repair_context(output_dir: Path, step: int, attempt: int, file_context: list[dict[str, str]]) -> None:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"repair_context_{step:02d}_{attempt:02d}.json"
    path.write_text(json.dumps(file_context, indent=2), encoding="utf-8")


def _save_repair_response(output_dir: Path, step: int, attempt: int, response: PatchRepairResponse) -> None:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"repair_response_{step:02d}_{attempt:02d}.json"
    path.write_text(response.model_dump_json(indent=2), encoding="utf-8")


def _search_repo(repo: Path, query: str, limit: int = 80) -> str:
    matches: list[str] = []
    lowered = query.lower()
    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo).as_posix()
        try:
            ensure_path_allowed(repo, rel)
        except SafetyError:
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            if lowered in line.lower():
                matches.append(f"{rel}:{number}: {line}")
                if len(matches) >= limit:
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches."


def _compact_step(step: StepRecord) -> dict[str, object]:
    return {
        "step": step.step,
        "action": step.action.action,
        "reasoning": step.action.reasoning,
        "observation_tail": step.observation[-2000:],
        "changed_files": step.changed_files,
        "verification": [
            {"command": result.command, "returncode": result.returncode, "timed_out": result.timed_out}
            for result in step.verification_results
        ],
        "error": step.error,
    }


def _final_status(requested_status: str | None, changed_files: list[str], verification_results, verification_required: bool) -> str:
    evidence_status = _status_from_verification(changed_files, verification_results)
    if requested_status == "completed" and verification_required and evidence_status != "completed":
        return evidence_status
    return requested_status or evidence_status


def _status_from_verification(changed_files: list[str], verification_results) -> str:
    if not changed_files:
        return "failed"
    if verification_results and all(result.succeeded for result in verification_results):
        return "completed"
    return "failed"
