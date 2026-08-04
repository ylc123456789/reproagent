"""Execute controller actions with repair support."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..apply import PatchApplyError, apply_patch_text, current_diff, extract_patch_paths, normalize_patch_text
from ..context import TEXT_SUFFIXES, build_repo_context
from ..context_policy import resolve_context_policy
from ..edits import StructuredEditError, find_all, insert_after_anchor, insert_before_anchor, replace_text_once
from ..models import CodeTaskSpec, ControllerAction, StepRecord
from ..report import write_diff
from ..runner import run_verify_commands
from ..safety import SafetyError, ensure_path_allowed
from .repair import repair_patch, repair_structured_edit, _save_failed_patch, _save_structured_edit_failure

def _normalize_action(spec: CodeTaskSpec, steps: list[StepRecord], action: ControllerAction) -> ControllerAction:
    """Fill safe missing fields before executing an action."""
    if action.action not in {"replace_text", "insert_before", "insert_after"} or action.path:
        return action
    inferred = _infer_structured_edit_path(spec, steps, action)
    if not inferred:
        return action
    reason = action.reasoning or ""
    suffix = f" Inferred missing path as {inferred}."
    return action.model_copy(update={"path": inferred, "reasoning": reason + suffix})


def _infer_structured_edit_path(spec: CodeTaskSpec, steps: list[StepRecord], action: ControllerAction) -> str | None:
    """Infer a missing structured edit path from unique text matches."""
    needle = action.old_text if action.action == "replace_text" else action.anchor_text
    if not needle:
        return None
    recent_paths = _recent_action_paths(steps)
    recent_matches = _matching_paths_for_text(spec, needle, recent_paths)
    if len(recent_matches) == 1:
        return recent_matches[0]
    all_matches = _matching_paths_for_text(spec, needle, _candidate_text_paths(spec))
    if len(all_matches) == 1:
        return all_matches[0]
    return None


def _recent_action_paths(steps: list[StepRecord]) -> list[str]:
    """Return recent action paths without duplicates."""
    paths = []
    for step in reversed(steps):
        path = step.action.path
        if path and path not in paths:
            paths.append(path)
    return paths


def _matching_paths_for_text(spec: CodeTaskSpec, needle: str, paths: list[str]) -> list[str]:
    """Find candidate files containing exact text."""
    matches = []
    for rel in paths:
        try:
            path = ensure_path_allowed(spec.repo_path, rel, spec.allowed_paths or None)
        except Exception:
            continue
        if not path.is_file() or Path(rel).suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                matches.append(rel)
        except OSError:
            continue
    return matches


def _candidate_text_paths(spec: CodeTaskSpec) -> list[str]:
    """List safe text files that may be searched for inference."""
    paths = []
    for path in sorted(spec.repo_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(spec.repo_path).as_posix()
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            ensure_path_allowed(spec.repo_path, rel, spec.allowed_paths or None)
        except Exception:
            continue
        paths.append(rel)
    return paths


def execute_action(spec: CodeTaskSpec, action: ControllerAction, output_dir: Path, step: int, client: LLMClient) -> StepRecord:
    """Execute one normalized controller action."""
    if action.action == "list_tree":
        policy = resolve_context_policy(spec)
        context = build_repo_context(spec, max_files=policy.snippet_count, max_bytes=policy.snippet_chars, tree_limit=policy.repo_tree_limit)
        return StepRecord(step=step, action=action, observation="\n".join(context.tree[:policy.repo_tree_limit]))
    if action.action == "read_file":
        if not action.path:
            raise ValueError("read_file requires path")
        observation = _read_file_observation(spec, action)
        return StepRecord(step=step, action=action, observation=observation)
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
    if action.action == "write_file":
        if not action.path:
            raise ValueError("write_file requires path")
        if action.content is None:
            raise ValueError("write_file requires content")
        file_path = ensure_path_allowed(spec.repo_path, action.path, spec.allowed_paths or None)
        if file_path.exists():
            return StepRecord(step=step, action=action, observation=f"write_file refused: {action.path} already exists. Use replace_text to edit existing files.", error=f"file already exists: {action.path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(action.content, encoding="utf-8")
        diff_path = write_diff(current_diff(spec.repo_path), output_dir)
        syntax_note = _check_syntax_after_edit(spec.repo_path, action.path)
        return StepRecord(
            step=step, action=action,
            observation=f"Created {action.path} ({len(action.content)} chars). Current diff written to {diff_path}.{syntax_note}",
            changed_files=[action.path],
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
    """Run verification when finish follows unverified edits."""
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


def _check_syntax_after_edit(repo_root, file_path):
    """Run py_compile on a modified Python file; return error snippet or empty string."""
    if not file_path.endswith('.py'):
        return ''
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(repo_root / file_path)],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ''
    if result.returncode != 0:
        return '\n[SYNTAX CHECK FAILED] ' + result.stderr.strip()[-2000:]
    return ''


def _read_file_observation(spec: CodeTaskSpec, action: ControllerAction) -> str:
    """Read a file observation using line or size limits."""
    if not action.path:
        raise ValueError("read_file requires path")
    path = ensure_path_allowed(spec.repo_path, action.path, spec.allowed_paths or None)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if action.start_line is not None or action.end_line is not None:
        return _slice_lines(text, action.start_line, action.end_line)
    policy = resolve_context_policy(spec)
    if len(text) <= policy.read_file_chars:
        return text
    half = policy.read_file_chars // 2
    return text[:half] + "\n... <read_file truncated middle; use start_line/end_line for exact ranges> ...\n" + text[-half:]


def _slice_lines(text: str, start_line: int | None, end_line: int | None) -> str:
    """Return an inclusive 1-based line slice."""
    lines = text.splitlines(keepends=True)
    start = max((start_line or 1) - 1, 0)
    end = min(end_line or len(lines), len(lines))
    if end < start:
        raise ValueError("read_file end_line must be >= start_line")
    return "".join(lines[start:end])


def _match_count_hint(spec, relative_path, needle, action, occurrence_index):
    """Return a short note when occurrence_index selects among multiple matches."""
    if not needle or not occurrence_index:
        return ''
    try:
        path = ensure_path_allowed(spec.repo_path, relative_path, spec.allowed_paths or None)
        text = path.read_text(encoding='utf-8', errors='ignore')
        count = len(find_all(text, needle))
        if count > 1:
            return f' (occurrence {occurrence_index}/{count})'
    except Exception:
        pass
    return ''


def _execute_structured_edit_with_repair(
    spec: CodeTaskSpec,
    action: ControllerAction,
    output_dir: Path,
    step: int,
    client: LLMClient,
) -> tuple[list[str], str]:
    """Apply a structured edit with repair attempts."""
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


def _execute_structured_edit(spec: CodeTaskSpec, action: ControllerAction, output_dir: Path) -> tuple[list[str], str]:
    """Apply one structured edit action."""
    if not action.path:
        raise ValueError(f"{action.action} requires path")
    if action.action == "replace_text":
        if action.old_text is None or action.new_text is None:
            raise ValueError("replace_text requires old_text and new_text")
        match_hint = _match_count_hint(spec, action.path, action.old_text, "replace_text", action.occurrence_index)
        changed = replace_text_once(spec.repo_path, action.path, action.old_text, action.new_text, spec.allowed_paths, action.occurrence_index)
    elif action.action == "insert_before":
        if action.anchor_text is None or action.insert_text is None:
            raise ValueError("insert_before requires anchor_text and insert_text")
        match_hint = _match_count_hint(spec, action.path, action.anchor_text, "insert_before", action.occurrence_index)
        changed = insert_before_anchor(spec.repo_path, action.path, action.anchor_text, action.insert_text, spec.allowed_paths, action.occurrence_index)
    elif action.action == "insert_after":
        if action.anchor_text is None or action.insert_text is None:
            raise ValueError("insert_after requires anchor_text and insert_text")
        match_hint = _match_count_hint(spec, action.path, action.anchor_text, "insert_after", action.occurrence_index)
        changed = insert_after_anchor(spec.repo_path, action.path, action.anchor_text, action.insert_text, spec.allowed_paths, action.occurrence_index)
    else:
        raise ValueError(f"unsupported structured edit: {action.action}")
    diff_path = write_diff(current_diff(spec.repo_path), output_dir)
    syntax_note = _check_syntax_after_edit(spec.repo_path, changed)
    return [changed], f"Structured edit {action.action}{match_hint} applied to {changed}. Current diff written to {diff_path}.{syntax_note}"


def _apply_patch_with_repair(
    spec: CodeTaskSpec,
    patch_text: str,
    output_dir: Path,
    step: int,
    client: LLMClient,
) -> tuple[list[str], str]:
    """Apply a patch, repairing it when possible."""
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
            syntax_notes = "".join(_check_syntax_after_edit(spec.repo_path, p) for p in changed)
            return changed, f"Patch applied{suffix}. Current diff written to {diff_path}.{notes}{syntax_notes}"
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


def _search_repo(repo: Path, query: str, limit: int = 80) -> str:
    """Search safe text files for a query string."""
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
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            if lowered in line.lower():
                matches.append(f"{rel}:{number}: {line}")
                if len(matches) >= limit:
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches."


