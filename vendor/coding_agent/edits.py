"""Apply deterministic structured text edits safely."""
from __future__ import annotations

from pathlib import Path

from .safety import ensure_path_allowed


class StructuredEditError(RuntimeError):
    """Raised when a deterministic structured edit cannot be applied safely."""
    pass


def replace_text_once(
    repo_root: Path,
    relative_path: str,
    old_text: str,
    new_text: str,
    allowed_paths: list[str] | None = None,
    occurrence_index: int | None = None,
) -> str:
    """Replace exact text after resolving a safe single match."""
    path = ensure_path_allowed(repo_root, relative_path, allowed_paths)
    text = path.read_text(encoding="utf-8", errors="ignore")
    index = _resolve_match_index(text, old_text, relative_path, "replace_text", occurrence_index)
    updated = text[:index] + new_text + text[index + len(old_text) :]
    path.write_text(updated, encoding="utf-8")
    return relative_path


def insert_before_anchor(
    repo_root: Path,
    relative_path: str,
    anchor_text: str,
    insert_text: str,
    allowed_paths: list[str] | None = None,
    occurrence_index: int | None = None,
) -> str:
    """Insert text before a resolved anchor."""
    return _insert_at_anchor(
        repo_root,
        relative_path,
        anchor_text,
        insert_text,
        before=True,
        allowed_paths=allowed_paths,
        occurrence_index=occurrence_index,
    )


def insert_after_anchor(
    repo_root: Path,
    relative_path: str,
    anchor_text: str,
    insert_text: str,
    allowed_paths: list[str] | None = None,
    occurrence_index: int | None = None,
) -> str:
    """Insert text after a resolved anchor."""
    return _insert_at_anchor(
        repo_root,
        relative_path,
        anchor_text,
        insert_text,
        before=False,
        allowed_paths=allowed_paths,
        occurrence_index=occurrence_index,
    )


def _insert_at_anchor(
    repo_root: Path,
    relative_path: str,
    anchor_text: str,
    insert_text: str,
    before: bool,
    allowed_paths: list[str] | None,
    occurrence_index: int | None,
) -> str:
    """Insert text around a resolved anchor location."""
    path = ensure_path_allowed(repo_root, relative_path, allowed_paths)
    text = path.read_text(encoding="utf-8", errors="ignore")
    action = "insert_before" if before else "insert_after"
    index = _resolve_match_index(text, anchor_text, relative_path, action, occurrence_index)
    if before:
        if _inserting_at_line_boundary(text, index) and insert_text and not insert_text.endswith("\n"):
            insert_text += "\n"
        updated = text[:index] + insert_text + text[index:]
    else:
        insert_at = index + len(anchor_text)
        line_anchor = False
        if not anchor_text.endswith("\n") and insert_at < len(text) and text[insert_at] == "\n":
            insert_at += 1
            line_anchor = True
        if line_anchor and insert_text and not insert_text.endswith("\n"):
            insert_text += "\n"
        updated = text[:insert_at] + insert_text + text[insert_at:]
    path.write_text(updated, encoding="utf-8")
    return relative_path


def _resolve_match_index(
    text: str,
    needle: str,
    relative_path: str,
    action: str,
    occurrence_index: int | None,
) -> int:
    """Resolve exact text to one match index."""
    if needle == "":
        raise StructuredEditError(f"{action} requires non-empty match text in {relative_path}")
    matches = find_all(text, needle)
    if occurrence_index is not None:
        if occurrence_index < 1 or occurrence_index > len(matches):
            raise StructuredEditError(
                f"{action} occurrence_index {occurrence_index} is outside 1..{len(matches)} in {relative_path}"
            )
        return matches[occurrence_index - 1]
    if len(matches) != 1:
        raise StructuredEditError(f"{action} expected exactly one match in {relative_path}, found {len(matches)}")
    return matches[0]


def find_all(text: str, needle: str) -> list[int]:
    """Return all non-overlapping match offsets."""
    matches: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index == -1:
            return matches
        matches.append(index)
        start = index + max(len(needle), 1)

def _inserting_at_line_boundary(text: str, index: int) -> bool:
    """Return whether an index begins a line."""
    return index == 0 or text[index - 1] == "\n"
