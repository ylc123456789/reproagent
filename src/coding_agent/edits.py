from __future__ import annotations

from pathlib import Path

from .safety import ensure_path_allowed


class StructuredEditError(RuntimeError):
    pass


def replace_text_once(
    repo_root: Path,
    relative_path: str,
    old_text: str,
    new_text: str,
    allowed_paths: list[str] | None = None,
) -> str:
    path = ensure_path_allowed(repo_root, relative_path, allowed_paths)
    text = path.read_text(encoding="utf-8", errors="ignore")
    count = text.count(old_text)
    if count != 1:
        raise StructuredEditError(f"replace_text expected exactly one match in {relative_path}, found {count}")
    path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return relative_path


def insert_before_anchor(
    repo_root: Path,
    relative_path: str,
    anchor_text: str,
    insert_text: str,
    allowed_paths: list[str] | None = None,
) -> str:
    return _insert_at_anchor(repo_root, relative_path, anchor_text, insert_text, before=True, allowed_paths=allowed_paths)


def insert_after_anchor(
    repo_root: Path,
    relative_path: str,
    anchor_text: str,
    insert_text: str,
    allowed_paths: list[str] | None = None,
) -> str:
    return _insert_at_anchor(repo_root, relative_path, anchor_text, insert_text, before=False, allowed_paths=allowed_paths)


def _insert_at_anchor(
    repo_root: Path,
    relative_path: str,
    anchor_text: str,
    insert_text: str,
    before: bool,
    allowed_paths: list[str] | None,
) -> str:
    path = ensure_path_allowed(repo_root, relative_path, allowed_paths)
    text = path.read_text(encoding="utf-8", errors="ignore")
    count = text.count(anchor_text)
    if count != 1:
        position = "before" if before else "after"
        raise StructuredEditError(
            f"insert_{position} expected exactly one anchor match in {relative_path}, found {count}"
        )
    index = text.index(anchor_text)
    if before:
        updated = text[:index] + insert_text + text[index:]
    else:
        insert_at = index + len(anchor_text)
        if not anchor_text.endswith("\n") and insert_at < len(text) and text[insert_at] == "\n":
            insert_at += 1
        updated = text[:insert_at] + insert_text + text[insert_at:]
    path.write_text(updated, encoding="utf-8")
    return relative_path
