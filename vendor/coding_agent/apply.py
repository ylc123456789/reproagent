from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .safety import ensure_path_allowed

DIFF_PATH_RE = re.compile(r"^(?:---|\+\+\+) (?:a/|b/)?(.+)$")


class PatchApplyError(RuntimeError):
    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def normalize_patch_text(patch_text: str) -> str:
    text = patch_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text + "\n" if text else ""


def extract_patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in normalize_patch_text(patch_text).splitlines():
        match = DIFF_PATH_RE.match(line)
        if not match:
            continue
        path = match.group(1).strip()
        if path == "/dev/null":
            continue
        paths.append(path)
    return sorted(set(paths))


def validate_patch(repo_root: Path, patch_text: str, allowed_paths: list[str] | None = None) -> list[str]:
    normalized = normalize_patch_text(patch_text)
    if not normalized.strip():
        raise PatchApplyError("empty patch")
    paths = extract_patch_paths(normalized)
    if not paths:
        raise PatchApplyError("patch does not contain file paths")
    for path in paths:
        ensure_path_allowed(repo_root, path, allowed_paths)
    return paths


def check_patch_text(repo_root: Path, patch_text: str, allowed_paths: list[str] | None = None) -> list[str]:
    normalized = normalize_patch_text(patch_text)
    changed_paths = validate_patch(repo_root, normalized, allowed_paths)
    result = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=repo_root,
        input=normalized,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise PatchApplyError(f"git apply --check failed:\n{result.stderr}", result.stderr)
    return changed_paths


def apply_patch_text(repo_root: Path, patch_text: str, allowed_paths: list[str] | None = None) -> list[str]:
    normalized = normalize_patch_text(patch_text)
    changed_paths = check_patch_text(repo_root, normalized, allowed_paths)
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=repo_root,
        input=normalized,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise PatchApplyError(f"git apply failed:\n{result.stderr}", result.stderr)
    return changed_paths


def current_diff(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout
