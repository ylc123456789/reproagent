from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .safety import SafetyError, ensure_path_allowed

DIFF_PATH_RE = re.compile(r"^(?:---|\+\+\+) (?:a/|b/)?(.+)$")


def extract_patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        match = DIFF_PATH_RE.match(line)
        if not match:
            continue
        path = match.group(1).strip()
        if path == "/dev/null":
            continue
        paths.append(path)
    return sorted(set(paths))


def validate_patch(repo_root: Path, patch_text: str, allowed_paths: list[str] | None = None) -> list[str]:
    if not patch_text.strip():
        raise SafetyError("empty patch")
    paths = extract_patch_paths(patch_text)
    if not paths:
        raise SafetyError("patch does not contain file paths")
    for path in paths:
        ensure_path_allowed(repo_root, path, allowed_paths)
    return paths


def apply_patch_text(repo_root: Path, patch_text: str, allowed_paths: list[str] | None = None) -> list[str]:
    changed_paths = validate_patch(repo_root, patch_text, allowed_paths)
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=repo_root,
        input=patch_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git apply failed:\n{result.stderr}")
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
