"""Enforce path and command safety constraints."""
from __future__ import annotations

import shlex
from pathlib import Path


BLOCKED_PATH_PARTS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "data",
    "datasets",
    "checkpoints",
    "weights",
    "models",
}

BLOCKED_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
}


class SafetyError(ValueError):
    """Raised when a path or command violates safety policy."""
    pass


def ensure_repo_relative(path: str) -> str:
    """Resolve and validate a repository-relative path."""
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        raise SafetyError(f"absolute paths are not allowed in patches: {path}")
    if normalized.startswith("../") or "/../" in normalized or normalized == "..":
        raise SafetyError(f"path traversal is not allowed in patches: {path}")
    return normalized.removeprefix("./")


def ensure_path_allowed(repo_root: Path, relative_path: str, allowed_paths: list[str] | None = None) -> Path:
    """Validate a path against safety and allow-list rules."""
    safe_rel = ensure_repo_relative(relative_path)
    parts = set(Path(safe_rel).parts)
    suffix = Path(safe_rel).suffix.lower()
    if parts & BLOCKED_PATH_PARTS:
        raise SafetyError(f"blocked path segment in patch: {safe_rel}")
    if suffix in BLOCKED_SUFFIXES:
        raise SafetyError(f"blocked file type in patch: {safe_rel}")
    if allowed_paths:
        allowed = [ensure_repo_relative(item).rstrip("/") for item in allowed_paths]
        if not any(safe_rel == item or safe_rel.startswith(f"{item}/") for item in allowed):
            raise SafetyError(f"path is outside allowed_paths: {safe_rel}")
    resolved = (repo_root / safe_rel).resolve()
    if repo_root.resolve() not in [resolved, *resolved.parents]:
        raise SafetyError(f"path escapes repo root: {safe_rel}")
    return resolved


def validate_command(command: str) -> None:
    """Reject dangerous shell commands."""
    lowered = command.lower()
    blocked_fragments = [
        "rm -rf",
        "sudo ",
        "chmod -r",
        "chown -r",
        "shutdown",
        "reboot",
        "curl",
        "wget",
    ]
    if any(fragment in lowered for fragment in blocked_fragments):
        if ("curl" in lowered or "wget" in lowered) and "| bash" not in lowered and "| sh" not in lowered:
            return
        raise SafetyError(f"blocked verification command: {command}")
    try:
        shlex.split(command)
    except ValueError as exc:
        raise SafetyError(f"invalid shell command: {command}") from exc
