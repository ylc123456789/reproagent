"""Command execution utilities."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .env import build_backend_command, find_conda
from .models import CommandResult

BLOCKED_SNIPPETS = ["sudo", "rm -rf", "curl", "wget", "| bash", "> /", "shutdown", "reboot", "conda activate"]
ENV_STAGE_LONG_RUNNING_HINTS = ["examples/", "train.py", "demo.py", "mnist", "epoch", "--epochs"]
EXPERIMENT_BROAD_HINTS = ["tests/run_all.py", "run_all.py"]


def is_safe_command(command: str, stage: str | None = None) -> tuple[bool, str | None]:
    lowered = command.lower()
    for bad in BLOCKED_SNIPPETS:
        if bad in lowered:
            return False, f"blocked unsafe snippet: {bad}"
    if _has_parent_directory_traversal(command):
        return False, "blocked parent-directory traversal"
    if stage == "environment":
        for hint in ENV_STAGE_LONG_RUNNING_HINTS:
            if hint in lowered:
                return False, f"environment stage should not run experiment/demo/training command: {hint}"
    if stage == "experiment":
        for hint in EXPERIMENT_BROAD_HINTS:
            if hint in lowered:
                return False, f"experiment stage should start with a targeted small test, not broad command: {hint}"
        if _is_bare_pytest(lowered):
            return False, "experiment stage should target a small test file, not the whole pytest suite"
    return True, None


def _is_bare_pytest(lowered_command: str) -> bool:
    normalized = " ".join(lowered_command.split())
    return normalized in {"pytest", "python -m pytest"}


def _has_parent_directory_traversal(command: str) -> bool:
    normalized = command.replace("\\", "/")
    tokens = normalized.split()
    return any(token == ".." or token.startswith("../") or "/../" in token for token in tokens)


def run_commands(commands: list[str], cwd: Path, workspace: Path, stage: str, attempt: int, timeout: int, env_name: str) -> list[CommandResult]:
    results: list[CommandResult] = []
    for index, command in enumerate(commands, start=1):
        ok, reason = is_safe_command(command, stage=stage)
        if not ok:
            results.append(_write_blocked_result(command, workspace, stage, attempt, index, reason or "blocked"))
            continue
        results.append(_run_one(command, cwd, workspace, stage, attempt, index, timeout, env_name))
    return results


def _run_one(command: str, cwd: Path, workspace: Path, stage: str, attempt: int, index: int, timeout: int, env_name: str) -> CommandResult:
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = f"{stage}_{attempt:02d}_{index:02d}"
    stdout_path = logs / f"{prefix}.stdout"
    stderr_path = logs / f"{prefix}.stderr"
    backend_command = build_backend_command(env_name, command, conda=find_conda())
    start = time.monotonic()
    try:
        result = subprocess.run(backend_command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout, env=_command_env(workspace))
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        code = result.returncode
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = f"Command timed out after {timeout}s"
        code = -1
    duration = round(time.monotonic() - start, 2)
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    return CommandResult(command=command, exit_code=code, stdout_path=stdout_path, stderr_path=stderr_path, duration_seconds=duration, backend_command=backend_command)


def _command_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    tmp_dir = workspace / ".tmp"
    pip_cache_dir = workspace / ".cache" / "pip"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pip_cache_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["PIP_CACHE_DIR"] = str(pip_cache_dir)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    omp = env.get("OMP_NUM_THREADS", "").strip()
    if not omp.isdigit() or int(omp) <= 0:
        env["OMP_NUM_THREADS"] = "16"
    return env


def _write_blocked_result(command: str, workspace: Path, stage: str, attempt: int, index: int, reason: str) -> CommandResult:
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = f"{stage}_{attempt:02d}_{index:02d}"
    stdout_path = logs / f"{prefix}.stdout"
    stderr_path = logs / f"{prefix}.stderr"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text(f"Command blocked by safety policy: {reason}\n", encoding="utf-8")
    return CommandResult(command=command, exit_code=-2, stdout_path=stdout_path, stderr_path=stderr_path, duration_seconds=0.0)
