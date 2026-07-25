"""Command execution utilities."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .env import build_backend_command, find_conda
from .models import CommandResult

BLOCKED_SNIPPETS = ["sudo", "rm -rf", "curl", "wget", "| bash", "> /", "shutdown", "reboot", "conda activate"]
ENV_STAGE_LONG_RUNNING_HINTS = ["examples/", "train.py", "demo.py", "mnist", "epoch", "--epochs"]


def is_safe_command(command: str, stage: str | None = None) -> tuple[bool, str | None]:
    lowered = command.lower()
    for bad in BLOCKED_SNIPPETS:
        if bad in lowered:
            return False, f"blocked unsafe snippet: {bad}"
    if ".." in command:
        return False, "blocked parent-directory traversal"
    if stage == "environment":
        for hint in ENV_STAGE_LONG_RUNNING_HINTS:
            if hint in lowered:
                return False, f"environment stage should not run experiment/demo/training command: {hint}"
    return True, None


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
        result = subprocess.run(backend_command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
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


def _write_blocked_result(command: str, workspace: Path, stage: str, attempt: int, index: int, reason: str) -> CommandResult:
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = f"{stage}_{attempt:02d}_{index:02d}"
    stdout_path = logs / f"{prefix}.stdout"
    stderr_path = logs / f"{prefix}.stderr"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text(f"Command blocked by safety policy: {reason}\n", encoding="utf-8")
    return CommandResult(command=command, exit_code=-2, stdout_path=stdout_path, stderr_path=stderr_path, duration_seconds=0.0)
