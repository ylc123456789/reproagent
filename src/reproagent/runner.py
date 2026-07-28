"""Command execution utilities."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .env import build_backend_command, find_conda
from .models import CommandResult

BLOCKED_SNIPPETS = ["sudo", "rm -rf", "| bash", "> /", "shutdown", "reboot", "conda activate"]


def is_safe_command(command: str, stage: str | None = None) -> tuple[bool, str | None]:
    lowered = command.lower()
    for bad in BLOCKED_SNIPPETS:
        if bad in lowered:
            return False, f"blocked unsafe snippet: {bad}"
    if _has_parent_directory_traversal(command):
        return False, "blocked parent-directory traversal"
    return True, None


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
    print(f"[reproagent] running {stage} command {attempt}.{index}: {command}", flush=True)
    start = time.monotonic()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    code = -1
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            backend_command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            env=_command_env(workspace),
        )
        threads = [
            threading.Thread(target=_stream_pipe, args=(process.stdout, sys.stdout, stdout_chunks), daemon=True),
            threading.Thread(target=_stream_pipe, args=(process.stderr, sys.stderr, stderr_chunks), daemon=True),
        ]
        for thread in threads:
            thread.start()
        code = process.wait(timeout=timeout)
        for thread in threads:
            thread.join(timeout=1)
    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        message = f"Command timed out after {timeout}s\n"
        stderr_chunks.append(message)
        print(message, file=sys.stderr, end="", flush=True)
        code = -1
    duration = round(time.monotonic() - start, 2)
    stdout_path.write_text("".join(stdout_chunks), encoding="utf-8", errors="replace")
    stderr_path.write_text("".join(stderr_chunks), encoding="utf-8", errors="replace")
    print(f"[reproagent] finished {stage} command {attempt}.{index}: exit={code}, duration={duration}s", flush=True)
    return CommandResult(command=command, exit_code=code, stdout_path=stdout_path, stderr_path=stderr_path, duration_seconds=duration, backend_command=backend_command)


def _stream_pipe(pipe, display, chunks: list[str]) -> None:
    if pipe is None:
        return
    try:
        for line in pipe:
            chunks.append(line)
            print(line, file=display, end="", flush=True)
    finally:
        pipe.close()


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
