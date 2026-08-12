"""Command execution utilities."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .environment import build_backend_command, find_conda
from ..models import CommandResult

BLOCKED_SNIPPETS = ["sudo", "rm -rf", "| bash", "> /", "shutdown", "reboot", "conda activate"]
COMMAND_HEARTBEAT_SECONDS = 60

_INTERNAL_ACTIONS = frozenset({
    "audit_env",
    "call_coding_agent",
    "finish",
})


def is_safe_command(command: str, stage: str | None = None) -> tuple[bool, str | None]:
    """Return whether a command is allowed by the runner safety policy."""
    lowered = command.lower()
    for bad in BLOCKED_SNIPPETS:
        if bad in lowered:
            return False, f"blocked unsafe snippet: {bad}"
    if _has_parent_directory_traversal(command):
        return False, "blocked parent-directory traversal"
    if stage == "probe" and not _is_probe_command(command):
        return False, "probe stage only allows help/config/listing/inline inspection commands"
    return True, None


def _normalize_command(command: str) -> str:
    """Normalize normalize command."""
    return command.replace(".**version**", ".__version__")


def _is_probe_command(command: str) -> bool:
    """Return whether a command is safe probe-only inspection."""
    lowered = command.strip().lower()
    if "--help" in lowered or lowered.endswith(" -h") or " -h " in lowered:
        return True
    if lowered.startswith(("python -m py_compile ", "python3 -m py_compile ")):
        return True
    allowed_prefixes = ("ls", "find", "rg", "grep", "sed", "cat", "head", "tail", "wc", "pwd",
                        "which ", "python --version", "python3 --version",
                        "python -c", "python3 -c")
    return lowered.startswith(allowed_prefixes)


def _has_parent_directory_traversal(command: str) -> bool:
    """Return whether has parent directory traversal."""
    normalized = command.replace("\\", "/")
    tokens = normalized.split()
    return any(token == ".." or token.startswith("../") or "/../" in token for token in tokens)


def run_commands(commands: list[str], cwd: Path, workspace: Path, stage: str, attempt: int, timeout: int, env_name: str) -> list[CommandResult]:
    """Run a list of commands and collect their results."""
    results: list[CommandResult] = []
    for index, command in enumerate(commands, start=1):
        if command.strip() in _INTERNAL_ACTIONS:
            results.append(_write_blocked_result(
                command, workspace, stage, attempt, index,
                f"'{command}' is an agent action, not a shell command. Use the '{command}' tool instead.",
            ))
            continue
        ok, reason = is_safe_command(command, stage=stage)
        if not ok:
            results.append(_write_blocked_result(command, workspace, stage, attempt, index, reason or "blocked"))
            continue
        results.append(_run_one(_normalize_command(command), cwd, workspace, stage, attempt, index, timeout, env_name))
    return results


def _run_one(command: str, cwd: Path, workspace: Path, stage: str, attempt: int, index: int, timeout: int, env_name: str) -> CommandResult:
    """Run one command with logging, timeout, and safety checks."""
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = f"{stage}_{attempt:02d}_{index:02d}"
    stdout_path = logs / f"{prefix}.stdout"
    stderr_path = logs / f"{prefix}.stderr"
    backend_command = build_backend_command(env_name, command, conda=find_conda())
    print(f"[reproagent] running {stage} command {attempt}.{index}: {command}", flush=True)
    print(f"[reproagent] logs: stdout={stdout_path}, stderr={stderr_path}", flush=True)
    start = time.monotonic()
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    code = -1
    process: subprocess.Popen[str] | None = None
    threads: list[threading.Thread] = []
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
        with stdout_path.open("a", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open("a", encoding="utf-8", errors="replace") as stderr_file:
            threads = [
                threading.Thread(target=_stream_pipe, args=(process.stdout, sys.stdout, stdout_file, stage), daemon=True),
                threading.Thread(target=_stream_pipe, args=(process.stderr, sys.stderr, stderr_file, stage), daemon=True),
            ]
            for thread in threads:
                thread.start()
            code = _wait_for_process(process, timeout, stage, attempt, index, stdout_path, stderr_path)
            # Drain BEFORE closing: after process exit the pipes hit EOF and
            # the stream threads finish on their own. Closing the streams
            # first races with _stream_pipe's reads and silently drops the
            # buffered tail of the output ("I/O operation on closed file").
            for thread in threads:
                thread.join(timeout=5)
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
        for thread in threads:
            thread.join(timeout=5)
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        message = f"Command timed out after {timeout}s\n"
        with stderr_path.open("a", encoding="utf-8", errors="replace") as stderr_file:
            stderr_file.write(message)
            stderr_file.flush()
        print(message, file=sys.stderr, end="", flush=True)
        code = -1
    duration = round(time.monotonic() - start, 2)
    print(f"[reproagent] finished {stage} command {attempt}.{index}: exit={code}, duration={duration}s", flush=True)
    return CommandResult(command=command, exit_code=code, stdout_path=stdout_path, stderr_path=stderr_path, duration_seconds=duration, backend_command=backend_command)


def _wait_for_process(process: subprocess.Popen[str], timeout: int, stage: str, attempt: int, index: int, stdout_path: Path, stderr_path: Path) -> int:
    """Wait for a process while printing periodic liveness heartbeats."""
    deadline = time.monotonic() + timeout
    next_heartbeat = time.monotonic() + COMMAND_HEARTBEAT_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(1.0, COMMAND_HEARTBEAT_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            now = time.monotonic()
            if stage == "experiment" and now >= next_heartbeat:
                stdout_size = stdout_path.stat().st_size if stdout_path.exists() else 0
                stderr_size = stderr_path.stat().st_size if stderr_path.exists() else 0
                elapsed = round(timeout - remaining)
                print(
                    f"[reproagent] still running {stage} command {attempt}.{index}: "
                    f"elapsed={elapsed}s, stdout={stdout_size}B, stderr={stderr_size}B",
                    flush=True,
                )
                next_heartbeat = now + COMMAND_HEARTBEAT_SECONDS


def _stream_pipe(pipe, display, log_file, stage: str) -> None:
    """Stream subprocess output to a log file."""
    if pipe is None:
        return
    pending: list[str] = []
    try:
        while True:
            try:
                char = pipe.read(1)
            except (ValueError, OSError):
                break  # pipe closed concurrently
            if char == "":
                break
            try:
                log_file.write(char)
                log_file.flush()
            except (ValueError, OSError):
                break
            pending.append(char)
            if char in ("\n", "\r"):
                _display_chunk_if_relevant(stage, "".join(pending), display)
                pending.clear()
        if pending:
            _display_chunk_if_relevant(stage, "".join(pending), display)
    finally:
        pipe.close()


def _display_chunk_if_relevant(stage: str, chunk: str, display) -> None:
    """Display useful live output chunks."""
    if _should_display_line(stage, chunk):
        print(chunk, file=display, end="", flush=True)


def _should_display_line(stage: str, line: str) -> bool:
    """Return whether a line should be shown live."""
    return stage == "experiment"


def _pip_cache_dir(workspace: Path, env: dict[str, str]) -> Path:
    """Resolve the pip cache directory.

    Priority:
      1. REPROAGENT_PIP_CACHE (explicit)
      2. sibling of REPROAGENT_DATASET_CACHE (e.g. /root/autodl-tmp/datasets
         -> /root/autodl-tmp/pip-cache) — zero-config on machines that
         already have the dataset cache set
      3. per-workspace fallback (previous behaviour)

    A shared cache means large wheels (torch ~500MB+) are downloaded at most
    once per mirror source, instead of once per workspace.
    """
    explicit = env.get("REPROAGENT_PIP_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    dataset_cache = env.get("REPROAGENT_DATASET_CACHE", "").strip()
    if dataset_cache:
        return Path(dataset_cache).expanduser().parent / "pip-cache"
    return workspace / ".cache" / "pip"


def _command_env(workspace: Path) -> dict[str, str]:
    """Build the subprocess environment for a command."""
    env = os.environ.copy()
    pip_cache_dir = _pip_cache_dir(workspace, env)
    try:
        pip_cache_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        # shared location not writable — fall back to per-workspace cache
        pip_cache_dir = workspace / ".cache" / "pip"
        pip_cache_dir.mkdir(parents=True, exist_ok=True)
    env["PIP_CACHE_DIR"] = str(pip_cache_dir)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    omp = env.get("OMP_NUM_THREADS", "").strip()
    if not omp.isdigit() or int(omp) <= 0:
        cpu_count = os.cpu_count() or 4
        env["OMP_NUM_THREADS"] = str(min(16, cpu_count))
    cache = env.get("REPROAGENT_DATASET_CACHE", "")
    if cache:
        try:
            if Path(cache).is_dir():
                for key in ("TORCH_HOME", "HF_HOME", "TORCHVISION_DATASETS", "HUGGINGFACE_HUB_CACHE"):
                    env.setdefault(key, cache)
        except (OSError, PermissionError):
            pass
    return env


def _write_blocked_result(command: str, workspace: Path, stage: str, attempt: int, index: int, reason: str) -> CommandResult:
    """Write write blocked result."""
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = f"{stage}_{attempt:02d}_{index:02d}"
    stdout_path = logs / f"{prefix}.stdout"
    stderr_path = logs / f"{prefix}.stderr"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text(f"Command blocked by safety policy: {reason}\n", encoding="utf-8")
    return CommandResult(command=command, exit_code=-2, stdout_path=stdout_path, stderr_path=stderr_path, duration_seconds=0.0)
