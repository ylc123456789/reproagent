"""Run verification commands and capture their logs."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .models import CommandResult
from .safety import validate_command


def run_verify_commands(
    repo_root: Path,
    commands: list[str],
    log_dir: Path,
    timeout_seconds: int,
) -> list[CommandResult]:
    """Run verification commands with captured logs."""
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[CommandResult] = []
    for index, command in enumerate(commands, start=1):
        validate_command(command)
        stdout_path = log_dir / f"verify_{index:02d}.stdout"
        stderr_path = log_dir / f"verify_{index:02d}.stderr"
        start = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        duration = time.monotonic() - start
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        results.append(
            CommandResult(
                command=command,
                returncode=returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                duration_seconds=duration,
                timed_out=timed_out,
            )
        )
    return results
