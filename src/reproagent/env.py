"""Conda execution backend.

The MVP is conda-first: every task gets one isolated conda environment. LLM
commands stay plain shell commands; this module wraps them with `conda run`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .models import EnvironmentInfo, ReproState


def ensure_environment(state: ReproState) -> EnvironmentInfo:
    """Create or reuse the task conda environment."""
    assert state.repo_context is not None
    conda = find_conda()
    if conda is None:
        raise RuntimeError("conda was not found on PATH or common install paths. Install Miniconda/Anaconda in Ubuntu-D before running reproagent.")

    env_name = _env_name(state.task.task_id)
    logs = state.task.workspace_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "conda_setup.stdout"
    stderr_path = logs / "conda_setup.stderr"

    if _conda_env_exists(conda, env_name):
        info = EnvironmentInfo(env_name=env_name, created=False, setup_stdout_path=stdout_path, setup_stderr_path=stderr_path)
        stdout_path.write_text(f"conda env already exists: {env_name}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return info

    env_file = _find_environment_yml(state.repo_context.repo_path)
    if env_file is not None:
        cmd = [conda, "env", "create", "-n", env_name, "-f", str(env_file)]
        used_yml = True
    else:
        cmd = [conda, "create", "-n", env_name, f"python={state.task.python_version}", "-y"]
        used_yml = False

    result = subprocess.run(cmd, cwd=str(state.repo_context.repo_path), text=True, capture_output=True, timeout=state.task.timeout_seconds)
    stdout_path.write_text(result.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(result.stderr or "", encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"conda environment setup failed for {env_name}; see {stderr_path}")

    return EnvironmentInfo(
        env_name=env_name,
        created=True,
        used_environment_yml=used_yml,
        setup_command=" ".join(cmd),
        setup_stdout_path=stdout_path,
        setup_stderr_path=stderr_path,
    )


def build_backend_command(env_name: str, command: str, conda: str | None = None) -> list[str]:
    """Wrap a plain shell command so it runs inside the task conda env."""
    conda_exe = conda or find_conda() or "conda"
    return [conda_exe, "run", "-n", env_name, "bash", "-c", command]


def find_conda() -> str | None:
    """Find conda in PATH or common Ubuntu-D install locations."""
    found = shutil.which("conda")
    if found:
        return found
    for candidate in (
        Path.home() / "miniconda3" / "bin" / "conda",
        Path("/home/cyl/miniconda3/bin/conda"),
        Path("/opt/conda/bin/conda"),
    ):
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def _env_name(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", task_id)
    return f"repro_{safe}"[:80]


def _find_environment_yml(repo_path: Path) -> Path | None:
    for name in ("environment.yml", "environment.yaml", "conda.yml", "conda.yaml"):
        path = repo_path / name
        if path.exists():
            return path
    return None


def _conda_env_exists(conda: str, env_name: str) -> bool:
    result = subprocess.run([conda, "env", "list", "--json"], text=True, capture_output=True, timeout=60)
    if result.returncode != 0:
        return False
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    for env_path in data.get("envs", []):
        if Path(env_path).name == env_name:
            return True
    return False
