"""Conda execution backend.

The MVP is conda-first: every task gets one isolated conda environment. LLM
commands stay plain shell commands; this module wraps them with `conda run`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..models import EnvironmentInfo, ReproState


def ensure_environment(state: ReproState) -> EnvironmentInfo:
    """Create or reuse the task conda environment."""
    assert state.repo_context is not None
    conda = find_conda()
    if conda is None:
        raise RuntimeError("conda was not found. Set REPROAGENT_CONDA_EXE or install Miniconda/Anaconda so conda is on PATH.")

    logs = state.task.workspace_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "conda_setup.stdout"
    stderr_path = logs / "conda_setup.stderr"

    if state.task.env_name:
        # Explicit binding: use the referenced environment unchanged.
        # Never create a substitute; unresolvable references fail loudly.
        if _find_conda_env(conda, state.task.env_name) is None:
            raise RuntimeError(
                f"Explicit environment {state.task.env_name!r} was not found. "
                "Provide an existing conda env name or absolute prefix, or "
                "leave env_name empty to let reproagent create one."
            )
        stdout_path.write_text(f"explicit environment bound: {state.task.env_name}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return EnvironmentInfo(
            env_name=state.task.env_name,
            created=False,
            setup_stdout_path=stdout_path,
            setup_stderr_path=stderr_path,
        )

    env_name = _env_name(state.task.task_id, namespace=state.task.env_namespace, isolate=state.task.isolate_env)

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

    result = _run_conda_setup_with_retries(
        cmd,
        cwd=state.repo_context.repo_path,
        timeout=state.task.timeout_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
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



def _run_conda_setup_with_retries(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
    attempts: int = 3,
    delay_seconds: float = 3.0,
) -> subprocess.CompletedProcess[str]:
    """Run conda setup commands with transient-error retries."""
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
        last_result = result
        stdout_chunks.append(f"===== conda setup attempt {attempt}/{attempts} =====\n")
        stdout_chunks.append(result.stdout or "")
        stderr_chunks.append(f"===== conda setup attempt {attempt}/{attempts} =====\n")
        stderr_chunks.append(result.stderr or "")
        stdout_path.write_text("".join(stdout_chunks), encoding="utf-8", errors="replace")
        stderr_path.write_text("".join(stderr_chunks), encoding="utf-8", errors="replace")
        if result.returncode == 0:
            return result
        if attempt < attempts and _is_transient_conda_setup_error(result.stderr or ""):
            time.sleep(delay_seconds)
            continue
        return result
    assert last_result is not None
    return last_result


def _is_transient_conda_setup_error(stderr: str) -> bool:
    """Return whether conda setup failed for a transient reason."""
    lowered = stderr.lower()
    markers = (
        "condahttperror",
        "http 502",
        "http 503",
        "http 504",
        "bad gateway",
        "gateway timeout",
        "remote server error",
        "connection aborted",
        "connection reset",
        "read timed out",
    )
    return any(marker in lowered for marker in markers)



def conda_run_flag(env_ref: str) -> str:
    """Return the conda run flag for an environment reference.

    One documented rule: absolute paths are conda PREFIXES (`-p`);
    everything else is a NAME (`-n`). Shared by command execution, the
    environment audit, and CodingAgent verification wrapping.
    """
    return "-p" if Path(env_ref).expanduser().is_absolute() else "-n"


def build_backend_command(env_name: str, command: str, conda: str | None = None) -> list[str]:
    """Wrap a plain shell command so it runs inside the task conda env."""
    conda_exe = conda or find_conda() or "conda"
    return [conda_exe, "run", "--no-capture-output", conda_run_flag(env_name), env_name,
            "bash", "-o", "pipefail", "-c", command]


def find_conda() -> str | None:
    """Find conda from explicit config, PATH, or common install locations."""
    configured = os.environ.get("REPROAGENT_CONDA_EXE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    found = shutil.which("conda")
    if found:
        return found
    for candidate in (
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
        Path("/opt/conda/bin/conda"),
    ):
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def _env_name(task_id: str, namespace: str = "", isolate: bool = False) -> str:
    """Create a unique conda environment name.

    When namespace is provided (e.g. a ResAgent run_id), the env is shared
    across tasks within that project: ``resenv_<sanitized_namespace>``.
    When isolate_env is True, falls back to per-task naming even with a
    namespace.
    """
    if not isolate and namespace:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", namespace)
        return f"resenv_{safe}"[:80]
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", task_id)
    return f"repro_{safe}"[:80]


def _find_environment_yml(repo_path: Path) -> Path | None:
    """Find a repository-provided conda environment file."""
    for name in ("environment.yml", "environment.yaml", "conda.yml", "conda.yaml"):
        path = repo_path / name
        if path.exists():
            return path
    return None


def _conda_env_exists(conda: str, env_name: str) -> bool:
    """Return whether a conda environment already exists."""
    return _find_conda_env(conda, env_name) is not None


def _find_conda_env(conda: str, env_ref: str) -> str | None:
    """Resolve an environment reference to a conda env path, or None.

    One documented rule: an absolute path is matched as a conda PREFIX;
    anything else is matched as an environment NAME.
    """
    result = subprocess.run([conda, "env", "list", "--json"], text=True, capture_output=True, timeout=60)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    ref = Path(env_ref).expanduser()
    for env_path in data.get("envs", []):
        candidate = Path(env_path)
        if ref.is_absolute():
            if candidate == ref or candidate.resolve() == ref.resolve():
                return env_path
        elif candidate.name == env_ref:
            return env_path
    return None
