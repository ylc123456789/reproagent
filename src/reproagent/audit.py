"""Post-setup environment audit."""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from .env import build_backend_command, find_conda
from .models import EnvironmentAudit, ReproState
from .runner import _command_env


PROBE_CODE = """
import json
import shutil
import subprocess
import sys

data = {
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "python_version": sys.version,
    "which_python": shutil.which("python"),
    "which_pip": shutil.which("pip"),
}
try:
    pip = subprocess.run([sys.executable, "-m", "pip", "--version"], text=True, capture_output=True, timeout=30)
    data["pip_version"] = pip.stdout.strip() or pip.stderr.strip()
except Exception as exc:
    data["pip_version_error"] = str(exc)
for lib, check_gpu in ((torch, True), (tensorflow, False), (jax, False)):
    try:
        mod = __import__(lib)
        info = {version: getattr(mod, __version__, None)}
        if check_gpu:
            info[cuda_compiled] = getattr(getattr(mod, version, None), cuda, None)
            info[cuda_available] = bool(mod.cuda.is_available())
            info[device_count] = mod.cuda.device_count()
        data[lib] = info
    except Exception:
        pass
print(json.dumps(data, indent=2, ensure_ascii=False))
"""

def audit_environment(state: ReproState) -> EnvironmentAudit:
    """Inspect the prepared conda env and summarize important mismatches."""
    assert state.environment is not None
    logs = state.task.workspace_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "environment_audit.stdout"
    stderr_path = logs / "environment_audit.stderr"

    probe = "python -c " + shlex.quote(PROBE_CODE)
    backend_command = build_backend_command(state.environment.env_name, probe, conda=find_conda())
    result = subprocess.run(
        backend_command,
        cwd=str(state.repo_context.repo_path if state.repo_context else state.task.workspace_dir),
        text=True,
        capture_output=True,
        timeout=min(state.task.timeout_seconds, 180),
        env=_command_env(state.task.workspace_dir),
    )
    stdout_path.write_text(result.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(result.stderr or "", encoding="utf-8", errors="replace")

    details: list[str] = []
    has_warnings = False
    requires_repair = False
    gpu_visible = _gpu_visible(state)
    success = result.returncode == 0
    summary = "Environment audit completed." if success else "Environment audit command failed."
    if success:
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            data = {}
            success = False
            summary = "Environment audit output was not valid JSON."
        if data:
            executable = str(data.get("sys_executable") or "")
            env_prefix = _infer_env_prefix(str(data.get("sys_prefix") or ""), executable)
            pip_version = str(data.get("pip_version") or "")
            details.append(f"Python executable: {executable or 'unknown'}")
            details.append(f"Pip: {pip_version or 'unknown'}")
            env_name_matches = env_prefix.name == state.environment.env_name
            if not env_name_matches:
                success = False
                details.append(f"Mismatch: sys.prefix does not match expected conda env {state.environment.env_name}: {env_prefix}")
            if executable and (not env_name_matches or not _path_is_under(executable, env_prefix)):
                success = False
                details.append(f"Mismatch: python executable is not inside expected conda env {state.environment.env_name}.")
            if pip_version and str(env_prefix) not in pip_version:
                details.append(f"Mismatch: pip is not inside expected conda env {state.environment.env_name}.")
            torch_info = data.get("torch")
            if isinstance(torch_info, dict):
                details.append(
                    "Torch: "
                    f"version={torch_info.get('version')}, "
                    f"compiled_cuda={torch_info.get('cuda_compiled')}, "
                    f"cuda_available={torch_info.get('cuda_available')}, "
                    f"device_count={torch_info.get('device_count')}"
                )
                torch_file = str(torch_info.get("file") or "")
                if torch_file and not _path_is_under(torch_file, env_prefix):
                    success = False
                    details.append(f"Mismatch: torch is loaded outside expected conda env {state.environment.env_name}.")
                if gpu_visible and not torch_info.get("cuda_available"):
                    has_warnings = True
                    requires_repair = True
                    details.append("GPU repair required: NVIDIA GPU is visible, but torch.cuda.is_available() is false.")
                    if torch_info.get("cuda_compiled"):
                        details.append("Likely CUDA/driver mismatch: installed torch has CUDA support but cannot initialize CUDA on this machine.")
                    else:
                        details.append("Likely CPU-only torch install: choose a GPU-capable build compatible with this machine.")
            elif data.get("torch_error"):
                details.append(f"Torch import: unavailable ({data['torch_error']})")
            stderr_text = (result.stderr or "").lower()
            if "numpy 1.x" in stderr_text or "_array_api not found" in stderr_text or "numpy is not available" in stderr_text:
                has_warnings = True
                requires_repair = True
                details.append("Dependency repair required: torch emitted a NumPy ABI compatibility warning; pin numpy<2 or choose a compatible torch/numpy pair.")
            if success and requires_repair:
                summary = "Environment audit requires repair."
            elif success and has_warnings:
                summary = "Environment audit passed with warnings."
            else:
                summary = "Environment audit passed." if success else "Environment audit found issues."

    return EnvironmentAudit(
        success=success,
        summary=summary,
        details=details,
        has_warnings=has_warnings,
        requires_repair=requires_repair,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

def _gpu_visible(state: ReproState) -> bool:
    """Return whether CUDA appears visible in audit output."""
    if not state.repo_context:
        return False
    hardware = state.repo_context.hardware_text.lower()
    return "gpu 0:" in hardware or ("nvidia-smi:" in hardware and "not available" not in hardware)

def _infer_env_prefix(sys_prefix: str, executable: str) -> Path:
    """Infer the current conda environment prefix."""
    if sys_prefix:
        return Path(sys_prefix)
    executable_path = Path(executable)
    if executable_path.parent.name == "bin":
        return executable_path.parent.parent
    return Path("")

def _path_is_under(path: str, parent: Path) -> bool:
    """Return whether a path resolves under a parent directory."""
    if not path or not str(parent):
        return False
    try:
        Path(path).resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
