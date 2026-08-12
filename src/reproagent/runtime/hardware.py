"""Collect lightweight hardware context for the LLM."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess


def collect_hardware_text(timeout: int = 20) -> str:
    """Return a compact hardware summary.

    This is observational only. It does not configure CUDA or schedule GPUs.
    """
    parts: list[str] = []
    parts.append(f"OS: {platform.platform()}")
    parts.append(f"CPU cores visible: {os.cpu_count() or 'unknown'}")
    mem = _memory_summary()
    if mem:
        parts.append(mem)
    smi = _nvidia_smi(timeout=timeout)
    if smi:
        parts.append("nvidia-smi:\n" + smi)
    else:
        parts.append("nvidia-smi: not available or no NVIDIA GPU visible")
    parts.append(
        "Policy: the final goal is faithful reproduction of the paper results. "
        "If an NVIDIA GPU is visible, configure ML dependencies so GPU execution is available before running experiments. "
        "Choose PyTorch/JAX/TensorFlow builds compatible with the reported driver/CUDA capability instead of blindly installing the newest build. "
        "Use small smoke/demo/eval commands to validate the environment before any expensive training."
    )
    return "\n".join(parts)


def _memory_summary() -> str | None:
    """Return a short system memory summary."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            data = f.read().splitlines()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in data:
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(rest.strip().split()[0])
    if not values:
        return None
    chunks = []
    if "MemTotal" in values:
        chunks.append(f"RAM total: {values['MemTotal'] / 1024 / 1024:.1f} GB")
    if "MemAvailable" in values:
        chunks.append(f"RAM available: {values['MemAvailable'] / 1024 / 1024:.1f} GB")
    return ", ".join(chunks)


def _nvidia_smi(timeout: int) -> str | None:
    """Return nvidia-smi output when available."""
    exe = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not exe or not os.path.exists(exe):
        return None
    query = [
        exe,
        "--query-gpu=name,memory.total,memory.free,driver_version,cuda_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(query, text=True, capture_output=True, timeout=timeout)
    except Exception:
        return None
    if result.returncode == 0 and result.stdout.strip():
        lines = []
        for idx, row in enumerate(result.stdout.strip().splitlines()):
            cols = [c.strip() for c in row.split(",")]
            if len(cols) >= 5:
                name, total, free, driver, cuda = cols[:5]
                lines.append(
                    f"GPU {idx}: {name}, VRAM total {total} MiB, "
                    f"VRAM free {free} MiB, driver {driver}, CUDA {cuda}"
                )
            else:
                lines.append(f"GPU {idx}: {row}")
        return "\n".join(lines)
    # Fall back to regular nvidia-smi output tail/head if query format fails.
    try:
        fallback = subprocess.run([exe], text=True, capture_output=True, timeout=timeout)
    except Exception:
        return None
    if fallback.returncode == 0 and fallback.stdout.strip():
        return fallback.stdout.strip()[:4000]
    return None
