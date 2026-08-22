"""Deterministic environment identity — ENVIRONMENT_SPEC_V1.

The fingerprint algorithm is byte-identical to the frozen P0 reference
(ResAgent contracts/README.md + scripts/m2_contract_check.py): canonical
JSON (sorted keys, ASCII-safe, no insignificant whitespace) over the
identity-bearing field subset, then SHA-256.  Cross-repo byte consistency
is a milestone-2 hard requirement — do NOT "improve" this algorithm
locally; any change must first bump the contract version.

LLM never participates: spec collection and fingerprints are pure code.
"""
from __future__ import annotations

import platform
from pathlib import Path

from ..models import ReproTask

# ── canonical serialization and fingerprints ──────────────────────
# The algorithms live in ONE place: the vendored contract file (byte-
# identical across the three repos; tests assert the sha256). This module
# keeps only task-specific collection logic and re-exports the contract
# functions for existing callers.
from .._vendor import env_contract_v1 as _contract

canonical_dumps = _contract.canonical_dumps
sha256_hex = _contract.sha256_hex
identity_subset = _contract.identity_subset
spec_fingerprint = _contract.spec_fingerprint
resolved_fingerprint = _contract.resolved_fingerprint
slug_project = _contract.project_slug
env_id = _contract.env_id


def project_slug_for(task: ReproTask) -> str:
    """Project slug source (§4.3): orchestrated mode uses project_ref;
    standalone mode uses the repo basename WITHOUT its path, so the same
    repo yields the same slug from any location."""
    source = (task.project_ref or "").strip()
    if not source:
        for candidate in (task.repo_url, task.copy_from, task.external_repo_path):
            if candidate:
                basename = Path(candidate.rstrip("/")).name
                source = basename[:-4] if basename.endswith(".git") else basename
                break
    return slug_project(source)


# ── spec collection ───────────────────────────────────────────────

def _os_family() -> str:
    system = platform.system().lower()
    return system if system in ("linux", "macos", "windows") else "linux"


def _arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def _accelerator_spec(task, repo_path: Path, *, probe_log=None) -> dict:
    """Accelerator identity per the frozen collection semantics:

    - type: requires_gpu AND a usable local GPU probe → cuda, else cpu
    - variant: explicit task/dependency constraints only, else ""
    - a failed probe with requires_gpu=True is a FEASIBILITY warning
      written to the setup log — never a silent downgrade.
    """
    variant = _contract.constraint_cuda_variant(repo_path)
    if not getattr(task, "requires_gpu", False):
        return {"type": "cpu", "variant": variant}
    if _contract.probe_gpu_usable():
        return {"type": "cuda", "variant": variant}
    if probe_log is not None:
        try:
            with Path(probe_log).open("a", encoding="utf-8") as handle:
                handle.write(
                    "accelerator warning: requires_gpu=True but no usable "
                    "nvidia-smi GPU probe — spec records cpu; GPU feasibility "
                    "is unverified.\n"
                )
        except OSError:
            pass
    return {"type": "cpu", "variant": variant}


def collect_environment_spec(task: ReproTask, repo_path: Path, *,
                             probe_log: Path | None = None) -> dict:
    """Build the ENVIRONMENT_SPEC_V1 dict for this task + machine + repo.

    probe_log (setup stderr) receives the feasibility warning when
    requires_gpu=True but the GPU probe fails — never a silent downgrade.
    """
    return {
        "schema": "ENVIRONMENT_SPEC_V1",
        # ONE python-version rule, from the vendored contract: explicit
        # task value > repository environment.yml pin > contract default.
        "python": _contract.select_python_version(task.python_version, Path(repo_path)),
        "os": _os_family(),
        "arch": _arch(),
        "accelerator": _accelerator_spec(task, Path(repo_path), probe_log=probe_log),
        "dependency_files": _contract.collect_dependency_files(Path(repo_path)),
        "channels": [],
        # The caller's mirror strategy NAME as-is; "" only when unset.
        "pip_index_profile": "" if task.mirror_profile in ("", "none") else task.mirror_profile,
        "framework_constraints": [],
        "notes": "",
    }
