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

import hashlib
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

from ..models import ReproTask

_DEPENDENCY_FILE_NAMES = (
    "environment.yml", "environment.yaml", "conda.yml", "conda.yaml",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
)
_REQUIREMENTS_PATTERN = re.compile(r"requirements[^/]*\.txt")
_CU_VARIANT_RE = re.compile(r"\+cu(\d{2,3})\b")


# ── canonical serialization and fingerprints ──────────────────────

def canonical_dumps(obj) -> str:
    """Canonical JSON: sorted keys, ASCII-safe, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity_subset(spec: dict) -> dict:
    """The identity-bearing subset of ENVIRONMENT_SPEC_V1 (schema notes).

    Fields not marked `identity: true` in the frozen schema (notes,
    timestamps, absolute paths) never enter the fingerprint.
    """
    return {
        "python": spec["python"],
        "os": spec["os"],
        "arch": spec["arch"],
        "accelerator": {
            "type": spec["accelerator"]["type"],
            "variant": spec["accelerator"].get("variant", ""),
        },
        "dependency_files": [
            {key: entry[key] for key in ("path", "sha256", "revision") if key in entry}
            for entry in sorted(spec["dependency_files"], key=lambda entry: entry["path"])
        ],
        "channels": sorted(spec.get("channels", [])),
        "pip_index_profile": spec.get("pip_index_profile", ""),
        "framework_constraints": sorted(spec.get("framework_constraints", [])),
    }


def spec_fingerprint(spec: dict) -> str:
    """Requested-environment identity (creation/lookup key)."""
    return sha256_hex(canonical_dumps(identity_subset(spec)))


def resolved_fingerprint(resolved: dict) -> str:
    """Actual-inventory identity (drift detection).

    Hashed over the manifest `resolved` object fields, which map 1:1 to
    the milestone-2 identity inputs: python version, conda inventory,
    pip inventory, framework versions/binary variants, ABI summary.
    """
    normalized = {
        "python": resolved.get("python"),
        "conda_inventory_sha256": resolved.get("conda_inventory_sha256"),
        "pip_inventory_sha256": resolved.get("pip_inventory_sha256"),
        "frameworks": resolved.get("frameworks") or {},
        "abi_summary": resolved.get("abi_summary") or "",
    }
    return sha256_hex(canonical_dumps(normalized))


def slug_project(name: str) -> str:
    """Human-readable project slug per contracts/README.md rules."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug) or "project"


def env_id(project: str, fingerprint: str) -> str:
    """resenv_<project-slug>_<spec_fingerprint[:12]> — identity is the
    fingerprint; the slug is a human hint only."""
    return f"resenv_{slug_project(project)}_{fingerprint[:12]}"


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


def _normalize_python_version(version: str) -> str:
    match = re.match(r"(\d+\.\d+)", version or "")
    return match.group(1) if match else "3.10"


def _probe_nvidia_smi(timeout: int = 10) -> str | None:
    """Robust driver probe: the plain nvidia-smi header's `CUDA Version:
    X.Y` line (no --query-gpu field dependency).  Returns the CUDA version
    or None; never raises.

    FEASIBILITY-ONLY — the probed driver version NEVER enters identity.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run([exe], text=True, capture_output=True, timeout=timeout)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout or "")
    return match.group(1) if match else None


def _constraint_cuda_variant(repo_path: Path) -> str:
    """CUDA binary variant only from EXPLICIT dependency constraints
    (e.g. torch==2.6.*+cu124 → cu124).  Conflicting variants resolve to
    '' (ambiguous).  The driver's maximum supported CUDA is never mapped
    into a wheel variant (13.0 != cu130)."""
    variants: set[str] = set()
    for entry in _dependency_files(repo_path):
        try:
            text = (Path(repo_path) / entry["path"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _CU_VARIANT_RE.finditer(text):
            variants.add("cu" + match.group(1))
    return sorted(variants)[0] if len(variants) == 1 else ""


def _accelerator_spec(task, repo_path: Path, *, probe_log=None) -> dict:
    """Accelerator identity per the frozen collection semantics:

    - type: requires_gpu AND a usable local GPU probe → cuda, else cpu
    - variant: explicit task/dependency constraints only, else ""
    - a failed probe with requires_gpu=True is a FEASIBILITY warning
      written to the setup log — never a silent downgrade.
    """
    variant = _constraint_cuda_variant(repo_path)
    if not getattr(task, "requires_gpu", False):
        return {"type": "cpu", "variant": variant}
    if _probe_nvidia_smi() is not None:
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


def _dependency_files(repo_path: Path) -> list[dict]:
    """Dependency declarations, sorted by repo-relative path, with the
    SHA-256 of the RAW file bytes (never decode/transcode first)."""
    files: list[dict] = []
    for candidate in sorted(repo_path.rglob("*")):
        if not candidate.is_file():
            continue
        rel = candidate.relative_to(repo_path)
        name = rel.name
        if (name in _DEPENDENCY_FILE_NAMES
                or _REQUIREMENTS_PATTERN.fullmatch(name)
                or name.endswith(".lock")):
            try:
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                continue
            files.append({"path": str(rel), "sha256": digest})
    return files


def collect_environment_spec(task: ReproTask, repo_path: Path, *,
                             probe_log: Path | None = None) -> dict:
    """Build the ENVIRONMENT_SPEC_V1 dict for this task + machine + repo.

    probe_log (setup stderr) receives the feasibility warning when
    requires_gpu=True but the GPU probe fails — never a silent downgrade.
    """
    return {
        "schema": "ENVIRONMENT_SPEC_V1",
        "python": _normalize_python_version(task.python_version),
        "os": _os_family(),
        "arch": _arch(),
        "accelerator": _accelerator_spec(task, Path(repo_path), probe_log=probe_log),
        "dependency_files": _dependency_files(Path(repo_path)),
        "channels": [],
        # The caller's mirror strategy NAME as-is; "" only when unset.
        "pip_index_profile": "" if task.mirror_profile in ("", "none") else task.mirror_profile,
        "framework_constraints": [],
        "notes": "",
    }
