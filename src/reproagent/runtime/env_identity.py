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
    "requirements.txt", "pyproject.toml", "setup.py",
    "poetry.lock", "Pipfile.lock",
)
_REQUIREMENTS_PATTERN = re.compile(r"requirements[^/]*\.txt")
_MIRROR_TO_INDEX = {"none": "", "cn": "aliyun", "autodl": "aliyun"}


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


def _detect_accelerator(timeout: int = 10) -> dict:
    """Accelerator type + binary variant from nvidia-smi.

    The CUDA driver version decides the binary variant: 12.4 → cu124.
    No usable nvidia-smi → cpu.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"type": "cpu", "variant": ""}
    query = [exe, "--query-gpu=driver_version,cuda_version", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(query, text=True, capture_output=True, timeout=timeout)
    except Exception:
        return {"type": "cpu", "variant": ""}
    if result.returncode != 0 or not result.stdout.strip():
        return {"type": "cpu", "variant": ""}
    cuda = result.stdout.strip().splitlines()[0].split(",")[-1].strip()
    variant = "cu" + "".join(cuda.split(".")[:2]) if cuda else ""
    return {"type": "cuda", "variant": variant}


def _dependency_files(repo_path: Path) -> list[dict]:
    """Dependency declarations, sorted by repo-relative path, with content hash."""
    files: list[dict] = []
    for candidate in sorted(repo_path.rglob("*")):
        if not candidate.is_file():
            continue
        rel = candidate.relative_to(repo_path)
        name = rel.name
        if name in _DEPENDENCY_FILE_NAMES or _REQUIREMENTS_PATTERN.fullmatch(name):
            try:
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                continue
            files.append({"path": str(rel), "sha256": digest})
    return files


def collect_environment_spec(task: ReproTask, repo_path: Path) -> dict:
    """Build the ENVIRONMENT_SPEC_V1 dict for this task + machine + repo."""
    return {
        "schema": "ENVIRONMENT_SPEC_V1",
        "python": _normalize_python_version(task.python_version),
        "os": _os_family(),
        "arch": _arch(),
        "accelerator": _detect_accelerator(),
        "dependency_files": _dependency_files(Path(repo_path)),
        "channels": [],
        "pip_index_profile": _MIRROR_TO_INDEX.get(task.mirror_profile, ""),
        "framework_constraints": [],
        "notes": "",
    }
