"""ENVIRONMENT_*_V1 — the canonical contract algorithms.

This file is the ONLY implementation of the M2 environment-identity
algorithms. reproagent and CodingAgent vendor it byte-identically at
  reproagent: src/reproagent/_vendor/env_contract_v1.py
  CodingAgent: src/coding_agent/_vendor/env_contract_v1.py
and each repo carries a test asserting the vendored copy's sha256 equals
this file's. Text-described algorithms diverge; a shared file cannot.

Rules herein (contracts/README.md is the prose version):
- canonical JSON: sorted keys, ASCII-safe, no insignificant whitespace
- spec identity excludes operational metadata (notes, pip_index_profile)
- resolved inventory is computed AFTER dependency installation, inside
  the target env, with the exact probes below

Stdlib only. No imports beyond the standard library.
"""

from __future__ import annotations

import hashlib
import json
import re

CONTRACT_VERSION = "1.1.0"

# ── canonical serialization ──────────────────────────────────────────


def canonical_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── spec identity (ENVIRONMENT_SPEC_V1) ──────────────────────────────


def identity_subset(spec: dict) -> dict:
    """Identity-bearing subset of a spec. Operational fields (notes,
    pip_index_profile) never enter identity."""
    return {
        "python": spec["python"],
        "os": spec["os"],
        "arch": spec["arch"],
        "accelerator": {
            "type": spec["accelerator"]["type"],
            "variant": spec["accelerator"].get("variant", ""),
        },
        "dependency_files": [
            {k: f[k] for k in ("path", "sha256", "revision") if k in f}
            for f in sorted(spec["dependency_files"], key=lambda f: f["path"])
        ],
        "channels": sorted(spec.get("channels", [])),
        "framework_constraints": sorted(spec.get("framework_constraints", [])),
    }


def spec_fingerprint(spec: dict) -> str:
    return sha256_hex(canonical_dumps(identity_subset(spec)))


def project_slug(project: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (project or "").lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug) or "project"


def env_id(project: str, fingerprint: str) -> str:
    return f"resenv_{project_slug(project)}_{fingerprint[:12]}"


# ── resolved inventory (the exact probes and their normalization) ────

# Probes run INSIDE the target env. `python_bin` is <prefix>/bin/python;
# `conda_exe` is the system conda. The command LISTS are part of the
# contract — changing them changes the identity.
PYTHON_VERSION_ARGV = ("{python_bin}", "--version")
PIP_LIST_ARGV = ("{python_bin}", "-m", "pip", "list", "--format=json")
CONDA_LIST_ARGV = ("{conda_exe}", "list", "-p", "{prefix}", "--json")

_FRAMEWORKS = ("torch", "tensorflow", "jax")


def normalize_pip_inventory(pip_list_json: str) -> list[dict]:
    """pip list --format=json output → sorted [{name, version}] (name
    lowercased). Unparseable input yields []."""
    try:
        entries = json.loads(pip_list_json)
    except (json.JSONDecodeError, TypeError):
        return []
    normalized = [
        {"name": str(e.get("name", "")).lower(), "version": str(e.get("version", ""))}
        for e in entries if isinstance(e, dict) and e.get("name")
    ]
    return sorted(normalized, key=lambda e: e["name"])


def normalize_conda_inventory(conda_list_json: str) -> list[dict]:
    """conda list --json output → sorted [{name, version, channel}]."""
    try:
        entries = json.loads(conda_list_json)
    except (json.JSONDecodeError, TypeError):
        return []
    normalized = [
        {
            "name": str(e.get("name", "")).lower(),
            "version": str(e.get("version", "")),
            "channel": str(e.get("channel", "")),
        }
        for e in entries if isinstance(e, dict) and e.get("name")
    ]
    return sorted(normalized, key=lambda e: e["name"])


def frameworks_from_pip(pip_inventory: list[dict]) -> dict:
    """Framework facts parsed from the normalized pip inventory (no
    imports into the target env — probing by import is a side effect).
    The CUDA variant comes from the wheel's local version tag, normalized
    to torch's own convention (2.6.0+cu124 → cuda "12.4")."""
    frameworks: dict = {}
    for entry in pip_inventory:
        name = entry["name"]
        if name not in _FRAMEWORKS:
            continue
        version = entry["version"]
        info: dict = {"version": version.split("+")[0]}
        local_tag = version.split("+", 1)[1] if "+" in version else ""
        if name == "torch" and local_tag.startswith("cu"):
            digits = local_tag[2:]
            if digits.isdigit() and len(digits) >= 2:
                info["cuda"] = f"{digits[:-1]}.{digits[-1]}"
        frameworks[name] = info
    return frameworks


def build_resolved(*, python_version: str, pip_list_json: str,
                   conda_list_json: str, abi_summary: str) -> dict:
    """Assemble the normalized resolved object from raw probe outputs."""
    pip_inventory = normalize_pip_inventory(pip_list_json)
    conda_inventory = normalize_conda_inventory(conda_list_json)
    python = python_version.strip()
    if python.lower().startswith("python "):
        python = python.split()[1]
    return {
        "python": python,
        "conda_inventory_sha256": sha256_hex(canonical_dumps(conda_inventory)),
        "pip_inventory_sha256": sha256_hex(canonical_dumps(pip_inventory)),
        "frameworks": frameworks_from_pip(pip_inventory),
        "abi_summary": abi_summary,
    }


def resolved_fingerprint(resolved: dict) -> str:
    """Drift-detection identity over the five resolved fields."""
    normalized = {
        "python": resolved.get("python"),
        "conda_inventory_sha256": resolved.get("conda_inventory_sha256"),
        "pip_inventory_sha256": resolved.get("pip_inventory_sha256"),
        "frameworks": resolved.get("frameworks") or {},
        "abi_summary": resolved.get("abi_summary") or "",
    }
    return sha256_hex(canonical_dumps(normalized))


# ── spec collection (the enumeration/selection rules are identity too) ─────

# Dependency declarations joining identity, matched by NAME at any depth.
# This set is normative: adding or dropping a name changes fingerprints.
DEPENDENCY_FILE_NAMES = (
    "environment.yml", "environment.yaml", "conda.yml", "conda.yaml",
    "pyproject.toml", "setup.py", "setup.cfg",
)
_REQUIREMENTS_NAME = re.compile(r"requirements[^/]*\.(txt|in)")


def collect_dependency_files(repo_path) -> list[dict]:
    """Enumerate + hash dependency declarations, sorted by repo-relative
    POSIX path. Hash is over the RAW BYTES (never decoded text)."""
    from pathlib import Path
    root = Path(repo_path)
    files: list[dict] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        name = candidate.name
        if (name in DEPENDENCY_FILE_NAMES or _REQUIREMENTS_NAME.fullmatch(name)
                or name.endswith(".lock")):
            try:
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                continue
            files.append({"path": candidate.relative_to(root).as_posix(),
                          "sha256": digest})
    return sorted(files, key=lambda f: f["path"])


def select_python_version(task_python: str, repo_path) -> str:
    """Python identity: explicit task value > environment.yml pin > the
    contract default. NEVER the caller's own interpreter — host state is
    not identity."""
    explicit = _major_minor(task_python)
    if explicit:
        return explicit
    from pathlib import Path
    for name in ("environment.yml", "environment.yaml"):
        candidate = Path(repo_path) / name
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            match = re.search(r"\bpython\s*[=~<>]?\s*(\d+\.\d+)", text)
            if match:
                return match.group(1)
    return DEFAULT_PYTHON


DEFAULT_PYTHON = "3.10"
_CU_TAG = re.compile(r"\+cu(\d{2,3})\b")


def constraint_cuda_variant(repo_path) -> str:
    """Binary variant from explicit dependency constraints (+cu124 local
    tags in dependency files). Conflicting variants resolve to ""
    (ambiguous). Never inferred from drivers or the caller's installed
    frameworks."""
    from pathlib import Path
    variants: set[str] = set()
    for entry in collect_dependency_files(repo_path):
        try:
            text = (Path(repo_path) / entry["path"]).read_text(
                encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in _CU_TAG.finditer(text):
            variants.add(f"cu{match.group(1)}")
    return variants.pop() if len(variants) == 1 else ""


def _major_minor(version: str) -> str:
    match = re.match(r"\s*(\d+\.\d+)", version or "")
    return match.group(1) if match else ""


def probe_gpu_usable(timeout: int = 15) -> bool:
    """FEASIBILITY probe only — never identity. A GPU is usable when
    nvidia-smi exits 0 and prints its standard header. Header formats vary
    (driver CUDA vs WSL's "CUDA UMD Version"), so the check is the
    NVIDIA-SMI banner, not a specific version field."""
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        result = subprocess.run([exe], text=True, capture_output=True,
                                timeout=timeout)
    except Exception:
        return False
    return result.returncode == 0 and "NVIDIA-SMI" in (result.stdout or "")
