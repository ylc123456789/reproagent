"""Content-addressed environment manager — resource root, manifests, locks.

Implements the ENVIRONMENT_MANIFEST_V1 lifecycle: creating → ready|failed,
ready → drifted.  All decisions are deterministic code — the LLM never
participates in fingerprints, manifest state, locks, or cleanup candidates.

Layout under <resource_root>:
  environments/<env_id>/manifest.json (+ audits/, usage/)
  locks/<spec_fingerprint>.lock
  conda-envs/<env_id>/
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .env_identity import canonical_dumps, sha256_hex
from .environment import build_backend_command, task_run_id

MANIFEST_SCHEMA = "ENVIRONMENT_MANIFEST_V1"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── layout ────────────────────────────────────────────────────────

def environments_dir(root: str | Path) -> Path:
    return Path(root) / "environments"


def locks_dir(root: str | Path) -> Path:
    return Path(root) / "locks"


def conda_envs_dir(root: str | Path) -> Path:
    return Path(root) / "conda-envs"


def manifest_path(root: str | Path, env_id: str) -> Path:
    return environments_dir(root) / env_id / "manifest.json"


def audits_dir(root: str | Path, env_id: str) -> Path:
    return environments_dir(root) / env_id / "audits"


# ── manifest lifecycle ────────────────────────────────────────────

def new_manifest(*, env_id: str, prefix: str, spec: dict, spec_fingerprint: str,
                 created_by: dict, provenance: dict) -> dict:
    """Skeleton in `creating` state; physical creation is the caller's job."""
    return {
        "schema": MANIFEST_SCHEMA,
        "env_id": env_id,
        "state": "creating",
        "certification": "none",
        "spec_fingerprint": spec_fingerprint,
        "resolved_fingerprint": None,
        "prefix": prefix,
        "manager": "reproagent",
        "created_by": created_by,
        "created_at": utcnow(),
        "updated_at": utcnow(),
        "last_used_at": None,
        "pinned": False,
        "provenance": provenance,
        "spec": spec,
        "resolved": None,
        "audits": [],
        "usage": [],
    }


def write_manifest_atomic(root: str | Path, env_id: str, manifest: dict) -> Path:
    """Temp file + os.replace: readers never observe a half-written manifest."""
    env_dir = environments_dir(root) / env_id
    env_dir.mkdir(parents=True, exist_ok=True)
    path = env_dir / "manifest.json"
    manifest["updated_at"] = utcnow()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_manifest(root: str | Path, env_id: str) -> dict | None:
    path = manifest_path(root, env_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_manifests(root: str | Path) -> list[dict]:
    """All readable manifests; corrupt entries are tolerated and skipped."""
    env_dir = environments_dir(root)
    if not env_dir.is_dir():
        return []
    manifests: list[dict] = []
    for path in sorted(env_dir.glob("*/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("schema") == MANIFEST_SCHEMA:
            manifests.append(data)
    return manifests


def _transition(root, env_id, from_state: str, mutate) -> dict:
    """Read-modify-write under the state machine; refuses invalid transitions."""
    manifest = read_manifest(root, env_id)
    if manifest is None:
        raise ValueError(f"no manifest for {env_id}")
    if manifest.get("state") != from_state:
        raise ValueError(
            f"invalid manifest transition for {env_id}: "
            f"{manifest.get('state')} -> (required {from_state})"
        )
    mutate(manifest)
    write_manifest_atomic(root, env_id, manifest)
    return manifest


def mark_ready(root, env_id: str, resolved_fingerprint: str, resolved: dict) -> dict:
    """creating → ready; `ready` means physical creation completed. It says
    nothing about certification — that is a separate field."""
    return _transition(root, env_id, "creating", lambda m: (
        m.update(state="ready", resolved_fingerprint=resolved_fingerprint, resolved=resolved)
    ))


def mark_failed(root, env_id: str) -> dict:
    """creating → failed (creation error or dead creator)."""
    return _transition(root, env_id, "creating", lambda m: m.update(state="failed"))


def mark_drifted(root, env_id: str, expected: str, actual: str, details: str = "") -> dict:
    """ready → drifted; records the mismatch for diagnostics."""
    return _transition(root, env_id, "ready", lambda m: (
        m.update(state="drifted"),
        m.setdefault("drift", {}).update(expected=expected, actual=actual, details=details),
    ))


# ── creation lock (host/pid/heartbeat) ────────────────────────────

def _lock_path(root: str | Path, fingerprint: str) -> Path:
    return locks_dir(root) / f"{fingerprint}.lock"


def acquire_creation_lock(root: str | Path, fingerprint: str) -> Path | None:
    """Atomic exclusive creation lock via O_CREAT|O_EXCL.

    Returns the lock file path, or None when another creator holds it.
    The lock records host/pid/heartbeat so stale locks are recoverable
    by process liveness — never by age alone.
    """
    locks_dir(root).mkdir(parents=True, exist_ok=True)
    path = _lock_path(root, fingerprint)
    info = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": utcnow(),
        "heartbeat_at": utcnow(),
    }
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(info, handle)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def read_lock(root: str | Path, fingerprint: str) -> dict | None:
    path = _lock_path(root, fingerprint)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def heartbeat_lock(lock_path: str | Path) -> None:
    path = Path(lock_path)
    info = read_lock(path.parent, path.stem)
    if info is None:
        return
    info["heartbeat_at"] = utcnow()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(info), encoding="utf-8")
    os.replace(tmp, path)


def release_creation_lock(lock_path: str | Path) -> None:
    Path(lock_path).unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def holder_alive(info: dict) -> bool:
    """Conservative liveness for lock/lease holders.

    A host different from ours is always treated as alive — shared
    resource roots make remote processes unprobeable.  Local holders are
    checked with kill(pid, 0).
    """
    host = str(info.get("host", "") or "")
    if host and host != socket.gethostname():
        return True
    pid = int(info.get("pid", 0) or 0)
    return pid > 0 and _pid_alive(pid)


def recover_stale_lock(root: str | Path, fingerprint: str) -> dict | None:
    """Remove a creation lock whose local holder is verifiably dead.

    Locks held by a live pid or by a different host are never touched —
    recovery is liveness-based only, never age-based.
    """
    info = read_lock(root, fingerprint)
    if info is None:
        return None
    if holder_alive(info):
        return None
    _lock_path(root, fingerprint).unlink(missing_ok=True)
    return info


# ── resolved inventory ────────────────────────────────────────────

def _run_probe(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_resolved_inventory(conda: str, prefix: str, timeout: int = 120) -> dict:
    """Normalized installed-inventory summary (manifest `resolved` object).

    Probes run inside the target env; normalization and hashing come from
    the vendored contract file (the ONE implementation across repos).
    A failed probe yields an empty inventory for that channel.
    """
    from .._vendor import env_contract_v1 as _contract

    python_result = _run_probe(build_backend_command(prefix, "python --version", conda=conda), timeout)
    python_text = ""
    if python_result is not None and python_result.returncode == 0:
        python_text = (python_result.stdout or "").strip() or (python_result.stderr or "").strip()

    pip_result = _run_probe(
        build_backend_command(prefix, "python -m pip list --format=json", conda=conda), timeout)
    pip_text = pip_result.stdout if pip_result is not None and pip_result.returncode == 0 else ""

    conda_result = _run_probe([conda, "list", "-p", prefix, "--json"], timeout)
    conda_text = conda_result.stdout if conda_result is not None and conda_result.returncode == 0 else ""

    try:
        libc_name, libc_version = platform.libc_ver()
        abi = f"{libc_name}{libc_version}" if libc_name else ""
    except Exception:
        abi = ""

    resolved = _contract.build_resolved(
        python_version=python_text,
        pip_list_json=pip_text,
        conda_list_json=conda_text,
        abi_summary=abi,
    )
    if not python_text:
        resolved["python"] = None  # callers treat incomplete as unverifiable
    if not pip_text:
        resolved["pip_inventory_sha256"] = None
    if not conda_text:
        resolved["conda_inventory_sha256"] = None
    return resolved


# ── usage and audit records ───────────────────────────────────────

def audit_artifact_v1(*, env_id: str, level: str, outcome: str,
                      resolved_fingerprint: str, audited_by: dict,
                      checks: list[dict], notes: str = "") -> dict:
    """ENVIRONMENT_AUDIT_V1 record (experiment level only from reproagent)."""
    stamp = utcnow().replace("-", "").replace(":", "")
    import uuid
    return {
        "schema": "ENVIRONMENT_AUDIT_V1",
        "audit_id": f"audit_{env_id}_{stamp}_{uuid.uuid4().hex[:6]}",
        "env_id": env_id,
        "level": level,
        "outcome": outcome,
        "resolved_fingerprint": resolved_fingerprint,
        "audited_by": audited_by,
        "at": utcnow(),
        "checks": checks,
        "notes": notes,
    }


def record_audit(root: str | Path, env_id: str, *, artifact: dict, resolved: dict,
                 resolved_fingerprint: str, certification: str | None = None,
                 usage: dict | None = None) -> dict:
    """Persist the audit artifact JSON and update the manifest: audits entry,
    inventory, optional certification upgrade, and usage/last_used_at."""
    manifest = read_manifest(root, env_id)
    if manifest is None:
        raise ValueError(f"no manifest for {env_id}")
    audits_dir(root, env_id).mkdir(parents=True, exist_ok=True)
    rel = f"audits/{Path(artifact['audit_id']).name}.json"
    artifact_path = environments_dir(root) / env_id / rel
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8")

    manifest["audits"].append({
        "artifact": rel,
        "level": artifact["level"],
        "outcome": artifact["outcome"],
        "at": artifact["at"],
    })
    if certification:
        manifest["certification"] = certification
    manifest["resolved"] = resolved
    manifest["resolved_fingerprint"] = resolved_fingerprint
    if usage:
        manifest["usage"].append(usage)
        manifest["last_used_at"] = usage["at"]
    write_manifest_atomic(root, env_id, manifest)
    return manifest


def record_usage(root: str | Path, env_id: str, *, run_id: str = "", task_id: str = "") -> dict:
    """Append a usage entry and refresh last_used_at."""
    manifest = read_manifest(root, env_id)
    if manifest is None:
        raise ValueError(f"no manifest for {env_id}")
    entry = {"run_id": run_id, "task_id": task_id, "at": utcnow()}
    manifest["usage"].append(entry)
    manifest["last_used_at"] = entry["at"]
    write_manifest_atomic(root, env_id, manifest)
    return manifest



def _operator_audit_checks(audit) -> list[dict]:
    """Map the reproagent environment audit onto ENVIRONMENT_AUDIT_V1 checks."""
    checks: list[dict] = [{
        "name": "policy",
        "outcome": "pass",
        "detail": audit.summary,
        "evidence_path": str(audit.stdout_path)
        if audit.stdout_path is not None and audit.stdout_path.exists() else "",
    }]
    details = audit.details or []
    if any("torch" in detail.lower() for detail in details):
        checks.append({"name": "framework_import", "outcome": "pass",
                       "detail": next(d for d in details if "torch" in d.lower())})
    if any("gpu" in detail.lower() or "cuda" in detail.lower() for detail in details):
        checks.append({"name": "accelerator", "outcome": "pass",
                       "detail": "CUDA-capable environment per audit"})
    return checks


def finalize_manifest_after_audit(state, audit) -> dict | None:
    """After a successful operator audit in content-addressed mode: recompute
    the inventory, update resolved_fingerprint, upgrade certification to
    experiment (§6.3 — reproagent is the only experiment certifier), persist
    the ENVIRONMENT_AUDIT_V1 artifact, and append usage.

    Returns the updated manifest, or None when the mode does not apply.
    """
    from .env_identity import (
        collect_environment_spec,
        env_id,
        project_slug_for,
        resolved_fingerprint,
        spec_fingerprint,
    )
    from .environment import find_conda

    task = state.task
    if task.reuse_mode != "content_addressed" or not task.resource_root:
        return None
    repo_path = state.repo_context.repo_path if state.repo_context else task.workspace_dir / "repo"
    root = Path(task.resource_root).expanduser()
    spec = collect_environment_spec(task, repo_path)
    identifier = env_id(project_slug_for(task), spec_fingerprint(spec))
    manifest = read_manifest(root, identifier)
    if manifest is None:
        return None  # unmanaged environment — nothing to finalize
    conda = find_conda() or "conda"
    inventory = collect_resolved_inventory(conda, manifest.get("prefix", ""))
    fingerprint = resolved_fingerprint(inventory)
    artifact = audit_artifact_v1(
        env_id=identifier, level="experiment", outcome="pass",
        resolved_fingerprint=fingerprint,
        audited_by={"module": "reproagent", "run_id": task_run_id(task), "task_id": task.task_id},
        checks=_operator_audit_checks(audit),
        notes=audit.summary,
    )
    return record_audit(
        root, identifier, artifact=artifact, resolved=inventory,
        resolved_fingerprint=fingerprint, certification="experiment",
        usage={"run_id": task_run_id(task), "task_id": task.task_id, "at": utcnow()},
    )


# ── inventory invalidation and spec compliance ────────────────────

def invalidate_inventory(root: str | Path, env_id: str) -> dict | None:
    """Package changes make the recorded inventory unverifiable.

    Resets the ready manifest's resolved_fingerprint (and experiment
    certification) until the next passing audit finalizes the new
    inventory.  The in-memory certification gate is the hard stop; this
    keeps the manifest from pretending to verify a mutated env.
    """
    manifest = read_manifest(root, env_id)
    if manifest is None or manifest.get("state") != "ready":
        return manifest
    manifest["resolved_fingerprint"] = None
    if manifest.get("certification") == "experiment":
        manifest["certification"] = "none"
    write_manifest_atomic(root, env_id, manifest)
    return manifest


def invalidate_task_inventory(state) -> None:
    """Content-addressed bookkeeping after a package mutation (best-effort)."""
    task = state.task
    if task.reuse_mode != "content_addressed" or not task.resource_root:
        return
    try:
        from .env_identity import (
            collect_environment_spec,
            env_id,
            project_slug_for,
            spec_fingerprint,
        )

        repo_path = state.repo_context.repo_path if state.repo_context else task.workspace_dir / "repo"
        spec = collect_environment_spec(task, repo_path)
        identifier = env_id(project_slug_for(task), spec_fingerprint(spec))
        invalidate_inventory(task.resource_root, identifier)
    except Exception:
        pass  # bookkeeping only; the certification gate is the hard stop


def parse_requirements(repo_path: Path) -> list[dict]:
    """Parsed requirements*.txt entries: {name, constraint}.

    Unparseable lines (editable, URL, option lines) are skipped; the
    comparison is conservative — unknown operators degrade to presence
    checks in check_spec_compliance.
    """
    entries: list[dict] = []
    for req_file in sorted(Path(repo_path).glob("requirements*.txt")):
        try:
            lines = req_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "--", "git+", "http", "https")):
                continue
            name = line.split("=")[0].split("<")[0].split(">")[0].split("!")[0].split("[")[0].strip()
            if not name:
                continue
            constraint = line[len(name):].strip() if len(name) < len(line) else ""
            entries.append({"name": name.lower(), "constraint": constraint})
    return entries


def check_spec_compliance(conda: str, prefix: str, entries: list[dict],
                          timeout: int = 120) -> list[str]:
    """Verify requirements-declared distributions exist in the environment.

    Returns violation strings; empty list means compliant.  This is the
    second drift-detection line: even with a matching fingerprint, a
    spec-declared package missing from the env fails the audit.
    """
    if not entries:
        return []
    result = _run_probe(
        build_backend_command(prefix, "python -m pip list --format=json", conda=conda), timeout)
    if result is None or result.returncode != 0:
        return ["pip inventory unavailable — cannot verify spec compliance"]
    try:
        installed = {item["name"].lower(): str(item["version"]) for item in json.loads(result.stdout)}
    except (json.JSONDecodeError, KeyError, TypeError):
        return ["pip inventory unparseable — cannot verify spec compliance"]
    violations: list[str] = []
    for entry in entries:
        version = installed.get(entry["name"])
        if version is None:
            violations.append(f"missing distribution: {entry['name']}{entry['constraint']}")
            continue
        if entry["constraint"].startswith("==") and version != entry["constraint"][2:].strip():
            violations.append(
                f"{entry['name']}=={version} does not satisfy {entry['constraint']}")
    return violations


def check_task_spec_compliance(state) -> list[str]:
    """Spec-compliance probe for the task's managed environment."""
    task = state.task
    if task.reuse_mode != "content_addressed" or not task.resource_root:
        return []
    try:
        from .env_identity import (
            collect_environment_spec,
            env_id,
            project_slug_for,
            spec_fingerprint,
        )
        from .environment import find_conda

        repo_path = state.repo_context.repo_path if state.repo_context else task.workspace_dir / "repo"
        spec = collect_environment_spec(task, repo_path)
        identifier = env_id(project_slug_for(task), spec_fingerprint(spec))
        manifest = read_manifest(task.resource_root, identifier)
        if manifest is None or manifest.get("state") != "ready":
            return []
        return check_spec_compliance(
            find_conda() or "conda", manifest.get("prefix", ""), parse_requirements(repo_path))
    except Exception as exc:
        return [f"spec compliance probe failed: {exc}"]


# ── cleanup plan (dry-run ONLY — apply belongs to M2-P4) ──────────

def plan_cleanup(root: str | Path) -> dict:
    """Deterministic dry-run candidates. Never deletes anything."""
    candidates: list[dict] = []
    for manifest in list_manifests(root):
        env_id = manifest.get("env_id", "")
        state = manifest.get("state")
        if manifest.get("pinned"):
            continue
        if state in ("drifted", "failed"):
            candidates.append({
                "env_id": env_id,
                "reason": state,
                "prefix": manifest.get("prefix", ""),
                "last_used_at": manifest.get("last_used_at"),
            })
        elif state == "creating":
            # stale creating is only a candidate when no live creator holds the lock
            fp = manifest.get("spec_fingerprint", "")
            lock = read_lock(root, fp) if fp else None
            if lock is None or not holder_alive(lock):
                candidates.append({
                    "env_id": env_id,
                    "reason": "stale_creating",
                    "prefix": manifest.get("prefix", ""),
                    "last_used_at": manifest.get("last_used_at"),
                })
    return {"dry_run": True, "candidates": candidates}


def delete_environment(root: str | Path, env_id: str) -> dict:
    """Physically delete this module's env (prefix + manifest dir).

    M2-P4 apply path. The caller (ResAgent cleanup) owns policy; this
    function owns only identity and containment guards. Never raises.
    """
    import shutil

    manifest = read_manifest(root, env_id)
    if manifest is None:
        return {"env_id": env_id, "deleted": False, "reason": "manifest_missing"}
    if manifest.get("manager") != "reproagent":
        return {"env_id": env_id, "deleted": False,
                "reason": f"not_managed_by_reproagent:{manifest.get('manager', '')}"}

    prefix_text = str(manifest.get("prefix", "") or "")
    envs_root = conda_envs_dir(root).resolve()
    if prefix_text:
        resolved = Path(prefix_text).resolve()
        if resolved != envs_root and envs_root not in resolved.parents:
            # 越界保护: never delete anything outside the managed envs dir.
            return {"env_id": env_id, "deleted": False,
                    "reason": "prefix_outside_resource_root"}

    try:
        if prefix_text and Path(prefix_text).is_dir():
            shutil.rmtree(prefix_text)
        env_dir = environments_dir(root) / env_id
        if env_dir.is_dir():
            shutil.rmtree(env_dir)
    except OSError as exc:
        return {"env_id": env_id, "deleted": False, "reason": f"os_error:{exc}"}
    return {"env_id": env_id, "deleted": True, "reason": ""}
