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
from .environment import build_backend_command

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


def recover_stale_lock(root: str | Path, fingerprint: str) -> dict | None:
    """Remove a creation lock whose holder is verifiably dead.

    A lock held by a live pid is never touched — recovery is liveness-
    based only, never age-based.
    """
    info = read_lock(root, fingerprint)
    if info is None:
        return None
    pid = int(info.get("pid", 0) or 0)
    if pid > 0 and _pid_alive(pid):
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

    Every probe is defensive: a failure yields None for that field, and
    callers treat an incomplete inventory as unverifiable.
    """
    resolved: dict = {
        "python": None,
        "conda_inventory_sha256": None,
        "pip_inventory_sha256": None,
        "frameworks": {},
        "abi_summary": "",
    }

    python_result = _run_probe(build_backend_command(prefix, "python --version", conda=conda), timeout)
    if python_result is not None and python_result.returncode == 0:
        text = (python_result.stdout or "").strip() or (python_result.stderr or "").strip()
        if text.startswith("Python "):
            resolved["python"] = text[len("Python "):].split()[0]

    pip_result = _run_probe(
        build_backend_command(prefix, "python -m pip list --format=json", conda=conda), timeout)
    if pip_result is not None and pip_result.returncode == 0:
        try:
            resolved["pip_inventory_sha256"] = sha256_hex(canonical_dumps(json.loads(pip_result.stdout)))
        except json.JSONDecodeError:
            pass

    conda_result = _run_probe([conda, "list", "-p", prefix, "--json"], timeout)
    if conda_result is not None and conda_result.returncode == 0:
        try:
            resolved["conda_inventory_sha256"] = sha256_hex(canonical_dumps(json.loads(conda_result.stdout)))
        except json.JSONDecodeError:
            pass

    probe_code = (
        "import importlib, json; out = {}\n"
        "for name in ('torch', 'tensorflow', 'jax'):\n"
        "    try:\n"
        "        mod = importlib.import_module(name)\n"
        "        info = {'version': str(getattr(mod, '__version__', ''))}\n"
        "        if name == 'torch' and getattr(getattr(mod, 'version', None), 'cuda', None):\n"
        "            info['cuda'] = str(mod.version.cuda)\n"
        "        out[name] = info\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps(out))\n"
    )
    framework_result = _run_probe(
        build_backend_command(prefix, "python -c " + shlex.quote(probe_code), conda=conda), timeout)
    if framework_result is not None and framework_result.returncode == 0:
        try:
            resolved["frameworks"] = json.loads(framework_result.stdout)
        except json.JSONDecodeError:
            pass

    try:
        libc_name, libc_version = platform.libc_ver()
        resolved["abi_summary"] = f"{libc_name}{libc_version}" if libc_name else ""
    except Exception:
        pass

    return resolved


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
            if lock is None or not _pid_alive(int(lock.get("pid", 0) or 0)):
                candidates.append({
                    "env_id": env_id,
                    "reason": "stale_creating",
                    "prefix": manifest.get("prefix", ""),
                    "last_used_at": manifest.get("last_used_at"),
                })
    return {"dry_run": True, "candidates": candidates}
