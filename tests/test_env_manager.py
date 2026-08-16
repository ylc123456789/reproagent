"""Manifest state machine, atomic writes, creation locks, inventory, cleanup plan."""

import json
import os
import subprocess
from pathlib import Path

from reproagent.runtime import env_manager


def _manifest(root, env_id="resenv_demo_66630c82f611", **overrides):
    manifest = env_manager.new_manifest(
        env_id=env_id,
        prefix=str(root / "conda-envs" / env_id),
        spec={"schema": "ENVIRONMENT_SPEC_V1", "python": "3.11"},
        spec_fingerprint="f" * 64,
        created_by={"module": "reproagent", "run_id": "res-x", "task_id": "t1"},
        provenance={"repo_origin": "local", "repo_commit": "abc"},
    )
    manifest.update(overrides)
    return manifest


# ── manifest lifecycle ────────────────────────────────────────────

def test_write_and_read_manifest_roundtrip(tmp_path):
    manifest = _manifest(tmp_path)
    path = env_manager.write_manifest_atomic(tmp_path, manifest["env_id"], manifest)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()

    loaded = env_manager.read_manifest(tmp_path, manifest["env_id"])
    assert loaded["state"] == "creating"
    assert loaded["schema"] == "ENVIRONMENT_MANIFEST_V1"
    assert loaded["spec"] == manifest["spec"]
    assert loaded["updated_at"]


def test_mark_ready_only_from_creating(tmp_path):
    manifest = _manifest(tmp_path)
    env_manager.write_manifest_atomic(tmp_path, manifest["env_id"], manifest)

    ready = env_manager.mark_ready(tmp_path, manifest["env_id"], "a" * 64, {"python": "3.11.9"})
    assert ready["state"] == "ready"
    assert ready["resolved_fingerprint"] == "a" * 64

    import pytest
    with pytest.raises(ValueError, match="invalid manifest transition"):
        env_manager.mark_ready(tmp_path, manifest["env_id"], "b" * 64, {})


def test_mark_drifted_records_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    env_manager.write_manifest_atomic(tmp_path, manifest["env_id"], manifest)
    env_manager.mark_ready(tmp_path, manifest["env_id"], "a" * 64, {})

    drifted = env_manager.mark_drifted(
        tmp_path, manifest["env_id"], expected="a" * 64, actual="b" * 64, details="pip changed")
    assert drifted["state"] == "drifted"
    assert drifted["drift"]["expected"] == "a" * 64
    assert drifted["drift"]["actual"] == "b" * 64


def test_mark_failed_and_no_fake_ready(tmp_path):
    manifest = _manifest(tmp_path)
    env_manager.write_manifest_atomic(tmp_path, manifest["env_id"], manifest)
    failed = env_manager.mark_failed(tmp_path, manifest["env_id"])
    assert failed["state"] == "failed"

    import pytest
    with pytest.raises(ValueError, match="invalid manifest transition"):
        env_manager.mark_ready(tmp_path, manifest["env_id"], "c" * 64, {})


def test_list_manifests_tolerates_corrupt_entry(tmp_path):
    good = _manifest(tmp_path, env_id="resenv_good_66630c82f611")
    env_manager.write_manifest_atomic(tmp_path, good["env_id"], good)
    corrupt_dir = env_manager.environments_dir(tmp_path) / "resenv_bad_66630c82f611"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    manifests = env_manager.list_manifests(tmp_path)
    assert [m["env_id"] for m in manifests] == [good["env_id"]]


# ── creation lock ─────────────────────────────────────────────────

def test_lock_exclusive_and_reacquirable(tmp_path):
    lock = env_manager.acquire_creation_lock(tmp_path, "f" * 64)
    assert lock is not None and lock.exists()

    assert env_manager.acquire_creation_lock(tmp_path, "f" * 64) is None

    env_manager.heartbeat_lock(lock)
    info = env_manager.read_lock(tmp_path, "f" * 64)
    assert info["pid"] == os.getpid()
    assert info["heartbeat_at"]

    env_manager.release_creation_lock(lock)
    assert env_manager.acquire_creation_lock(tmp_path, "f" * 64) is not None


def test_recover_stale_lock_requires_dead_holder(tmp_path):
    # live holder (this test process) — never recovered
    lock = env_manager.acquire_creation_lock(tmp_path, "f" * 64)
    assert env_manager.recover_stale_lock(tmp_path, "f" * 64) is None
    assert lock.exists()
    env_manager.release_creation_lock(lock)

    # verifiably dead holder — recovered
    dead = subprocess.run(["true"]).returncode  # noqa
    import subprocess as sp
    proc = sp.Popen(["true"])
    proc.wait()
    dead_pid = proc.pid
    locks_dir = env_manager.locks_dir(tmp_path)
    (locks_dir / f"{'f' * 64}.lock").write_text(
        json.dumps({"host": "x", "pid": dead_pid, "started_at": "", "heartbeat_at": ""}),
        encoding="utf-8",
    )
    recovered = env_manager.recover_stale_lock(tmp_path, "f" * 64)
    assert recovered is not None
    assert not (locks_dir / f"{'f' * 64}.lock").exists()


# ── resolved inventory (fake conda) ───────────────────────────────

def _fake_conda(tmp_path) -> str:
    """Executable shell dispatcher pretending to be conda.

    `conda run --no-capture-output -p <prefix> bash -o pipefail -c <cmd>`
    is unwrapped by searching for -p; `conda list -p <prefix> --json` is
    matched directly.
    """
    fake = tmp_path / "fake-conda"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"run\" ]; then\n"
        "  while [ \"$#\" -gt 0 ] && [ \"$1\" != \"-p\" ]; do shift; done\n"
        "  shift 2\n"
        "  shift\n"
        "  shift 2\n"
        "  shift\n"
        "  cmd=\"$1\"\n"
        "  case \"$cmd\" in\n"
        "    \"python --version\") echo 'Python 3.11.9' ;;\n"
        "    \"python -m pip\"*) echo '[{\"name\": \"torch\", \"version\": \"2.6.0\"}]' ;;\n"
        "    \"python -c\"*) echo '{\"torch\": {\"version\": \"2.6.0\", \"cuda\": \"12.4\"}}' ;;\n"
        "  esac\n"
        "elif [ \"$1\" = \"list\" ] && [ \"$2\" = \"-p\" ]; then\n"
        "  echo '[]'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return str(fake)


def test_collect_resolved_inventory(tmp_path):
    conda = _fake_conda(tmp_path)

    resolved = env_manager.collect_resolved_inventory(conda, "/envs/demo")

    assert resolved["python"] == "3.11.9"
    assert len(resolved["pip_inventory_sha256"]) == 64
    assert len(resolved["conda_inventory_sha256"]) == 64
    assert resolved["frameworks"]["torch"]["cuda"] == "12.4"
    assert resolved["abi_summary"]  # glibc on this platform


def test_collect_resolved_inventory_defensive_on_missing_conda(tmp_path):
    resolved = env_manager.collect_resolved_inventory(
        str(tmp_path / "no-such-conda"), "/envs/demo", timeout=5)
    assert resolved["python"] is None
    assert resolved["pip_inventory_sha256"] is None


# ── cleanup plan (dry-run only) ───────────────────────────────────

def test_plan_cleanup_excludes_pinned_and_ready(tmp_path):
    drifted = _manifest(tmp_path, env_id="resenv_drift_66630c82f611", state="drifted")
    env_manager.write_manifest_atomic(tmp_path, drifted["env_id"], drifted)
    ready = _manifest(tmp_path, env_id="resenv_ready_66630c82f611", state="ready")
    env_manager.write_manifest_atomic(tmp_path, ready["env_id"], ready)
    pinned = _manifest(tmp_path, env_id="resenv_pin_66630c82f611", state="failed", pinned=True)
    env_manager.write_manifest_atomic(tmp_path, pinned["env_id"], pinned)

    plan = env_manager.plan_cleanup(tmp_path)

    assert plan["dry_run"] is True
    reasons = {c["env_id"]: c["reason"] for c in plan["candidates"]}
    assert reasons == {"resenv_drift_66630c82f611": "drifted"}


def test_plan_cleanup_stale_creating_needs_dead_lock(tmp_path):
    import subprocess as sp
    creating = _manifest(tmp_path, env_id="resenv_stale_66630c82f611", state="creating")
    env_manager.write_manifest_atomic(tmp_path, creating["env_id"], creating)
    proc = sp.Popen(["true"])
    proc.wait()
    env_manager.locks_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    lock = env_manager.locks_dir(tmp_path) / f"{creating['spec_fingerprint']}.lock"
    lock.write_text(json.dumps({"host": "x", "pid": proc.pid}), encoding="utf-8")

    plan = env_manager.plan_cleanup(tmp_path)
    assert [c["env_id"] for c in plan["candidates"]] == ["resenv_stale_66630c82f611"]

    # with a LIVE lock holder the stale creating is protected
    live = env_manager.locks_dir(tmp_path) / f"{creating['spec_fingerprint']}.lock"
    live.write_text(json.dumps({"host": "x", "pid": os.getpid()}), encoding="utf-8")
    assert env_manager.plan_cleanup(tmp_path)["candidates"] == []
