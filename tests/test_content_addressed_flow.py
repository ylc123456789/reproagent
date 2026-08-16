"""Content-addressed ensure_environment: zero-creation reuse, concurrency,
drift refusal, and legacy regression."""

import subprocess
import threading
from pathlib import Path

import pytest

from reproagent.models import EnvironmentInfo, RepoContext, ReproState, ReproTask
from reproagent.runtime import env_manager
from reproagent.runtime import environment as env_module


def _fake_conda_script(tmp_path: Path, inventory_file: Path, create_log: Path,
                       create_sleep: float = 0.0) -> str:
    """Executable Python dispatcher pretending to be conda.

    `create -p <prefix>` logs into create_log (and can sleep to simulate a
    slow creator); `run` probes answer from the mutable inventory file, so
    tests can simulate drift by rewriting it.
    """
    fake = tmp_path / "fake-conda"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time, pathlib\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['list', '-p']:\n"
        "    print('[]')\n"
        "elif args[0] == 'run':\n"
        "    i = args.index('-p')\n"
        "    cmd = args[-1]\n"
        "    if cmd == 'python --version':\n"
        "        print('Python 3.11.9')\n"
        "    elif cmd.startswith('python -m pip'):\n"
        "        inv = pathlib.Path(%r)\n"
        "        print(inv.read_text() if inv.exists() else '[]')\n"
        "    elif cmd.startswith('python -c'):\n"
        "        print('{}')\n"
        "elif args[0] == 'create' and args[1] == '-p':\n"
        f"    time.sleep({create_sleep!r})\n"
        "    pathlib.Path(args[2]).mkdir(parents=True, exist_ok=True)\n"
        "    with open(%r, 'a') as f:\n"
        "        f.write(args[2] + '\\n')\n"
        "else:\n"
        "    sys.exit(1)\n" % (str(inventory_file), str(create_log)),
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return str(fake)


def _make_state(tmp_path: Path, root: Path, repo_url: str) -> ReproState:
    task = ReproTask(
        workspace_dir=tmp_path / "ws",
        reuse_mode="content_addressed",
        resource_root=str(root),
        python_version="3.10",
        repo_url=repo_url,
    )
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "requirements.txt").write_text("torch==2.6.0\n", encoding="utf-8")
    return ReproState(task=task, repo_context=RepoContext(repo_path=repo))


def _install_fake(tmp_path, monkeypatch, create_sleep: float = 0.0):
    inventory = tmp_path / "pip-inventory.json"
    inventory.write_text('[{"name": "torch", "version": "2.6.0"}]', encoding="utf-8")
    create_log = tmp_path / "creates.log"
    fake = _fake_conda_script(tmp_path, inventory, create_log, create_sleep=create_sleep)
    monkeypatch.setattr(env_module, "find_conda", lambda: fake)
    monkeypatch.setattr("reproagent.runtime.audit.find_conda", lambda: fake)
    return inventory, create_log


def _create_count(create_log: Path) -> int:
    return len(create_log.read_text(encoding="utf-8").splitlines()) if create_log.exists() else 0


# ── reuse / drift / config errors ─────────────────────────────────

def test_same_spec_second_ensure_zero_creation(tmp_path, monkeypatch):
    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)

    first = env_module.ensure_environment(_make_state(tmp_path, root, "https://github.com/org/demo.git"))
    assert first.created is True
    assert _create_count(create_log) == 1

    second = env_module.ensure_environment(_make_state(tmp_path, root, "https://github.com/org/demo.git"))
    assert second.created is False
    assert second.env_name == first.env_name
    assert _create_count(create_log) == 1  # zero creation

    manifests = env_manager.list_manifests(root)
    assert len(manifests) == 1
    assert manifests[0]["state"] == "ready"
    assert manifests[0]["certification"] == "experiment"  # re-audited on reuse
    assert len(manifests[0]["usage"]) == 2  # creator + reuser


def test_drift_refuses_reuse_and_marks_manifest(tmp_path, monkeypatch):
    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)
    env_module.ensure_environment(_make_state(tmp_path, root, "https://github.com/org/demo.git"))

    # manual pip install/uninstall → inventory changes → drift
    inventory.write_text('[{"name": "torch", "version": "2.7.0"}]', encoding="utf-8")

    with pytest.raises(RuntimeError, match="drift detected"):
        env_module.ensure_environment(_make_state(tmp_path, root, "https://github.com/org/demo.git"))

    manifest = env_manager.list_manifests(root)[0]
    assert manifest["state"] == "drifted"
    assert manifest["drift"]["expected"] != manifest["drift"]["actual"]


def test_content_addressed_requires_resource_root(tmp_path):
    task = ReproTask(workspace_dir=tmp_path / "ws", reuse_mode="content_addressed")
    state = ReproState(task=task, repo_context=RepoContext(repo_path=tmp_path / "repo"))
    with pytest.raises(RuntimeError, match="requires resource_root"):
        env_module.ensure_environment(state)


def test_drifted_manifest_refuses_reuse(tmp_path, monkeypatch):
    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)
    env_module.ensure_environment(_make_state(tmp_path, root, "https://github.com/org/demo.git"))

    manifest = env_manager.list_manifests(root)[0]
    env_manager.mark_drifted(root, manifest["env_id"], expected="a" * 64, actual="b" * 64)

    with pytest.raises(RuntimeError, match="is drifted"):
        env_module.ensure_environment(_make_state(tmp_path, root, "https://github.com/org/demo.git"))


# ── concurrency ───────────────────────────────────────────────────

def test_concurrent_same_spec_creates_once(tmp_path, monkeypatch):
    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch, create_sleep=1.0)
    results: dict[str, object] = {}
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        state = _make_state(tmp_path, root, "https://github.com/org/demo.git")
        barrier.wait()
        try:
            results[name] = env_module.ensure_environment(state)
        except Exception as exc:  # noqa: BLE001 — surfaced via assertions
            results[name] = exc

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("w0", "w1")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    infos = [r for r in results.values() if isinstance(r, EnvironmentInfo)]
    assert len(infos) == 2, results
    assert sum(1 for info in infos if info.created) == 1  # exactly one creator
    assert all(info.env_name == infos[0].env_name for info in infos)
    assert _create_count(create_log) == 1


# ── audit-time manifest finalization ──────────────────────────────

def test_audit_tool_finalizes_manifest(tmp_path, monkeypatch):
    """A passing audit_env in content-addressed mode upgrades the manifest:
    experiment certification, resolved fingerprint, audit artifact, usage."""
    from reproagent.controller.actions import _tool_audit_env
    from reproagent.models import AgentState

    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)
    state = _make_state(tmp_path, root, "https://github.com/org/demo.git")
    info = env_module.ensure_environment(state)

    agent_state = AgentState(
        task=state.task,
        repo_context=state.repo_context,
        environment=EnvironmentInfo(env_name=info.env_name),
    )
    observation = _tool_audit_env(agent_state)

    assert observation.audit is not None and observation.audit.success
    manifest = env_manager.list_manifests(root)[0]
    assert manifest["certification"] == "experiment"
    assert manifest["audits"], "audit entry expected"
    artifact_rel = manifest["audits"][0]["artifact"]
    assert (env_manager.environments_dir(root) / manifest["env_id"] / artifact_rel).exists()
    assert manifest["resolved_fingerprint"]


# ── env_name candidate semantics (M2-P1 fixup) ────────────────────

def test_spec_change_with_stale_env_name_candidate_creates_new(tmp_path, monkeypatch):
    """M2-P5 stage-3 shape: ResAgent injects the OLD env via env_name while
    the spec has changed. The fingerprint is the identity authority — the
    stale candidate must never be bound; a NEW env is created."""
    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)

    state_a = _make_state(tmp_path, root, "https://github.com/org/demo.git")
    info_a = env_module.ensure_environment(state_a)
    assert info_a.created is True
    assert _create_count(create_log) == 1

    state_b = _make_state(tmp_path, root, "https://github.com/org/demo.git")
    (tmp_path / "repo" / "requirements.txt").write_text("torch==2.9.0\n", encoding="utf-8")
    state_b.task.env_name = info_a.env_name  # stale injected candidate

    info_b = env_module.ensure_environment(state_b)

    assert info_b.created is True
    assert info_b.env_name != info_a.env_name
    assert _create_count(create_log) == 2
    assert len(env_manager.list_manifests(root)) == 2


def test_matching_env_name_candidate_reuses(tmp_path, monkeypatch):
    """An env_name candidate that equals the fingerprint identity goes
    through the normal verified reuse — zero creation."""
    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)

    state_a = _make_state(tmp_path, root, "https://github.com/org/demo.git")
    info_a = env_module.ensure_environment(state_a)

    state_b = _make_state(tmp_path, root, "https://github.com/org/demo.git")
    state_b.task.env_name = info_a.env_name  # matching candidate

    info_b = env_module.ensure_environment(state_b)

    assert info_b.created is False
    assert info_b.env_name == info_a.env_name
    assert _create_count(create_log) == 1


def test_legacy_explicit_env_name_binding_unchanged(tmp_path, monkeypatch):
    """Legacy mode keeps the direct env_name binding; resource_root is inert."""
    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["/fake/conda", "env", "list"]:
            return subprocess.CompletedProcess(cmd, 0, '{"envs": ["/opt/conda/envs/target_env"]}', "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(env_module, "find_conda", lambda: "/fake/conda")
    monkeypatch.setattr(env_module.subprocess, "run", fake_run)

    task = ReproTask(
        workspace_dir=tmp_path / "ws",
        env_name="target_env",
        resource_root=str(tmp_path / "resources"),
    )
    state = ReproState(task=task, repo_context=RepoContext(repo_path=tmp_path / "repo"))
    info = env_module.ensure_environment(state)

    assert info.env_name == "target_env"
    assert info.created is False
    assert not (tmp_path / "resources").exists()


# ── resolved fingerprint covers post-install inventory (fixup #2) ─

def _agent_state(state, env_name):
    from reproagent.models import AgentState

    return AgentState(
        task=state.task,
        repo_context=state.repo_context,
        environment=EnvironmentInfo(env_name=env_name),
    )


def test_finalize_after_install_reflects_new_packages(tmp_path, monkeypatch):
    """After numpy+six are installed and audited, the manifest's resolved
    fingerprint must differ from the numpy-only fingerprint — the post-
    install inventory is the authoritative one."""
    from reproagent.controller.actions import _tool_audit_env

    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)
    inventory.write_text('[{"name": "numpy", "version": "1.26.4"}]', encoding="utf-8")
    state = _make_state(tmp_path, root, "https://github.com/org/demo.git")
    (tmp_path / "repo" / "requirements.txt").write_text("numpy\n", encoding="utf-8")

    info = env_module.ensure_environment(state)
    agent = _agent_state(state, info.env_name)
    observation = _tool_audit_env(agent)
    assert observation.audit is not None and observation.audit.success
    fingerprint_numpy = env_manager.list_manifests(root)[0]["resolved_fingerprint"]

    # "pip install six": the inventory file is what the fake conda reports
    inventory.write_text(
        '[{"name": "numpy", "version": "1.26.4"}, {"name": "six", "version": "1.16.0"}]',
        encoding="utf-8",
    )
    observation = _tool_audit_env(agent)

    assert observation.audit is not None and observation.audit.success
    fingerprint_both = env_manager.list_manifests(root)[0]["resolved_fingerprint"]
    assert fingerprint_both != fingerprint_numpy
    assert fingerprint_both


def test_uninstall_then_reuse_triggers_drift(tmp_path, monkeypatch):
    """Uninstall six (manual drift) → the next reuse recomputes a different
    resolved fingerprint → drift marked + structured refusal."""
    from reproagent.controller.actions import _tool_audit_env

    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)
    inventory.write_text(
        '[{"name": "numpy", "version": "1.26.4"}, {"name": "six", "version": "1.16.0"}]',
        encoding="utf-8",
    )
    state = _make_state(tmp_path, root, "https://github.com/org/demo.git")
    (tmp_path / "repo" / "requirements.txt").write_text("numpy\nsix\n", encoding="utf-8")

    info = env_module.ensure_environment(state)
    observation = _tool_audit_env(_agent_state(state, info.env_name))
    assert observation.audit is not None and observation.audit.success

    # manual drift: six uninstalled outside any run
    inventory.write_text('[{"name": "numpy", "version": "1.26.4"}]', encoding="utf-8")

    # reuse with the SAME spec (no requirements rewrite — _make_state would
    # change the file back and silently produce a different fingerprint)
    reuse_task = ReproTask(
        workspace_dir=tmp_path / "ws", reuse_mode="content_addressed",
        resource_root=str(root), python_version="3.10",
        repo_url="https://github.com/org/demo.git",
    )
    reuse_state = ReproState(task=reuse_task, repo_context=RepoContext(repo_path=tmp_path / "repo"))
    with pytest.raises(RuntimeError, match="drift detected"):
        env_module.ensure_environment(reuse_state)

    manifest = env_manager.list_manifests(root)[0]
    assert manifest["state"] == "drifted"


def test_spec_compliance_audit_fails_on_missing_package(tmp_path, monkeypatch):
    """The second drift line: a requirements-declared distribution missing
    from the environment fails the audit (no finalization)."""
    from reproagent.controller.actions import _tool_audit_env

    root = tmp_path / "resources"
    inventory, create_log = _install_fake(tmp_path, monkeypatch)
    inventory.write_text('[{"name": "numpy", "version": "1.26.4"}]', encoding="utf-8")
    state = _make_state(tmp_path, root, "https://github.com/org/demo.git")
    (tmp_path / "repo" / "requirements.txt").write_text("numpy\nsix\n", encoding="utf-8")

    info = env_module.ensure_environment(state)
    observation = _tool_audit_env(_agent_state(state, info.env_name))

    assert observation.audit is not None
    assert observation.audit.success is False
    assert any("missing distribution: six" in d for d in observation.audit.details)
    manifest = env_manager.list_manifests(root)[0]
    assert manifest["certification"] != "experiment"  # never finalized


# ── legacy regression ─────────────────────────────────────────────

def test_default_legacy_mode_unchanged(tmp_path, monkeypatch):
    """reuse_mode defaults to legacy: naming, env-list probing, and creation
    behave exactly as before the feature; resource_root is ignored."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["/fake/conda", "env", "list"]:
            return subprocess.CompletedProcess(cmd, 0, '{"envs": []}', "")
        if cmd[:2] == ["/fake/conda", "create"]:
            return subprocess.CompletedProcess(cmd, 0, "created", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(env_module, "find_conda", lambda: "/fake/conda")
    monkeypatch.setattr(env_module.subprocess, "run", fake_run)

    task = ReproTask(
        workspace_dir=tmp_path / "ws",
        resource_root=str(tmp_path / "resources"),  # ignored in legacy mode
    )
    state = ReproState(task=task, repo_context=RepoContext(repo_path=tmp_path / "repo"))
    info = env_module.ensure_environment(state)

    assert info.created is True
    assert info.env_name.startswith("repro_")  # legacy per-task naming
    assert not (tmp_path / "resources" / "environments").exists()
