"""Deterministic environment identity — P0 golden fixture parity.

The fixtures are copies of the frozen M2-P0 contracts (ResAgent
contracts/fixtures/).  Fingerprints must be byte-identical across repos —
these tests fail if the algorithm drifts from the reference.
"""

import json
from pathlib import Path

from reproagent.runtime.env_identity import (
    canonical_dumps,
    collect_environment_spec,
    env_id,
    identity_subset,
    project_slug_for,
    resolved_fingerprint,
    slug_project,
    spec_fingerprint,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "m2"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ── golden fingerprint parity ─────────────────────────────────────

def test_golden_spec_fingerprints_byte_identical():
    golden = _load("fingerprint_golden.json")
    computed: dict[str, str] = {}
    for case in golden["cases"]:
        spec = _load(case["spec"])
        fp = spec_fingerprint(spec)
        computed[case["name"]] = fp
        assert fp == case["spec_fingerprint"], f"{case['name']}: {fp} != golden"


def test_golden_equal_relations():
    golden = _load("fingerprint_golden.json")
    for a, b in golden["equal"]:
        assert spec_fingerprint(_load(f"spec/{a}.json")) == spec_fingerprint(_load(f"spec/{b}.json"))


def test_golden_distinct_relations():
    golden = _load("fingerprint_golden.json")
    for a, b in golden["distinct"]:
        assert spec_fingerprint(_load(f"spec/{a}.json")) != spec_fingerprint(_load(f"spec/{b}.json"))


def test_notes_do_not_join_identity():
    """The two cuda124 fixtures differ only in notes → identical fingerprint."""
    a = _load("spec/torch_cuda124.json")
    b = _load("spec/torch_cuda124_altpath.json")
    assert identity_subset(a) == identity_subset(b)


# ── env_id / slug rules ───────────────────────────────────────────

def test_env_id_format():
    fp = "66630c82f6113079d7d193a02700f9b926417dedc0d35a44cfd35f53e1694d00"
    assert env_id("torchdiffeq", fp) == "resenv_torchdiffeq_66630c82f611"


def test_slug_project_normalization():
    assert slug_project("PyTorch Examples") == "pytorch-examples"
    assert slug_project("a//b  c") == "a-b-c"
    assert slug_project("") == "project"
    assert slug_project("---") == "project"


def test_project_slug_prefers_project_ref():
    from reproagent.models import ReproTask

    task = ReproTask(
        repo_url="https://github.com/org/paper-repo.git",
        project_ref="res-2026-demo",
        workspace_dir=Path("/tmp/ws"),
    )
    assert project_slug_for(task) == "res-2026-demo"


def test_project_slug_falls_back_to_repo_basename():
    from reproagent.models import ReproTask

    task = ReproTask(
        repo_url="https://github.com/org/paper-repo.git",
        workspace_dir=Path("/tmp/ws"),
    )
    assert project_slug_for(task) == "paper-repo"


def test_project_slug_uses_copy_basename_without_path():
    """Same repo at different paths must yield the same slug."""
    from reproagent.models import ReproTask

    a = ReproTask(copy_from="/data/repos/my_research", workspace_dir=Path("/tmp/wa"))
    b = ReproTask(copy_from="/other/path/my_research", workspace_dir=Path("/tmp/wb"))
    assert project_slug_for(a) == project_slug_for(b) == "my-research"


# ── resolved fingerprint ──────────────────────────────────────────

def test_resolved_fingerprint_stable_and_sensitive():
    base = {
        "python": "3.11.9",
        "conda_inventory_sha256": "a" * 64,
        "pip_inventory_sha256": "b" * 64,
        "frameworks": {"torch": {"version": "2.6.0", "cuda": "12.4"}},
        "abi_summary": "glibc2.35",
    }
    changed = {**base, "pip_inventory_sha256": "c" * 64}
    assert resolved_fingerprint(base) != resolved_fingerprint(changed)
    assert resolved_fingerprint(base) == resolved_fingerprint(json.loads(json.dumps(base)))
    empty = resolved_fingerprint({})
    assert empty == resolved_fingerprint({"python": None})


# ── spec collection ───────────────────────────────────────────────

def test_collect_spec_dependency_files_and_mirror(tmp_path, monkeypatch):
    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("torch==2.6.0\n", encoding="utf-8")
    (repo / "environment.yml").write_text("name: demo\n", encoding="utf-8")
    (repo / "train.py").write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module._contract, "probe_gpu_usable", lambda: False)  # no GPU

    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws",
                     python_version="3.10.9", mirror_profile="cn")
    spec = collect_environment_spec(task, repo)

    assert spec["schema"] == "ENVIRONMENT_SPEC_V1"
    assert spec["python"] == "3.10"
    assert spec["os"] == "linux"
    assert spec["arch"] == "x86_64"
    assert spec["accelerator"] == {"type": "cpu", "variant": ""}
    assert spec["pip_index_profile"] == "cn"  # caller's strategy name as-is
    paths = [entry["path"] for entry in spec["dependency_files"]]
    assert paths == sorted(paths)
    assert "requirements.txt" in paths and "environment.yml" in paths
    assert all(len(entry["sha256"]) == 64 for entry in spec["dependency_files"])


def test_canonical_dumps_matches_reference_shape():
    """Sorted keys, ASCII, no whitespace — the cross-repo invariant."""
    assert canonical_dumps({"b": 1, "a": {"c": 2}}) == '{"a":{"c":2},"b":1}'


# ── python version selection (RP1: one rule, from the contract) ───

def test_python_version_explicit_task_wins(tmp_path, monkeypatch):
    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "environment.yml").write_text("dependencies:\n  - python=3.9\n", encoding="utf-8")
    monkeypatch.setattr(module._contract, "probe_gpu_usable", lambda: False)
    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", python_version="3.11.5")
    spec = collect_environment_spec(task, repo)
    assert spec["python"] == "3.11"


def test_python_version_reads_environment_yml_pin(tmp_path, monkeypatch):
    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "environment.yml").write_text(
        "name: demo\ndependencies:\n  - python=3.9\n", encoding="utf-8")
    monkeypatch.setattr(module._contract, "probe_gpu_usable", lambda: False)
    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", python_version="")
    spec = collect_environment_spec(task, repo)
    assert spec["python"] == "3.9"


def test_python_version_defaults_without_any_source(tmp_path, monkeypatch):
    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(module._contract, "probe_gpu_usable", lambda: False)
    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", python_version="")
    spec = collect_environment_spec(task, repo)
    assert spec["python"] == "3.10"  # contract DEFAULT_PYTHON


def test_python_version_joins_identity(tmp_path, monkeypatch):
    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("torch==2.6.0\n", encoding="utf-8")
    monkeypatch.setattr(module._contract, "probe_gpu_usable", lambda: False)
    task_a = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", python_version="3.10")
    task_b = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", python_version="3.12")
    fp_a = spec_fingerprint(collect_environment_spec(task_a, repo))
    fp_b = spec_fingerprint(collect_environment_spec(task_b, repo))
    assert fp_a != fp_b


# ── accelerator collection semantics ──────────────────────────────

def test_accelerator_requires_gpu_false_never_probes(tmp_path, monkeypatch):
    import pytest

    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    monkeypatch.setattr(module._contract, "probe_gpu_usable",
                        lambda: pytest.fail("must not probe without requires_gpu"))
    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", requires_gpu=False)
    spec = collect_environment_spec(task, tmp_path)
    assert spec["accelerator"] == {"type": "cpu", "variant": ""}


def test_accelerator_requires_gpu_with_gpu_yields_cuda(tmp_path, monkeypatch):
    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    monkeypatch.setattr(module._contract, "probe_gpu_usable", lambda: True)
    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", requires_gpu=True)
    spec = collect_environment_spec(task, tmp_path)
    assert spec["accelerator"] == {"type": "cuda", "variant": ""}
    # the GPU probe NEVER contributes a wheel variant (no cu130 invention)
    assert "cu" not in spec["accelerator"]["variant"]


def test_accelerator_variant_from_explicit_constraint(tmp_path, monkeypatch):
    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("torch==2.6.0+cu124\n", encoding="utf-8")
    monkeypatch.setattr(module._contract, "probe_gpu_usable", lambda: True)
    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", requires_gpu=True)
    spec = collect_environment_spec(task, repo)
    assert spec["accelerator"] == {"type": "cuda", "variant": "cu124"}

    # conflicting variants are ambiguous → ""
    (repo / "requirements-gpu.txt").write_text("torch==2.4.0+cu121\n", encoding="utf-8")
    spec = collect_environment_spec(task, repo)
    assert spec["accelerator"]["variant"] == ""


def test_probe_gpu_usable_banner_rule(tmp_path, monkeypatch):
    """The vendored probe checks the NVIDIA-SMI banner, not a version field."""
    import shutil

    import reproagent.runtime.env_identity as module

    fake = tmp_path / "nvidia-smi"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'NVIDIA-SMI 610.57.01  KMD Version: 610.88  CUDA UMD Version: 13.3'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake))
    assert module._contract.probe_gpu_usable() is True

    fake.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Driver Version: 550.54'\n",
        encoding="utf-8",
    )
    assert module._contract.probe_gpu_usable() is False  # no banner — not usable

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert module._contract.probe_gpu_usable() is False


def test_requires_gpu_probe_failure_writes_warning(tmp_path, monkeypatch):
    import reproagent.runtime.env_identity as module
    from reproagent.models import ReproTask

    monkeypatch.setattr(module._contract, "probe_gpu_usable", lambda: False)
    log = tmp_path / "setup.stderr"
    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws", requires_gpu=True)
    spec = collect_environment_spec(task, tmp_path, probe_log=log)
    assert spec["accelerator"] == {"type": "cpu", "variant": ""}
    assert "accelerator warning" in log.read_text(encoding="utf-8")
