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
    monkeypatch.setattr(module.shutil, "which", lambda name: None)  # no GPU

    task = ReproTask(repo_url="r", workspace_dir=tmp_path / "ws",
                     python_version="3.10.9", mirror_profile="cn")
    spec = collect_environment_spec(task, repo)

    assert spec["schema"] == "ENVIRONMENT_SPEC_V1"
    assert spec["python"] == "3.10"
    assert spec["os"] == "linux"
    assert spec["arch"] == "x86_64"
    assert spec["accelerator"] == {"type": "cpu", "variant": ""}
    assert spec["pip_index_profile"] == "aliyun"
    paths = [entry["path"] for entry in spec["dependency_files"]]
    assert paths == sorted(paths)
    assert "requirements.txt" in paths and "environment.yml" in paths
    assert all(len(entry["sha256"]) == 64 for entry in spec["dependency_files"])


def test_canonical_dumps_matches_reference_shape():
    """Sorted keys, ASCII, no whitespace — the cross-repo invariant."""
    assert canonical_dumps({"b": 1, "a": {"c": 2}}) == '{"a":{"c":2},"b":1}'
