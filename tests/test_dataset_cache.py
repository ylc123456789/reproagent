"""Tests for dataset_cache: scan -> resolve -> link -> render."""

from pathlib import Path

from reproagent.runtime.dataset_cache import (
    prepare_dataset_links,
    render_dataset_block,
    scan_dataset_roots,
)
from reproagent.models import ReproTask


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create workspace/repo with the given files; return repo_path."""
    repo = tmp_path / "ws" / "repo"
    repo.mkdir(parents=True)
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return repo


def _make_cache(tmp_path: Path, layout: list[str]) -> Path:
    cache = tmp_path / "cache"
    for rel in layout:
        p = cache / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    return cache


# ── scanning ────────────────────────────────────────────────────────────────

def test_scan_torchvision_positional_resolves_against_repo_root(tmp_path):
    """'../data' in mnist/main.py must resolve to workspace/data — the cwd
    is the repo root, NOT the script's directory (the original bug)."""
    repo = _make_repo(tmp_path, {
        "mnist/main.py": "dataset = datasets.MNIST('../data', train=True, download=True)\n",
    })
    refs = scan_dataset_roots(repo)
    assert len(refs) == 1
    r = refs[0]
    assert r["dataset"] == "MNIST"
    assert r["declared"] == "../data"
    assert r["resolved"] == str((tmp_path / "ws" / "data").resolve())
    assert r["file"] == "mnist/main.py"


def test_scan_root_kwarg(tmp_path):
    repo = _make_repo(tmp_path, {
        "train.py": "datasets.CIFAR10(root='./data', train=False)\n",
    })
    refs = scan_dataset_roots(repo)
    assert refs[0]["dataset"] == "CIFAR10"
    assert refs[0]["resolved"] == str((tmp_path / "ws" / "repo" / "data").resolve())


def test_scan_argparse_default(tmp_path):
    repo = _make_repo(tmp_path, {
        "main.py": (
            "parser.add_argument('--data', type=str,\n"
            "                    default='./data/wikitext-2')\n"
        ),
    })
    refs = scan_dataset_roots(repo)
    assert len(refs) == 1
    assert refs[0]["resolved"].endswith("data/wikitext-2")


def test_scan_skips_absolute_url_and_interpolated(tmp_path):
    repo = _make_repo(tmp_path, {
        "a.py": "datasets.MNIST('/abs/data')\n"
                "datasets.CIFAR10('http://example.com/x')\n"
                "datasets.STL10(root=f'{home}/data')\n"
                "datasets.SVHN(root='$DATA/x')\n",
    })
    assert scan_dataset_roots(repo) == []


def test_scan_never_links_repo_or_workspace_root(tmp_path):
    repo = _make_repo(tmp_path, {"a.py": "datasets.MNIST('.', train=True)\n"})
    assert scan_dataset_roots(repo) == []


# ── link preparation ────────────────────────────────────────────────────────

def test_prepare_creates_symlink_and_detects_hit(tmp_path):
    repo = _make_repo(tmp_path, {
        "mnist/main.py": "datasets.MNIST('../data', download=True)\n",
    })
    cache = _make_cache(tmp_path, ["MNIST/raw/train-images-idx3-ubyte"])

    refs = prepare_dataset_links(
        repo_path=repo, workspace_dir=tmp_path / "ws", cache_root=str(cache),
    )
    r = refs[0]
    assert r["link"] == "created"
    assert r["cache_hit"] is True
    link = tmp_path / "ws" / "data"
    assert link.is_symlink()
    assert (link / "MNIST" / "raw" / "train-images-idx3-ubyte").exists()


def test_prepare_subdirectory_granularity(tmp_path):
    """./data/wikitext-2 should link to cache/wikitext-2, not the cache root."""
    repo = _make_repo(tmp_path, {
        "main.py": "parser.add_argument('--data', default='./data/wikitext-2')\n",
    })
    cache = _make_cache(tmp_path, ["wikitext-2/train.txt"])

    refs = prepare_dataset_links(
        repo_path=repo, workspace_dir=tmp_path / "ws", cache_root=str(cache),
    )
    link = repo / "data" / "wikitext-2"
    assert refs[0]["link"] == "created"
    assert link.is_symlink()
    assert (link / "train.txt").exists()


def test_prepare_no_cache_dir_is_noop(tmp_path):
    repo = _make_repo(tmp_path, {
        "mnist/main.py": "datasets.MNIST('../data')\n",
    })
    refs = prepare_dataset_links(
        repo_path=repo, workspace_dir=tmp_path / "ws",
        cache_root=str(tmp_path / "nonexistent"),
    )
    assert refs[0]["link"] == "no_cache"
    assert not (tmp_path / "ws" / "data").exists()


def test_prepare_never_clobbers_existing_dir(tmp_path):
    repo = _make_repo(tmp_path, {
        "mnist/main.py": "datasets.MNIST('../data')\n",
    })
    real_data = tmp_path / "ws" / "data"
    real_data.mkdir()
    (real_data / "keep.txt").write_text("keep", encoding="utf-8")
    cache = _make_cache(tmp_path, ["MNIST/raw/x"])

    refs = prepare_dataset_links(
        repo_path=repo, workspace_dir=tmp_path / "ws", cache_root=str(cache),
    )
    assert refs[0]["link"] == "exists"
    assert (real_data / "keep.txt").exists()  # untouched


# ── prompt rendering ────────────────────────────────────────────────────────

def test_render_block_shows_absolute_paths(tmp_path):
    repo = _make_repo(tmp_path, {
        "mnist/main.py": "datasets.MNIST('../data', download=True)\n",
    })
    cache = _make_cache(tmp_path, ["MNIST/raw/x"])
    refs = prepare_dataset_links(
        repo_path=repo, workspace_dir=tmp_path / "ws", cache_root=str(cache),
    )
    task = ReproTask(paper_url="p", repo_url="r",
                     workspace_dir=tmp_path / "ws",
                     dataset_cache_dir=str(cache))
    block = render_dataset_block(task, repo, refs)
    assert f"cwd = {repo}" in block
    assert str(tmp_path / "ws" / "data") in block
    assert "HIT" in block
    assert "do NOT re-download" in block


def test_render_block_empty_without_cache(tmp_path):
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path / "ws")
    assert render_dataset_block(task, tmp_path, []) == ""


def test_turn_prompt_carries_cwd_and_dataset_block(tmp_path):
    """Every rebuilt turn prompt must re-state the absolute cwd and mapping."""
    from reproagent.context_policy import ContextPolicy
    from reproagent.models import AgentState, RepoContext
    from reproagent.prompts import build_turn_prompt

    repo = _make_repo(tmp_path, {
        "mnist/main.py": "datasets.MNIST('../data', download=True)\n",
    })
    cache = _make_cache(tmp_path, ["MNIST/raw/x"])
    task = ReproTask(paper_url="p", repo_url="r",
                     workspace_dir=tmp_path / "ws",
                     dataset_cache_dir=str(cache))
    links = prepare_dataset_links(
        repo_path=repo, workspace_dir=tmp_path / "ws", cache_root=str(cache),
    )
    state = AgentState(task=task, repo_context=RepoContext(repo_path=repo),
                       dataset_links=links)
    state.steps.append(_FakeStep())  # force the turn-prompt path

    policy = ContextPolicy.for_model("deepseek-v4-pro")
    prompt = build_turn_prompt(state, policy)
    assert f"cwd = {repo}" in prompt
    assert "Dataset cache" in prompt
    assert "HIT" in prompt


def _FakeStep():
    from reproagent.models import AgentObservation
    return AgentObservation(step=1, action="run_commands", stage_hint="probe")


# ── end-to-end wiring through run_controller ────────────────────────────────

def test_run_controller_prelinks_cache(tmp_path):
    """Replay of the MNIST incident: hardcoded '../data' + populated cache.
    run_controller must pre-create workspace/data -> cache before the loop,
    so the experiment finds the cached dataset without downloading."""
    from reproagent.controller import run_controller

    ws = tmp_path / "ws"
    repo = ws / "repo"
    (repo / "mnist").mkdir(parents=True)
    (repo / "mnist" / "main.py").write_text(
        "datasets.MNIST('../data', download=True)\n", encoding="utf-8")
    cache = tmp_path / "cache"
    (cache / "MNIST" / "raw").mkdir(parents=True)
    (cache / "MNIST" / "raw" / "train-images-idx3-ubyte").write_text("x")

    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=ws,
                     dataset_cache_dir=str(cache), mock_llm=True)
    state = run_controller(task)

    assert state.dataset_links, "dataset_links should be populated"
    link = ws / "data"
    assert link.is_symlink()
    assert (link / "MNIST" / "raw" / "train-images-idx3-ubyte").exists()
    assert state.status == "completed"
