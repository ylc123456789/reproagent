from pathlib import Path
import subprocess

import pytest

from reproagent.repository.context import _is_usable_repo, clone_repo, collect_context, setup_workspace
from reproagent.models import ReproTask


def _make_git_repo(path: Path, content: str = "hello") -> Path:
    """Create a small real git repo (init + one commit)."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "README.md").write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        check=True,
    )
    return path


def test_rejects_half_cloned_git_dir(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    assert not _is_usable_repo(repo)


def test_accepts_uploaded_source_tree_without_git(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello", encoding="utf-8")
    (repo / "pkg").mkdir()

    assert _is_usable_repo(repo)


def test_clone_retries_and_removes_bad_partial_repo(tmp_path, monkeypatch):
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path)
    calls = []

    def fake_run(cmd, text, capture_output, timeout):
        calls.append(cmd)
        repo_path = tmp_path / "repo"
        if len(calls) == 1:
            (repo_path / ".git").mkdir(parents=True)
            return subprocess.CompletedProcess(cmd, 1, "", "network failed")
        repo_path.mkdir(parents=True, exist_ok=True)
        (repo_path / "README.md").write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "cloned", "")

    monkeypatch.setattr("reproagent.repository.context.subprocess.run", fake_run)
    monkeypatch.setattr("reproagent.repository.context.time.sleep", lambda seconds: None)

    repo_path = clone_repo(task)

    assert repo_path == tmp_path / "repo"
    assert len(calls) == 2
    assert (repo_path / "README.md").exists()
    assert (tmp_path / "logs" / "clone_01.stderr").read_text(encoding="utf-8") == "network failed"
    assert (tmp_path / "logs" / "clone_02.stdout").read_text(encoding="utf-8") == "cloned"


def test_clone_repo_uses_existing_cache_when_network_clone_fails(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "paper-repo"
    cached.mkdir(parents=True)
    (cached / "README.md").write_text("cached", encoding="utf-8")
    task = ReproTask(
        paper_url="paper",
        repo_url="https://github.com/example/paper-repo.git",
        workspace_dir=tmp_path / "run",
        repo_cache_dir=cache_dir,
    )
    calls = []

    def fake_run(cmd, text, capture_output, timeout):
        calls.append(cmd)
        if cmd[:3] == ["git", "-C", str(cached)]:
            return subprocess.CompletedProcess(cmd, 1, "", "offline")
        if len(cmd) >= 7 and cmd[5] == str(cached):
            repo_path = Path(cmd[-1])
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / "README.md").write_text("from cache", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "cached clone", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("reproagent.repository.context.subprocess.run", fake_run)

    repo_path = clone_repo(task)

    assert repo_path == task.workspace_dir / "repo"
    assert (repo_path / "README.md").read_text(encoding="utf-8") == "from cache"
    assert not any(task.repo_url in cmd for cmd in calls)


# ── workspace source modes (execution contract v1) ────────────────

def test_setup_workspace_isolated_clones_url(tmp_path):
    src = _make_git_repo(tmp_path / "upstream", content="upstream")
    task = ReproTask(repo_url=str(src), workspace_dir=tmp_path / "ws")

    repo = setup_workspace(task)

    assert repo == tmp_path / "ws" / "repo"
    assert (repo / "README.md").read_text(encoding="utf-8") == "upstream"


def test_setup_workspace_copy_preserves_uncommitted_changes(tmp_path):
    src = _make_git_repo(tmp_path / "src")
    (src / "train.py").write_text("dirty change", encoding="utf-8")  # not committed

    task = ReproTask(copy_from=str(src), workspace_dir=tmp_path / "ws")
    repo = setup_workspace(task)

    assert (repo / "train.py").read_text(encoding="utf-8") == "dirty change"
    assert (repo / ".git").is_dir()  # history preserved


def test_setup_workspace_shared_returns_external_in_place(tmp_path):
    ext = _make_git_repo(tmp_path / "external")
    task = ReproTask(external_repo_path=str(ext), workspace_dir=tmp_path / "ws")

    repo = setup_workspace(task)

    assert repo == ext.resolve()
    assert not (tmp_path / "ws" / "repo").exists()  # never copied


def test_setup_workspace_rejects_multiple_sources(tmp_path):
    task = ReproTask(
        repo_url="https://example.invalid/r.git", copy_from="/tmp/x",
        workspace_dir=tmp_path / "ws",
    )
    with pytest.raises(ValueError, match="Exactly one repository source"):
        setup_workspace(task)


def test_setup_workspace_rejects_all_three_sources(tmp_path):
    task = ReproTask(
        repo_url="https://example.invalid/r.git", copy_from="/tmp/x",
        external_repo_path="/tmp/y", workspace_dir=tmp_path / "ws",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        setup_workspace(task)


def test_setup_workspace_rejects_no_source_without_existing_repo(tmp_path):
    task = ReproTask(workspace_dir=tmp_path / "ws")
    with pytest.raises(ValueError, match="No repository source"):
        setup_workspace(task)


def test_setup_workspace_zero_sources_reuses_existing_workspace_repo(tmp_path):
    ws = tmp_path / "ws"
    repo = _make_git_repo(ws / "repo")
    task = ReproTask(workspace_dir=ws)  # all three sources empty → resume

    assert setup_workspace(task) == repo


def test_setup_workspace_rejects_bad_copy_source(tmp_path):
    task = ReproTask(copy_from=str(tmp_path / "missing"), workspace_dir=tmp_path / "ws")
    with pytest.raises(ValueError, match="not a usable repository"):
        setup_workspace(task)


def test_setup_workspace_rejects_bad_external_repo(tmp_path):
    task = ReproTask(external_repo_path=str(tmp_path / "missing"), workspace_dir=tmp_path / "ws")
    with pytest.raises(ValueError, match="not a usable repository"):
        setup_workspace(task)


def test_collect_context_shared_writes_artifacts_to_own_workspace(tmp_path):
    """Shared mode: repo not copied; summary/logs stay in the operator's workspace."""
    ext = _make_git_repo(tmp_path / "external")
    task = ReproTask(external_repo_path=str(ext), workspace_dir=tmp_path / "ws")

    ctx = collect_context(task)

    assert ctx.repo_path == ext.resolve()
    assert (tmp_path / "ws" / "context" / "context_summary.md").exists()
    assert not (tmp_path / "ws" / "repo").exists()
