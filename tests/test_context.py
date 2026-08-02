from pathlib import Path
import subprocess

from reproagent.context import _is_usable_repo, clone_repo
from reproagent.models import ReproTask


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

    monkeypatch.setattr("reproagent.context.subprocess.run", fake_run)
    monkeypatch.setattr("reproagent.context.time.sleep", lambda seconds: None)

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

    monkeypatch.setattr("reproagent.context.subprocess.run", fake_run)

    repo_path = clone_repo(task)

    assert repo_path == task.workspace_dir / "repo"
    assert (repo_path / "README.md").read_text(encoding="utf-8") == "from cache"
    assert not any(task.repo_url in cmd for cmd in calls)
