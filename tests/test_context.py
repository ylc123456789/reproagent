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
