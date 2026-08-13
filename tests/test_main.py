from pathlib import Path

from reproagent.main import build_parser
from reproagent.controller.loop import _git_dirty


def test_run_parser_requires_and_reads_experiment_goal(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run", "--paper", "paper", "--repo", "repo", "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST and report accuracy.",
    ])
    assert args.experiment_goal == "Run MNIST and report accuracy."


def test_run_parser_reads_confirm_before_experiment(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run", "--paper", "paper", "--repo", "repo", "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST.", "--confirm-before-experiment",
    ])
    assert args.confirm_before_experiment


def test_run_parser_reads_repo_cache_dir(tmp_path):
    parser = build_parser()
    cache_dir = tmp_path / "repo-cache"
    args = parser.parse_args([
        "run", "--paper", "paper", "--repo", "repo",
        "--workspace", str(tmp_path / "run"), "--repo-cache-dir", str(cache_dir),
        "--experiment-goal", "Run MNIST.",
    ])
    assert args.repo_cache_dir == cache_dir


def test_git_dirty_ignores_untracked_run_artifacts(tmp_path):
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    (tmp_path / "runs" / "old").mkdir(parents=True)
    (tmp_path / "runs" / "old" / "artifact.txt").write_text("artifact", encoding="utf-8")
    assert _git_dirty(tmp_path) is False
    tracked.write_text("dirty", encoding="utf-8")
    assert _git_dirty(tmp_path) is True


def test_run_parser_reads_coding_agent_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run", "--paper", "paper", "--repo", "repo", "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST.", "--enable-coding-agent", "--max-coding-agent-steps", "5",
    ])
    assert args.enable_coding_agent
    assert args.max_coding_agent_steps == 5


def test_run_parser_reads_mirror_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run", "--paper", "paper", "--repo", "repo", "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST.", "--mirror-profile", "autodl", "--mirror-strict",
    ])
    assert args.mirror_profile == "autodl"
    assert args.mirror_strict


def test_run_parser_reads_codingagent_path_and_config(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run", "--paper", "paper", "--repo", "repo", "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST.", "--enable-coding-agent",
        "--codingagent-path", "/home/cyl/CodingAgent",
        "--config", str(tmp_path / "reproagent.yaml"),
    ])
    assert args.codingagent_path == Path("/home/cyl/CodingAgent")
    assert args.config == tmp_path / "reproagent.yaml"


def test_run_parser_paper_and_repo_now_optional(tmp_path):
    """Operator contract: --paper/--repo optional; workspace sources choose the mode."""
    parser = build_parser()
    args = parser.parse_args([
        "run", "--external-repo", str(tmp_path / "repo"),
        "--workspace", str(tmp_path / "ws"),
        "--experiment-goal", "Run MNIST.",
    ])
    assert args.paper == ""
    assert args.repo == ""
    assert args.external_repo == str(tmp_path / "repo")


def test_run_parser_reads_copy_from_and_setup_only(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run", "--copy-from", str(tmp_path / "src"),
        "--workspace", str(tmp_path / "ws"),
        "--experiment-goal", "prepare env", "--setup-only",
    ])
    assert args.copy_from == str(tmp_path / "src")
    assert args.setup_only
