from reproagent.llm import _chat_completions_url
from reproagent.controller.prompts import SYSTEM_PROMPT, build_initial_context
from reproagent.models import EnvironmentInfo, RepoContext, ReproTask


def test_chat_completions_url_from_base():
    assert _chat_completions_url("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1/chat/completions"


def test_chat_completions_url_accepts_full_endpoint():
    assert _chat_completions_url("https://example.com/v1/chat/completions") == "https://example.com/v1/chat/completions"


def test_system_prompt_includes_safety_rules():
    assert "sudo" in SYSTEM_PROMPT
    assert "rm -rf" in SYSTEM_PROMPT
    assert "call_coding_agent" in SYSTEM_PROMPT


def test_system_prompt_includes_tools():
    for tool in ("run_commands", "audit_env", "call_coding_agent", "finish"):
        assert tool in SYSTEM_PROMPT


def test_build_context_includes_goal_and_env(tmp_path):
    task = ReproTask(
        paper_url="https://arxiv.org/abs/1234.5678",
        repo_url="https://github.com/user/repo",
        workspace_dir=tmp_path,
        experiment_goal="Run bounded MNIST.",
    )
    ctx = RepoContext(
        repo_path=tmp_path / "repo",
        commit_hash="abc123",
        file_tree="train.py\nREADME.md",
        readme_text="## README",
        hardware_text="GPU: RTX 4090",
    )
    env = EnvironmentInfo(env_name="repro_test", created=True)
    text = build_initial_context(task, ctx, env)
    assert "Run bounded MNIST" in text
    assert "repro_test" in text
    assert "freshly created" in text
    assert "RTX 4090" in text


def test_build_context_reused_env(tmp_path):
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path, experiment_goal="g")
    ctx = RepoContext(repo_path=tmp_path / "repo", hardware_text="cpu", readme_text="r", file_tree="f")
    env = EnvironmentInfo(env_name="repro_test", created=False)
    text = build_initial_context(task, ctx, env)
    assert "reused" in text.lower()


def test_build_context_renders_input_artifacts(tmp_path):
    task = ReproTask(
        repo_url="r", workspace_dir=tmp_path, experiment_goal="g",
        input_artifacts=[
            {"path": "/contract/prior/measurements.json", "description": "baseline metrics"},
            {"path": "/contract/prior/checkpoint.pt"},
        ],
    )
    ctx = RepoContext(repo_path=tmp_path / "repo", hardware_text="cpu", readme_text="r", file_tree="f")
    text = build_initial_context(task, ctx, EnvironmentInfo(env_name="e"))

    assert "## Input Artifacts" in text
    assert "/contract/prior/measurements.json — baseline metrics" in text
    assert "/contract/prior/checkpoint.pt" in text


def test_build_context_omits_input_artifacts_when_empty(tmp_path):
    task = ReproTask(repo_url="r", workspace_dir=tmp_path, experiment_goal="g")
    ctx = RepoContext(repo_path=tmp_path / "repo", hardware_text="cpu", readme_text="r", file_tree="f")
    text = build_initial_context(task, ctx, EnvironmentInfo(env_name="e"))
    assert "Input Artifacts" not in text
