from pathlib import Path

from reproagent.llm import _chat_completions_url
from reproagent.models import ReproTask


def test_chat_completions_url_from_base():
    assert _chat_completions_url("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1/chat/completions"


def test_chat_completions_url_accepts_full_endpoint():
    url = "https://example.com/v1/chat/completions"
    assert _chat_completions_url(url) == url


def test_task_api_defaults():
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=Path("runs/x"))
    assert task.api_base == "https://api.openai.com/v1"
    assert task.api_key_env == "OPENAI_API_KEY"
    assert task.backend == "conda"


def test_recent_logs_includes_environment_audit(tmp_path):
    from reproagent.llm import _recent_logs
    from reproagent.models import EnvironmentAudit, ReproState, ReproTask

    audit_stdout = tmp_path / "audit.stdout"
    audit_stdout.write_text("audit raw output", encoding="utf-8")
    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path),
        environment_audit=EnvironmentAudit(
            success=True,
            summary="Environment audit requires repair.",
            details=["GPU repair required"],
            has_warnings=True,
            requires_repair=True,
            stdout_path=audit_stdout,
        ),
    )

    logs = _recent_logs(state)

    assert "latest environment audit" in logs
    assert "requires_repair=True" in logs
    assert "GPU repair required" in logs
    assert "audit raw output" in logs


def test_plan_experiment_includes_goal_in_prompt(tmp_path, monkeypatch):
    from reproagent import llm
    from reproagent.models import RepoContext, ReproState, ReproTask

    captured = {}

    def fake_complete(state, prompt):
        captured["prompt"] = prompt
        return '{"stage":"experiment","summary":"s","commands":[],"assumptions":[]}'

    goal = "Run a bounded GPU MNIST ODE-Net experiment and report test accuracy."
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal=goal)
    state = ReproState(task=task, repo_context=RepoContext(repo_path=tmp_path))
    monkeypatch.setattr(llm, "_openai_compatible_text", fake_complete)

    llm.plan_experiment(state)

    assert goal in captured["prompt"]
    assert "experiment goal" in captured["prompt"].lower()
    assert "Experiment profile" not in captured["prompt"]
