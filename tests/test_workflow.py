"""Tests for the agent-loop controller."""
from reproagent.controller import run_controller, _parse_action
from reproagent.models import ReproTask


def test_parse_action_run_commands():
    text = '{"thinking": "test", "action": "run_commands", "stage_hint": "probe", "commands": ["ls"]}'
    action = _parse_action(text)
    assert action is not None
    assert action.action == "run_commands"
    assert action.commands == ["ls"]


def test_parse_action_finish():
    text = '{"thinking": "done", "action": "finish", "finish_status": "completed", "finish_summary": "All done."}'
    action = _parse_action(text)
    assert action is not None
    assert action.action == "finish"
    assert action.finish_status == "completed"


def test_parse_action_audit_env():
    text = '{"thinking": "check", "action": "audit_env"}'
    action = _parse_action(text)
    assert action is not None
    assert action.action == "audit_env"


def test_parse_action_invalid_returns_none():
    assert _parse_action("not json") is None
    assert _parse_action('{"action": "unknown"}') is None


def test_controller_mock_run_completes(tmp_path):
    task = ReproTask(
        paper_url="paper", repo_url="repo", workspace_dir=tmp_path / "run",
        experiment_goal="Run test.", mock_llm=True, max_steps=5,
    )
    state = run_controller(task)
    assert state.status == "completed"
    assert state.result_path is not None
    assert state.result_path.exists()


def test_controller_mock_steps_recorded(tmp_path):
    task = ReproTask(
        paper_url="paper", repo_url="repo", workspace_dir=tmp_path / "run",
        experiment_goal="Run test.", mock_llm=True, max_steps=5,
    )
    state = run_controller(task)
    assert len(state.steps) >= 1
