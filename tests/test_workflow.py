"""Tests for the agent-loop controller."""
from reproagent.controller import run_controller
from reproagent.controller.actions import _parse_action, _tool_run_commands
from reproagent.models import AgentAction, AgentState, ReproTask


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


# ── setup_only (environment provisioning) ─────────────────────────

def test_setup_only_blocks_experiment_commands(tmp_path):
    """setup_only tasks must never execute experiment-stage commands."""
    task = ReproTask(
        repo_url="repo", workspace_dir=tmp_path, setup_only=True, mock_llm=True,
    )
    state = AgentState(task=task)
    action = AgentAction(
        thinking="try", action="run_commands", stage_hint="experiment",
        commands=["python train.py"],
    )

    observation = _tool_run_commands(action, state)

    assert not observation.command_results
    assert "experiment commands are not allowed" in observation.error


def test_setup_only_allows_non_experiment_commands(tmp_path):
    """Environment/probe commands remain available during setup_only."""
    task = ReproTask(
        repo_url="repo", workspace_dir=tmp_path, setup_only=True, mock_llm=True,
    )
    state = AgentState(task=task)
    action = AgentAction(
        thinking="env", action="run_commands", stage_hint="environment",
        commands=["echo ok"],
    )

    observation = _tool_run_commands(action, state)

    assert observation.command_results
    assert observation.command_results[0].exit_code == 0


def test_setup_only_mock_run_appends_provisioning_summary(tmp_path):
    """Mock setup_only run finishes with a deterministic env provisioning section."""
    task = ReproTask(
        repo_url="repo", workspace_dir=tmp_path / "run",
        experiment_goal="prepare env", mock_llm=True, max_steps=5, setup_only=True,
    )
    state = run_controller(task)

    assert state.status == "completed"
    assert "## Environment Provisioning" in state.final_summary
    assert "mock_env" in state.final_summary


# ── code delegation switch (contract O9) ──────────────────────────

def test_controller_delegation_disabled_exits_blocked(tmp_path, monkeypatch):
    """allow_code_delegation=False: call_coding_agent ends the run blocked
    with the coding issues listed, instead of delegating."""
    import reproagent.llm as llm_module

    monkeypatch.setattr(
        llm_module,
        "mock_response",
        lambda user: '{"thinking": "need patch", "action": "call_coding_agent", '
                     '"coding_goal": "add loss logging", "coding_issues": ["missing loss"]}',
    )
    task = ReproTask(
        repo_url="repo", workspace_dir=tmp_path / "run",
        experiment_goal="g", mock_llm=True, max_steps=5,
        allow_code_delegation=False,
    )
    state = run_controller(task)

    assert state.status == "blocked"
    assert "missing loss" in state.final_summary
    assert any(step.action == "call_coding_agent" for step in state.steps)
    assert not state.coding_results  # no delegation actually happened
