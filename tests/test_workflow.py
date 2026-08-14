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
    assert "command not allowed" in observation.error


def test_setup_only_blocks_script_execution_any_stage(tmp_path):
    """The deterministic whitelist does not trust stage_hint: script
    execution is blocked even when the model labels it environment/probe."""
    task = ReproTask(
        repo_url="repo", workspace_dir=tmp_path, setup_only=True, mock_llm=True,
    )
    state = AgentState(task=task)

    for stage_hint in ("environment", "probe", "experiment"):
        action = AgentAction(
            thinking="try", action="run_commands", stage_hint=stage_hint,
            commands=["bash run.sh", "./train.py"],
        )
        observation = _tool_run_commands(action, state)
        assert not observation.command_results
        assert "command not allowed" in observation.error


def test_setup_only_allows_non_experiment_commands(tmp_path):
    """Provisioning and inspection commands remain available during setup_only."""
    task = ReproTask(
        repo_url="repo", workspace_dir=tmp_path, setup_only=True, mock_llm=True,
    )
    state = AgentState(task=task)
    action = AgentAction(
        thinking="env", action="run_commands", stage_hint="environment",
        commands=["echo ok", "pwd", "ls"],
    )

    observation = _tool_run_commands(action, state)

    assert len(observation.command_results) == 3
    assert all(r.exit_code == 0 for r in observation.command_results)


def test_setup_only_mock_run_without_audit_ends_failed(tmp_path):
    """Deterministic gate: no successful audit → setup_only cannot complete."""
    task = ReproTask(
        repo_url="repo", workspace_dir=tmp_path / "run",
        experiment_goal="prepare env", mock_llm=True, max_steps=5, setup_only=True,
    )
    state = run_controller(task)

    assert state.status == "failed"
    assert "## Environment Provisioning" in state.final_summary
    assert "mock_env" in state.final_summary
    assert "requires a successful" in state.final_summary


def test_env_setup_uses_resolved_repo_path(tmp_path, monkeypatch):
    """Shared mode: the conda env is created against the RESOLVED (external)
    repo path — never a rebuilt workspace/repo path. environment.yml lookup
    and setup cwd both follow repo_context.repo_path."""
    from reproagent.controller.loop import ensure_environment_for_controller
    from reproagent.models import EnvironmentInfo, RepoContext

    captured = {}
    monkeypatch.setattr(
        "reproagent.controller.loop.ensure_environment",
        lambda env_state: (
            captured.update(repo_path=env_state.repo_context.repo_path)
            or EnvironmentInfo(env_name="e")
        ),
    )
    external = tmp_path / "project" / "repos" / "foo"
    repo_context = RepoContext(repo_path=external)
    task = ReproTask(external_repo_path=str(external), workspace_dir=tmp_path / "ws")

    ensure_environment_for_controller(task, repo_context)

    assert captured["repo_path"] == external


def test_certification_gate_blocks_experiment_before_audit(tmp_path):
    """No experiment command runs before a successful environment audit —
    regardless of the LLM's stage label."""
    task = ReproTask(repo_url="repo", workspace_dir=tmp_path, mock_llm=True)
    state = AgentState(task=task)  # last_audit is None

    for stage_hint in ("environment", "probe", "experiment"):
        action = AgentAction(
            thinking="try", action="run_commands", stage_hint=stage_hint,
            commands=["python train.py"],
        )
        observation = _tool_run_commands(action, state)
        assert not observation.command_results
        assert "not yet certified" in observation.error


def test_certification_gate_allows_repair_commands(tmp_path):
    """Dependency repair (pip-family) and inspection stay available pre-audit."""
    task = ReproTask(repo_url="repo", workspace_dir=tmp_path, mock_llm=True)
    state = AgentState(task=task)
    action = AgentAction(
        thinking="repair", action="run_commands", stage_hint="environment",
        commands=["echo repair", "pwd"],
    )

    observation = _tool_run_commands(action, state)

    assert len(observation.command_results) == 2
    assert all(r.exit_code == 0 for r in observation.command_results)


def test_certification_gate_lifts_after_successful_audit(tmp_path):
    """After a successful audit, experiment commands pass the gate."""
    from reproagent.models import EnvironmentAudit

    task = ReproTask(repo_url="repo", workspace_dir=tmp_path, mock_llm=True)
    state = AgentState(
        task=task,
        last_audit=EnvironmentAudit(success=True, summary="passed"),
    )
    action = AgentAction(
        thinking="run", action="run_commands", stage_hint="experiment",
        commands=["python -c \"print('epoch 1')\""],
    )

    observation = _tool_run_commands(action, state)

    assert observation.command_results
    assert observation.command_results[0].exit_code == 0


def test_certification_gate_blocks_inline_python_before_audit(tmp_path):
    """python -c is NOT a setup-safe escape hatch: arbitrary inline programs
    are blocked before audit regardless of stage labels."""
    task = ReproTask(repo_url="repo", workspace_dir=tmp_path, mock_llm=True)
    state = AgentState(task=task)
    action = AgentAction(
        thinking="try", action="run_commands", stage_hint="environment",
        commands=["python -c \"import torch; torch.save(model, 'p.pt')\""],
    )

    observation = _tool_run_commands(action, state)

    assert not observation.command_results
    assert "not yet certified" in observation.error


def test_env_mutating_command_invalidates_audit(tmp_path, monkeypatch):
    """A successful audit goes stale after a dependency change."""
    import pytest

    import reproagent.controller.actions as actions_module
    from reproagent.models import CommandResult, EnvironmentAudit

    # single-command mutations only — compound mutations are rejected by the
    # mutation policy before execution (see test_mutation_action_rejects_*)
    mutating = [
        "pip install torch", "python -m pip install torch", "conda install pytorch",
        "conda env update -f environment.yml",
        "pip uninstall torch", "uv pip install x", "poetry add x",
    ]
    for command in mutating:
        task = ReproTask(repo_url="repo", workspace_dir=tmp_path, mock_llm=True)
        state = AgentState(
            task=task,
            last_audit=EnvironmentAudit(success=True, summary="passed"),
        )
        dummy = CommandResult(
            command=command, exit_code=0,
            stdout_path=tmp_path / "o", stderr_path=tmp_path / "e", duration_seconds=0.0,
        )
        monkeypatch.setattr(actions_module, "_mock_run_commands", lambda commands, state: [dummy])

        action = AgentAction(
            thinking="install", action="run_commands", stage_hint="environment",
            commands=[command],
        )
        _tool_run_commands(action, state)

        assert state.last_audit is None, command


def test_non_mutating_command_keeps_audit(tmp_path, monkeypatch):
    """Inspection and non-install pip commands do not invalidate the audit."""
    import reproagent.controller.actions as actions_module
    from reproagent.models import CommandResult, EnvironmentAudit

    for command in ("pip show torch", "pip list", "echo ok", "head -20 README.md"):
        task = ReproTask(repo_url="repo", workspace_dir=tmp_path, mock_llm=True)
        state = AgentState(
            task=task,
            last_audit=EnvironmentAudit(success=True, summary="passed"),
        )
        dummy = CommandResult(
            command=command, exit_code=0,
            stdout_path=tmp_path / "o", stderr_path=tmp_path / "e", duration_seconds=0.0,
        )
        monkeypatch.setattr(actions_module, "_mock_run_commands", lambda commands, state: [dummy])

        _tool_run_commands(
            AgentAction(thinking="c", action="run_commands", stage_hint="probe",
                        commands=[command]),
            state,
        )
        assert state.last_audit is not None, command


def test_certification_gate_rejects_compound_before_audit(tmp_path):
    """Compound commands are blocked pre-audit (not just the first word)."""
    task = ReproTask(repo_url="repo", workspace_dir=tmp_path, mock_llm=True)
    state = AgentState(task=task)
    for command in ("echo ok && python train.py", "pip install x && python train.py",
                    "echo ok && pip install x"):
        observation = _tool_run_commands(
            AgentAction(thinking="try", action="run_commands", stage_hint="environment",
                        commands=[command]),
            state,
        )
        assert not observation.command_results, command
        assert "not yet certified" in observation.error


def _certified_state(tmp_path, task=None):
    from reproagent.models import EnvironmentAudit

    return AgentState(
        task=task or ReproTask(repo_url="repo", workspace_dir=tmp_path, mock_llm=True),
        last_audit=EnvironmentAudit(success=True, summary="passed"),
    )


def test_mutation_action_rejects_compound_mutation(tmp_path):
    """Bypass 1: a compound mutation in one list item is refused post-audit —
    the training half must never run against the mutated environment."""
    state = _certified_state(tmp_path)
    for command in ("pip install x && python train.py", "echo ok && pip install torch"):
        observation = _tool_run_commands(
            AgentAction(thinking="t", action="run_commands", stage_hint="environment",
                        commands=[command]),
            state,
        )
        assert not observation.command_results, command
        assert "must be single commands" in observation.error
        assert state.last_audit is not None  # nothing executed, certification intact


def test_mutation_action_rejects_experiment_in_same_action(tmp_path):
    """Bypass 2: ['pip install x', 'python train.py'] in one action is refused —
    the experiment item would run against the un-audited mutated environment."""
    state = _certified_state(tmp_path)
    observation = _tool_run_commands(
        AgentAction(thinking="t", action="run_commands", stage_hint="experiment",
                    commands=["pip install x", "python train.py"]),
        state,
    )

    assert not observation.command_results
    assert "cannot share one action" in observation.error
    assert state.last_audit is not None


def test_mutation_action_allows_mutation_plus_inspection(tmp_path, monkeypatch):
    """A mutation action may carry inspection commands; the audit is cleared
    when the action completes so the NEXT action must re-audit."""
    import reproagent.controller.actions as actions_module
    from reproagent.models import CommandResult

    state = _certified_state(tmp_path)
    dummy = CommandResult(
        command="x", exit_code=0,
        stdout_path=tmp_path / "o", stderr_path=tmp_path / "e", duration_seconds=0.0,
    )
    monkeypatch.setattr(actions_module, "_mock_run_commands", lambda commands, state: [dummy] * len(commands))

    observation = _tool_run_commands(
        AgentAction(thinking="t", action="run_commands", stage_hint="environment",
                    commands=["pip install x", "head -20 README.md"]),
        state,
    )

    assert len(observation.command_results) == 2
    assert state.last_audit is None  # cleared immediately at action end


def test_stale_audit_blocks_experiment_until_re_audit(tmp_path, monkeypatch):
    """audit → pip install → experiment: the experiment is refused because
    the install invalidated the audit; a fresh audit re-opens the gate."""
    import reproagent.controller.actions as actions_module
    import reproagent.llm as llm_module
    from reproagent.models import CommandResult, EnvironmentAudit

    dummy = CommandResult(
        command="pip install torch", exit_code=0,
        stdout_path=tmp_path / "o", stderr_path=tmp_path / "e", duration_seconds=0.0,
    )
    monkeypatch.setattr(actions_module, "_mock_run_commands", lambda commands, state: [dummy])
    monkeypatch.setattr(
        actions_module, "audit_environment",
        lambda state: EnvironmentAudit(success=True, summary="passed"),
    )
    responses = iter([
        '{"thinking": "audit", "action": "audit_env"}',
        '{"thinking": "install", "action": "run_commands", "stage_hint": "environment", '
        '"commands": ["pip install torch"]}',
        '{"thinking": "train", "action": "run_commands", "stage_hint": "experiment", '
        '"commands": ["python train.py"]}',
        '{"thinking": "done", "action": "finish", "finish_status": "completed", '
        '"finish_summary": "done"}',
    ])
    monkeypatch.setattr(llm_module, "mock_response", lambda user: next(responses))

    task = ReproTask(repo_url="repo", workspace_dir=tmp_path / "run",
                     experiment_goal="g", mock_llm=True, max_steps=6)
    state = run_controller(task)

    blocked = [s for s in state.steps if s.error and "not yet certified" in s.error]
    assert blocked, "experiment after pip install must be blocked"
    assert state.status == "completed"


def test_setup_only_mock_run_with_successful_audit_completes(tmp_path, monkeypatch):
    """With a passing audit, the mock setup_only run completes."""
    import reproagent.controller.actions as actions_module
    import reproagent.llm as llm_module
    from reproagent.models import EnvironmentAudit

    monkeypatch.setattr(
        actions_module, "audit_environment",
        lambda state: EnvironmentAudit(success=True, summary="passed", details=["torch ok"]),
    )
    responses = iter([
        '{"thinking": "audit", "action": "audit_env"}',
        '{"thinking": "done", "action": "finish", "finish_status": "completed", '
        '"finish_summary": "env ready"}',
    ])
    monkeypatch.setattr(llm_module, "mock_response", lambda user: next(responses))

    task = ReproTask(
        repo_url="repo", workspace_dir=tmp_path / "run",
        experiment_goal="prepare env", mock_llm=True, max_steps=5, setup_only=True,
    )
    state = run_controller(task)

    assert state.status == "completed"
    assert "## Environment Provisioning" in state.final_summary
    assert "PASSED" in state.final_summary


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
    assert not state.coding_results  # no delegation actually happened
    # structured field for orchestrators — no text parsing needed
    blocked_obs = next(step for step in state.steps if step.action == "call_coding_agent")
    assert blocked_obs.coding_issues == ["missing loss"]
