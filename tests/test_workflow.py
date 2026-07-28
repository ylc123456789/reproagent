from pathlib import Path

from reproagent.models import CommandResult, EnvironmentAudit, EnvironmentInfo, RepoContext, ReproTask


def test_plan_only_runs_probe_and_does_not_run_experiment(tmp_path, monkeypatch):
    from reproagent import main

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("demo", encoding="utf-8")

    monkeypatch.setattr(main, "collect_context", lambda task: RepoContext(repo_path=repo, file_tree="train.py", readme_text="demo"))
    monkeypatch.setattr(main, "ensure_environment", lambda state: EnvironmentInfo(env_name="repro_test", created=True))
    monkeypatch.setattr(main, "audit_environment", lambda state: EnvironmentAudit(success=True, summary="ok"))

    seen_stages = []

    def fake_run_commands(commands, cwd, workspace, stage, attempt, timeout, env_name):
        seen_stages.append(stage)
        if stage == "experiment":
            raise AssertionError("plan-only must not execute experiment commands")
        results = []
        for index, command in enumerate(commands, start=1):
            stdout = workspace / "logs" / f"{stage}_{attempt:02d}_{index:02d}.stdout"
            stderr = workspace / "logs" / f"{stage}_{attempt:02d}_{index:02d}.stderr"
            stdout.parent.mkdir(parents=True, exist_ok=True)
            stdout.write_text("probe output", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            results.append(CommandResult(command=command, exit_code=0, stdout_path=stdout, stderr_path=stderr, duration_seconds=0.1))
        return results

    monkeypatch.setattr(main, "run_commands", fake_run_commands)

    task = ReproTask(
        paper_url="paper",
        repo_url="repo",
        workspace_dir=tmp_path / "run",
        mock_llm=True,
        experiment_goal="Generate a bounded experiment plan.",
        plan_only=True,
    )

    state = main.run_task(task)

    assert state.status == "planned"
    assert state.planned_experiment is not None
    assert state.planned_experiment.stage == "experiment"
    assert state.probe_attempts
    assert not state.experiment_attempts
    assert seen_stages == ["probe"]
    assert (task.workspace_dir / "result.md").exists()


def test_experiment_validation_failure_stops_before_running_commands(tmp_path, monkeypatch):
    from reproagent import main
    from reproagent.models import CommandPlan, EnvironmentAudit, EnvironmentInfo, RepoContext, ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(main, "collect_context", lambda task: RepoContext(repo_path=repo, file_tree="train.py", readme_text="demo"))
    monkeypatch.setattr(main, "ensure_environment", lambda state: EnvironmentInfo(env_name="repro_test", created=True))
    monkeypatch.setattr(main, "audit_environment", lambda state: EnvironmentAudit(success=True, summary="ok"))
    monkeypatch.setattr(main, "plan_environment", lambda state: CommandPlan(stage="environment", summary="env", commands=[]))
    monkeypatch.setattr(main, "plan_probe", lambda state: CommandPlan(stage="probe", summary="probe", commands=[]))
    monkeypatch.setattr(
        main,
        "plan_experiment",
        lambda state: CommandPlan(
            stage="experiment",
            summary="bad plan",
            commands=["python train.py"],
            assumptions=["The script likely prints training loss."],
            feasibility="ready_to_run",
        ),
    )

    def fake_run_commands(commands, cwd, workspace, stage, attempt, timeout, env_name):
        if stage == "experiment":
            raise AssertionError("validation failure must stop before experiment execution")
        return []

    monkeypatch.setattr(main, "run_commands", fake_run_commands)

    state = main.run_task(ReproTask(
        paper_url="paper",
        repo_url="repo",
        workspace_dir=tmp_path / "run",
        mock_llm=True,
        experiment_goal="Run a bounded GPU experiment and report training loss.",
    ))

    assert state.status == "completed_with_failures"
    assert state.experiment_attempts
    assert state.experiment_attempts[0].plan.feasibility == "needs_patch"
    assert not state.experiment_attempts[0].results
