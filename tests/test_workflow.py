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


def test_revised_experiment_plan_is_validated_before_execution(tmp_path, monkeypatch):
    from reproagent import main
    from reproagent.models import CommandPlan, EnvironmentAudit, EnvironmentInfo, RepoContext, ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(main, "collect_context", lambda task: RepoContext(repo_path=repo, file_tree="examples/odenet_mnist.py", readme_text="demo"))
    monkeypatch.setattr(main, "ensure_environment", lambda state: EnvironmentInfo(env_name="repro_test", created=True))
    monkeypatch.setattr(main, "audit_environment", lambda state: EnvironmentAudit(success=True, summary="ok"))
    monkeypatch.setattr(main, "plan_environment", lambda state: CommandPlan(stage="environment", summary="env", commands=[]))
    monkeypatch.setattr(main, "plan_probe", lambda state: CommandPlan(stage="probe", summary="probe", commands=[]))
    monkeypatch.setattr(main, "review_experiment_plan_semantics", lambda state, plan: [])
    monkeypatch.setattr(
        main,
        "plan_experiment",
        lambda state: CommandPlan(
            stage="experiment",
            summary="unbounded initial plan",
            commands=["python examples/odenet_mnist.py --gpu 0"],
            feasibility="ready_to_run",
        ),
    )
    monkeypatch.setattr(
        main,
        "revise_after_failure",
        lambda state, stage: CommandPlan(
            stage="experiment",
            summary="bad revised setup-only plan",
            commands=["pip install torchvision", "python -c \"import torchvision; print(torchvision.__version__)\""],
            feasibility="ready_to_run",
        ),
    )

    seen_experiment_commands = []

    def fake_run_commands(commands, cwd, workspace, stage, attempt, timeout, env_name):
        if stage == "experiment":
            seen_experiment_commands.extend(commands)
            raise AssertionError("invalid revised experiment plan must not execute")
        return []

    monkeypatch.setattr(main, "run_commands", fake_run_commands)

    state = main.run_task(ReproTask(
        paper_url="paper",
        repo_url="repo",
        workspace_dir=tmp_path / "run",
        mock_llm=True,
        experiment_goal="Run a bounded GPU experiment using examples/odenet_mnist.py and report test accuracy.",
        max_run_attempts=2,
    ))

    assert state.status == "completed_with_failures"
    assert len(state.experiment_attempts) == 2
    assert state.experiment_attempts[1].plan.feasibility == "needs_config"
    assert any("only contains dependency/setup or inspection" in issue for issue in state.experiment_attempts[1].plan.needs_user_input)
    assert seen_experiment_commands == []


def test_llm_semantic_review_can_reject_ready_experiment_plan(tmp_path, monkeypatch):
    from reproagent import main
    from reproagent.models import CommandPlan, EnvironmentAudit, EnvironmentInfo, RepoContext, ReproTask

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(main, "collect_context", lambda task: RepoContext(repo_path=repo, file_tree="train.py", readme_text="demo"))
    monkeypatch.setattr(main, "ensure_environment", lambda state: EnvironmentInfo(env_name="repro_test", created=True))
    monkeypatch.setattr(main, "audit_environment", lambda state: EnvironmentAudit(success=True, summary="ok"))
    monkeypatch.setattr(main, "plan_environment", lambda state: CommandPlan(stage="environment", summary="env", commands=[]))
    monkeypatch.setattr(main, "plan_probe", lambda state: CommandPlan(stage="probe", summary="probe", commands=[]))
    monkeypatch.setattr(main, "review_experiment_plan_semantics", lambda state, plan: ["LLM reviewer says this does not measure the requested metric."])
    monkeypatch.setattr(
        main,
        "plan_experiment",
        lambda state: CommandPlan(
            stage="experiment",
            summary="run bounded",
            commands=["python train.py --epochs 1 --gpu 0"],
            feasibility="ready_to_run",
        ),
    )

    def fake_run_commands(commands, cwd, workspace, stage, attempt, timeout, env_name):
        if stage == "experiment":
            raise AssertionError("semantic validation failure must stop before execution")
        return []

    monkeypatch.setattr(main, "run_commands", fake_run_commands)

    state = main.run_task(ReproTask(
        paper_url="paper",
        repo_url="repo",
        workspace_dir=tmp_path / "run",
        mock_llm=False,
        experiment_goal="Run a bounded GPU experiment and report F1.",
        max_run_attempts=1,
    ))

    assert state.status == "completed_with_failures"
    assert state.experiment_attempts[0].plan.feasibility == "needs_config"
    assert "LLM reviewer" in state.experiment_attempts[0].plan.needs_user_input[0]
