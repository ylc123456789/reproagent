from reproagent.main import _final_summary, _run_probe_once_after_patch, _stage_succeeded, build_parser
from reproagent.models import CodingAgentResult, CommandPlan, CommandResult, EnvironmentInfo, RepoContext, ReproState, ReproTask, StageResult


def test_run_parser_requires_and_reads_experiment_goal(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--paper", "paper",
        "--repo", "repo",
        "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST and report accuracy.",
    ])

    assert args.experiment_goal == "Run MNIST and report accuracy."
    assert not args.confirm_before_experiment


def test_run_parser_reads_confirm_before_experiment(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--paper", "paper",
        "--repo", "repo",
        "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST and report accuracy.",
        "--confirm-before-experiment",
    ])

    assert args.confirm_before_experiment


def test_run_parser_reads_plan_only(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--paper", "paper",
        "--repo", "repo",
        "--workspace", str(tmp_path),
        "--experiment-goal", "Generate a bounded training plan.",
        "--plan-only",
    ])

    assert args.plan_only


def test_run_parser_reads_coding_agent_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--paper", "paper",
        "--repo", "repo",
        "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST and report accuracy.",
        "--enable-coding-agent",
        "--max-coding-agent-steps", "5",
    ])

    assert args.enable_coding_agent
    assert args.max_coding_agent_steps == 5


def test_final_summary_reports_blocked_coding_agent_without_llm(tmp_path, monkeypatch):
    def fail_final_review(state):
        raise AssertionError("LLM final review should not run after a blocking CodingAgent failure")

    import reproagent.main as main_module

    monkeypatch.setattr(main_module, "final_review", fail_final_review)
    state = ReproState(
        task=ReproTask(
            paper_url="paper",
            repo_url="repo",
            workspace_dir=tmp_path,
            experiment_goal="Run bounded MNIST.",
        ),
        planned_experiment=CommandPlan(
            stage="experiment",
            summary="Needs loss logging patch.",
            commands=["python train.py"],
            feasibility="needs_patch",
            needs_user_input=["Training loss is unavailable without a patch."],
        ),
        coding_agent_results=[
            CodingAgentResult(
                status="failed",
                summary="git apply failed: corrupt patch",
                report_path=tmp_path / "patch_report.md",
                output_dir=tmp_path / "patches" / "coding_agent_01",
            )
        ],
    )

    summary = _final_summary(state)

    assert "Experiment commands were not executed" in summary
    assert "CodingAgent status: failed" in summary
    assert "corrupt patch" in summary
    assert "Training loss is unavailable" in summary


def test_probe_stage_succeeds_with_partial_probe_failures(tmp_path):
    ok_stdout = tmp_path / "ok.stdout"
    ok_stderr = tmp_path / "ok.stderr"
    fail_stdout = tmp_path / "fail.stdout"
    fail_stderr = tmp_path / "fail.stderr"
    for path in [ok_stdout, ok_stderr, fail_stdout, fail_stderr]:
        path.write_text("", encoding="utf-8")
    stage_result = StageResult(
        stage="probe",
        attempt=1,
        plan=CommandPlan(stage="probe", summary="probe", commands=["python --help", "bad probe"]),
        results=[
            CommandResult(command="python --help", exit_code=0, stdout_path=ok_stdout, stderr_path=ok_stderr, duration_seconds=0.1),
            CommandResult(command="bad probe", exit_code=1, stdout_path=fail_stdout, stderr_path=fail_stderr, duration_seconds=0.1),
        ],
    )

    assert _stage_succeeded("probe", stage_result)
    assert not _stage_succeeded("experiment", stage_result)


def test_probe_stage_fails_when_all_probe_commands_fail(tmp_path):
    stdout = tmp_path / "fail.stdout"
    stderr = tmp_path / "fail.stderr"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    stage_result = StageResult(
        stage="probe",
        attempt=1,
        plan=CommandPlan(stage="probe", summary="probe", commands=["bad probe"]),
        results=[CommandResult(command="bad probe", exit_code=1, stdout_path=stdout, stderr_path=stderr, duration_seconds=0.1)],
    )

    assert not _stage_succeeded("probe", stage_result)


def test_post_patch_probe_uses_partial_success_rule(tmp_path, monkeypatch):
    import reproagent.main as main_module

    ok_stdout = tmp_path / 'ok.stdout'
    ok_stderr = tmp_path / 'ok.stderr'
    fail_stdout = tmp_path / 'fail.stdout'
    fail_stderr = tmp_path / 'fail.stderr'
    for path in [ok_stdout, ok_stderr, fail_stdout, fail_stderr]:
        path.write_text('', encoding='utf-8')

    probe_plan = CommandPlan(
        stage='probe',
        summary='post-patch probe',
        commands=['python train.py --help', 'python train.py --gpu 0'],
    )
    probe_results = [
        CommandResult(command='python train.py --help', exit_code=0, stdout_path=ok_stdout, stderr_path=ok_stderr, duration_seconds=0.1),
        CommandResult(command='python train.py --gpu 0', exit_code=-2, stdout_path=fail_stdout, stderr_path=fail_stderr, duration_seconds=0.0),
    ]

    monkeypatch.setattr(main_module, 'plan_probe', lambda state: probe_plan)
    monkeypatch.setattr(main_module, 'run_commands', lambda *args, **kwargs: probe_results)
    monkeypatch.setattr(main_module, 'save_state', lambda state: None)

    state = ReproState(
        task=ReproTask(paper_url='paper', repo_url='repo', workspace_dir=tmp_path, experiment_goal='Run bounded MNIST.'),
        repo_context=RepoContext(repo_path=tmp_path),
        environment=EnvironmentInfo(env_name='repro-test'),
    )

    assert _run_probe_once_after_patch(state)
    assert len(state.probe_attempts) == 1



def test_final_summary_without_experiment_results_does_not_call_llm(tmp_path, monkeypatch):
    def fail_final_review(state):
        raise AssertionError("LLM final review should not run without experiment results")

    import reproagent.main as main_module

    monkeypatch.setattr(main_module, "final_review", fail_final_review)
    state = ReproState(
        task=ReproTask(
            paper_url="paper",
            repo_url="repo",
            workspace_dir=tmp_path,
            experiment_goal="Run bounded MNIST and report loss.",
        ),
        planned_experiment=CommandPlan(
            stage="experiment",
            summary="Run MNIST.",
            commands=["python examples/odenet_mnist.py --gpu 0"],
            feasibility="blocked",
            needs_user_input=["Goal asks for a bounded run."],
        ),
    )

    summary = _final_summary(state)

    assert "Experiment commands were not executed" in summary
    assert "no reproduction metrics" in summary
    assert "Goal asks for a bounded run" in summary


def test_run_parser_reads_mirror_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--paper", "paper",
        "--repo", "repo",
        "--workspace", str(tmp_path),
        "--experiment-goal", "Run MNIST.",
        "--mirror-profile", "autodl",
        "--mirror-strict",
    ])

    assert args.mirror_profile == "autodl"
    assert args.mirror_strict
