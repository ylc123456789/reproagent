from reproagent.main import _final_summary, build_parser
from reproagent.models import CodingAgentResult, CommandPlan, ReproState, ReproTask


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
