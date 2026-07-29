from reproagent.main import build_parser


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
