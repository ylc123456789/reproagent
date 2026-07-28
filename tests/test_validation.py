from reproagent.models import CommandPlan, CommandResult, RepoContext, ReproState, ReproTask, StageResult
from reproagent.validation import validate_experiment_plan


def _state_with_probe(tmp_path, goal, probe_text):
    stdout = tmp_path / "probe.stdout"
    stderr = tmp_path / "probe.stderr"
    stdout.write_text(probe_text, encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal=goal),
        repo_context=RepoContext(repo_path=tmp_path),
    )
    state.probe_attempts.append(StageResult(
        stage="probe",
        attempt=1,
        plan=CommandPlan(stage="probe", summary="inspect", commands=["python train.py --help"]),
        results=[CommandResult(command="python train.py --help", exit_code=0, stdout_path=stdout, stderr_path=stderr, duration_seconds=0.1)],
    ))
    return state


def test_validation_flags_loss_claim_without_probe_output_evidence(tmp_path):
    state = _state_with_probe(
        tmp_path,
        "Run a bounded GPU experiment and report training loss and test accuracy.",
        "--nepochs NEPOCHS\n--gpu GPU\ncriterion = nn.CrossEntropyLoss()\nlogger.info('Train Acc ... Test Acc ...')\n",
    )
    plan = CommandPlan(
        stage="experiment",
        summary="Run bounded training",
        commands=["python examples/odenet_mnist.py --gpu 0 --nepochs 5"],
        assumptions=["The script likely prints training loss based on typical MNIST loops."],
        feasibility="ready_to_run",
    )

    validated = validate_experiment_plan(state, plan)

    assert validated.feasibility == "needs_patch"
    assert any("training loss" in issue for issue in validated.needs_user_input)
    assert any("guess" in issue.lower() for issue in validated.needs_user_input)


def test_validation_accepts_explicit_budget_and_loss_logging_evidence(tmp_path):
    state = _state_with_probe(
        tmp_path,
        "Run a bounded GPU experiment and report training loss.",
        "--nepochs NEPOCHS\n--gpu GPU\nloss = criterion(logits, y)\nlogger.info('Loss %.4f', loss.item())\n",
    )
    plan = CommandPlan(
        stage="experiment",
        summary="Run bounded training",
        commands=["python train.py --gpu 0 --nepochs 2"],
        assumptions=["Probe shows logger.info includes Loss."],
        feasibility="ready_to_run",
    )

    validated = validate_experiment_plan(state, plan)

    assert validated == plan


def test_validation_flags_shell_logging_and_missing_budget(tmp_path):
    state = _state_with_probe(tmp_path, "Run a bounded GPU experiment.", "--gpu GPU\n")
    plan = CommandPlan(
        stage="experiment",
        summary="Run training",
        commands=["cd repo && python train.py 2>&1 | tee train.log"],
        assumptions=[],
        feasibility="ready_to_run",
    )

    validated = validate_experiment_plan(state, plan)

    assert validated.feasibility == "blocked"
    assert any("bounded" in issue for issue in validated.needs_user_input)
    assert any("tee" in issue for issue in validated.needs_user_input)
