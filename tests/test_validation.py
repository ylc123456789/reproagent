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


def test_validation_accepts_gpu_default_probe_evidence(tmp_path):
    from reproagent.models import CommandPlan, CommandResult, RepoContext, ReproState, ReproTask, StageResult
    from reproagent.validation import validate_experiment_plan

    probe_stdout = tmp_path / "probe.stdout"
    probe_stdout.write_text("--gpu GPU\nparser.add_argument('--gpu', type=int, default=0)\ndevice = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')", encoding="utf-8")
    probe_stderr = tmp_path / "probe.stderr"
    probe_stderr.write_text("", encoding="utf-8")
    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal="Run bounded GPU MNIST."),
        repo_context=RepoContext(repo_path=tmp_path),
    )
    state.probe_attempts.append(StageResult(
        stage="probe",
        attempt=1,
        plan=CommandPlan(stage="probe", summary="inspect GPU defaults", commands=[]),
        results=[CommandResult(command="grep gpu", exit_code=0, stdout_path=probe_stdout, stderr_path=probe_stderr, duration_seconds=0.1)],
    ))
    plan = CommandPlan(stage="experiment", summary="run bounded", commands=["python examples/odenet_mnist.py --nepochs 1"], feasibility="ready_to_run")

    validated = validate_experiment_plan(state, plan)

    assert not any("GPU execution" in item for item in validated.needs_user_input)



def test_validation_accepts_loss_logging_from_completed_coding_agent_patch(tmp_path):
    from reproagent.models import CodingAgentResult

    diff = tmp_path / "diff.patch"
    diff.write_text(
        "\n".join([
            "+    loss_meter = RunningAverageMeter()",
            "+    loss_meter.update(loss.item())",
            "+    logger.info(",
            "+        \"Loss {:.4f} | Train Acc {:.4f} | Test Acc {:.4f}\".format(",
            "+            loss_meter.avg, train_acc, val_acc",
            "+        )",
            "+    )",
        ]),
        encoding="utf-8",
    )
    state = _state_with_probe(
        tmp_path,
        "Run a bounded GPU experiment and report training loss.",
        "--nepochs NEPOCHS\n--gpu GPU\ncriterion = nn.CrossEntropyLoss()\nlogger.info('Train Acc ... Test Acc ...')\n",
    )
    state.coding_agent_results.append(CodingAgentResult(
        status="completed",
        summary="Added training loss logging per epoch.",
        changed_files=["examples/odenet_mnist.py"],
        diff_path=diff,
    ))
    plan = CommandPlan(
        stage="experiment",
        summary="Run bounded training",
        commands=["python examples/odenet_mnist.py --gpu 0 --nepochs 1"],
        assumptions=["CodingAgent added loss logging."],
        feasibility="ready_to_run",
    )

    validated = validate_experiment_plan(state, plan)

    assert not any("training loss" in item for item in validated.needs_user_input)
