from reproagent.report import _clean_text


def test_clean_text_removes_common_mojibake():
    text = "Smoke Test 鈥?Spiral ODE and paper鈥檚 metrics"

    cleaned = _clean_text(text)

    assert "鈥" not in cleaned
    assert "paper's" in cleaned


def test_clean_text_repairs_medium_run_mojibake_samples():
    text = "GPU鈥慳ccelerated ODE鈥慛et time鈥憇eries full鈥憇cale torch鈮?.3.0"

    cleaned = _clean_text(text)

    assert "鈥" not in cleaned
    assert "GPU-accelerated" in cleaned
    assert "ODE-Net" in cleaned
    assert "time-series" in cleaned
    assert "full-scale" in cleaned
    assert "torch>=.3.0" in cleaned


def test_clean_text_repairs_latin1_utf8_mojibake():
    text = "paperâ€™s GPUâ€“accelerated run"

    cleaned = _clean_text(text)

    assert cleaned == "paper’s GPU–accelerated run"


def test_write_result_includes_experiment_goal(tmp_path):
    from reproagent.models import ReproState, ReproTask
    from reproagent.report import write_result

    goal = "Run MNIST ODE-Net and report accuracy."
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal=goal)
    state = ReproState(task=task, status="completed")

    path = write_result(state)

    assert f"Experiment goal: {goal}" in path.read_text(encoding="utf-8")


def test_write_result_includes_planned_experiment(tmp_path):
    from reproagent.models import CommandPlan, ReproState, ReproTask
    from reproagent.report import write_result

    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal="Run bounded MNIST")
    state = ReproState(task=task, status="planned")
    state.planned_experiment = CommandPlan(
        stage="experiment",
        summary="Run one epoch",
        commands=["python examples/odenet_mnist.py --nepochs 1 --gpu 0"],
        feasibility="ready_to_run",
        expected_runtime="minutes",
    )

    path = write_result(state)
    text = path.read_text(encoding="utf-8")

    assert "## Planned Experiment" in text
    assert "--nepochs 1 --gpu 0" in text
    assert "Feasibility: `ready_to_run`" in text
