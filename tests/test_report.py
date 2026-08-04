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

    assert cleaned == "paper's GPU-accelerated run"


def test_write_result_includes_experiment_goal(tmp_path):
    from reproagent.models import ReproState, ReproTask
    from reproagent.report import write_result

    goal = "Run MNIST ODE-Net and report accuracy."
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal=goal)
    state = ReproState(task=task, status="completed")

    path = write_result(state)

    assert f"Experiment goal: {goal}" in path.read_text(encoding="utf-8")


def test_write_result_includes_codingagent_path(tmp_path):
    from reproagent.models import ReproState, ReproTask
    from reproagent.report import write_result

    task = ReproTask(
        paper_url="paper",
        repo_url="repo",
        workspace_dir=tmp_path,
        experiment_goal="Run MNIST.",
        codingagent_path=tmp_path / "CodingAgent",
    )
    state = ReproState(task=task, status="completed")

    path = write_result(state)

    assert f"CodingAgent path: `{tmp_path / 'CodingAgent'}`" in path.read_text(encoding="utf-8")


def test_write_result_includes_reproagent_version(tmp_path):
    from reproagent.models import ReproAgentVersion, ReproState, ReproTask
    from reproagent.report import write_result

    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal="Run MNIST.")
    state = ReproState(
        task=task,
        status="completed",
        reproagent_version=ReproAgentVersion(
            source_path=tmp_path / "reproagent",
            git_remote="https://github.com/ylc123456789/reproagent.git",
            git_branch="main",
            git_commit="abc123def456",
            git_dirty=False,
        ),
    )

    path = write_result(state)
    text = path.read_text(encoding="utf-8")

    assert "## ReproAgent Version" in text
    assert "Git branch: `main`" in text
    assert "Git commit: `abc123def456`" in text
    assert "Git dirty: `False`" in text
