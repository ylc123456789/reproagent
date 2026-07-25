from reproagent.runner import is_safe_command


def test_blocks_sudo():
    ok, reason = is_safe_command("sudo apt install x")
    assert not ok
    assert "sudo" in reason


def test_blocks_conda_activate():
    ok, reason = is_safe_command("conda activate x && python train.py")
    assert not ok
    assert "conda activate" in reason


def test_allows_python_version():
    ok, reason = is_safe_command("python3 --version")
    assert ok
    assert reason is None



def test_environment_stage_blocks_demo_script():
    ok, reason = is_safe_command("python examples/ode_demo.py", stage="environment")
    assert not ok
    assert "environment stage" in reason


def test_experiment_stage_allows_demo_script():
    ok, reason = is_safe_command("python examples/ode_demo.py", stage="experiment")
    assert ok
    assert reason is None
