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



def test_experiment_stage_blocks_run_all():
    ok, reason = is_safe_command("python tests/run_all.py", stage="experiment")
    assert not ok
    assert "targeted small test" in reason


def test_experiment_stage_allows_targeted_test_file():
    ok, reason = is_safe_command("python tests/api_tests.py", stage="experiment")
    assert ok
    assert reason is None



def test_experiment_stage_blocks_bare_pytest():
    ok, reason = is_safe_command("python -m pytest", stage="experiment")
    assert not ok
    assert "whole pytest suite" in reason


def test_experiment_stage_allows_targeted_pytest_file():
    ok, reason = is_safe_command("python -m pytest tests/api_tests.py -v -x", stage="experiment")
    assert ok
    assert reason is None
