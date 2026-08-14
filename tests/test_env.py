from reproagent.runtime.environment import build_backend_command, _env_name


def test_build_backend_command_wraps_conda_run():
    cmd = build_backend_command("repro_demo", "python3 --version", conda="/fake/conda")
    assert cmd == ["/fake/conda", "run", "--no-capture-output", "-n", "repro_demo", "bash", "-o", "pipefail", "-c", "python3 --version"]


def test_env_name_sanitizes_task_id():
    assert _env_name("repro-20260724-abc123").startswith("repro_repro_20260724_abc123")



def test_find_conda_uses_env_var(tmp_path, monkeypatch):
    conda = tmp_path / "conda"
    conda.write_text("", encoding="utf-8")
    monkeypatch.setenv("REPROAGENT_CONDA_EXE", str(conda))

    from reproagent.runtime.environment import find_conda

    assert find_conda() == str(conda)


def test_conda_setup_retries_transient_http_errors(tmp_path, monkeypatch):
    import subprocess
    from reproagent.runtime import environment as env

    calls = []

    def fake_run(cmd, cwd, text, capture_output, timeout):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, "", "CondaHTTPError: HTTP 502 BAD GATEWAY")
        return subprocess.CompletedProcess(cmd, 0, "created", "")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env.time, "sleep", lambda seconds: None)

    result = env._run_conda_setup_with_retries(
        ["conda", "create"],
        cwd=tmp_path,
        timeout=30,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.returncode == 0
    assert len(calls) == 2
    assert "attempt 1/3" in (tmp_path / "stderr.log").read_text(encoding="utf-8")
    assert "attempt 2/3" in (tmp_path / "stdout.log").read_text(encoding="utf-8")


def test_conda_setup_does_not_retry_non_transient_errors(tmp_path, monkeypatch):
    import subprocess
    from reproagent.runtime import environment as env

    calls = []

    def fake_run(cmd, cwd, text, capture_output, timeout):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "PackagesNotFoundError: python=9.99")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    result = env._run_conda_setup_with_retries(
        ["conda", "create"],
        cwd=tmp_path,
        timeout=30,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.returncode == 1
    assert len(calls) == 1


# ── pipefail tests ──────────────────────────────────────────────

def test_pipefail_false_pipe_returns_nonzero(tmp_path):
    """false | cat must return non-zero with pipefail enabled."""
    import subprocess

    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", "false | cat"],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode != 0, f"expected non-zero, got {result.returncode}"


def test_pipefail_success_pipe_returns_zero(tmp_path):
    """true | cat must return zero with pipefail enabled."""
    import subprocess

    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", "true | cat"],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0


def test_pipefail_non_pipeline_preserves_exit_code(tmp_path):
    """A command without a pipeline must keep its original exit code."""
    import subprocess

    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", "exit 42"],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 42


# ── explicit environment binding (P4 contract) ────────────────────

def _fake_conda_env_list(envs: list[str]):
    import json as _json
    import subprocess

    def fake_run(cmd, text, capture_output, timeout):
        if cmd[:3] == ["/fake/conda", "env", "list"]:
            return subprocess.CompletedProcess(cmd, 0, _json.dumps({"envs": envs}), "")
        raise AssertionError(f"unexpected command: {cmd}")

    return fake_run


def _env_state(tmp_path, **task_kwargs):
    from reproagent.models import RepoContext, ReproState, ReproTask

    return ReproState(
        task=ReproTask(workspace_dir=tmp_path, **task_kwargs),
        repo_context=RepoContext(repo_path=tmp_path / "repo"),
    )


def test_ensure_environment_binds_explicit_name(tmp_path, monkeypatch):
    """env_name names an existing conda env → bound unchanged, no creation."""
    from reproagent.runtime import environment as env_module

    monkeypatch.setattr(env_module, "find_conda", lambda: "/fake/conda")
    monkeypatch.setattr(
        env_module.subprocess, "run",
        _fake_conda_env_list(["/opt/conda/envs/target_env"]),
    )

    info = env_module.ensure_environment(_env_state(tmp_path, env_name="target_env"))

    assert info.env_name == "target_env"
    assert info.created is False


def test_ensure_environment_binds_explicit_prefix(tmp_path, monkeypatch):
    """An absolute env_name is matched as a conda PREFIX."""
    from reproagent.runtime import environment as env_module

    monkeypatch.setattr(env_module, "find_conda", lambda: "/fake/conda")
    monkeypatch.setattr(
        env_module.subprocess, "run",
        _fake_conda_env_list(["/opt/conda/envs/target_env"]),
    )

    info = env_module.ensure_environment(
        _env_state(tmp_path, env_name="/opt/conda/envs/target_env")
    )

    assert info.env_name == "/opt/conda/envs/target_env"
    assert info.created is False


def test_ensure_environment_fails_on_missing_explicit_env(tmp_path, monkeypatch):
    """An unresolvable explicit env fails loudly — never a substitute creation."""
    import pytest
    from reproagent.runtime import environment as env_module

    monkeypatch.setattr(env_module, "find_conda", lambda: "/fake/conda")
    monkeypatch.setattr(env_module.subprocess, "run", _fake_conda_env_list([]))

    with pytest.raises(RuntimeError, match="Explicit environment 'ghost_env' was not found"):
        env_module.ensure_environment(_env_state(tmp_path, env_name="ghost_env"))
