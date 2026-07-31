from reproagent.env import build_backend_command, _env_name


def test_build_backend_command_wraps_conda_run():
    cmd = build_backend_command("repro_demo", "python3 --version", conda="/fake/conda")
    assert cmd == ["/fake/conda", "run", "-n", "repro_demo", "bash", "-c", "python3 --version"]


def test_env_name_sanitizes_task_id():
    assert _env_name("repro-20260724-abc123").startswith("repro_repro_20260724_abc123")



def test_find_conda_uses_env_var(tmp_path, monkeypatch):
    conda = tmp_path / "conda"
    conda.write_text("", encoding="utf-8")
    monkeypatch.setenv("REPROAGENT_CONDA_EXE", str(conda))

    from reproagent.env import find_conda

    assert find_conda() == str(conda)


def test_conda_setup_retries_transient_http_errors(tmp_path, monkeypatch):
    import subprocess
    from reproagent import env

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
    from reproagent import env

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
