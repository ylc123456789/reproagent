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
