from reproagent.audit import audit_environment
from reproagent.models import EnvironmentInfo, RepoContext, ReproState, ReproTask

def test_audit_detects_python_outside_expected_env(tmp_path, monkeypatch):
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path)
    state = ReproState(
        task=task,
        repo_context=RepoContext(
            repo_path=tmp_path,
            hardware_text="nvidia-smi:\nGPU 0: RTX 4060, VRAM total 8188 MiB, VRAM free 7000 MiB, driver 537.70, CUDA 12.2",
        ),
        environment=EnvironmentInfo(env_name="repro_demo"),
    )

    class Result:
        returncode = 0
        stdout = '{"sys_executable":"/home/cyl/miniconda3/bin/python","pip_version":"pip 1 from /home/cyl/miniconda3/lib/python3.14/site-packages/pip","torch":{"version":"2.13.0+cu130","file":"/home/cyl/miniconda3/lib/python3.14/site-packages/torch/__init__.py","cuda_compiled":"13.0","cuda_available":false,"device_count":0}}'
        stderr = ""

    monkeypatch.setattr("reproagent.audit.find_conda", lambda: "/fake/conda")
    monkeypatch.setattr("reproagent.audit.subprocess.run", lambda *args, **kwargs: Result())

    audit = audit_environment(state)

    assert not audit.success
    assert audit.summary == "Environment audit found issues."
    assert audit.has_warnings
    assert audit.requires_repair
    assert any("python executable is not inside expected conda env" in item for item in audit.details)
    assert any("GPU repair required" in item for item in audit.details)
    assert audit.stdout_path == tmp_path / "logs" / "environment_audit.stdout"

def test_audit_passes_when_python_and_torch_are_inside_env(tmp_path, monkeypatch):
    env_prefix = "/home/cyl/miniconda3/envs/repro_demo"
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path)
    state = ReproState(
        task=task,
        repo_context=RepoContext(repo_path=tmp_path),
        environment=EnvironmentInfo(env_name="repro_demo"),
    )

    class Result:
        returncode = 0
        stdout = '{"sys_executable":"' + env_prefix + '/bin/python","pip_version":"pip 1 from ' + env_prefix + '/lib/python3.10/site-packages/pip","torch":{"version":"2.1.0","file":"' + env_prefix + '/lib/python3.10/site-packages/torch/__init__.py","cuda_compiled":null,"cuda_available":false,"device_count":0}}'
        stderr = ""

    monkeypatch.setattr("reproagent.audit.find_conda", lambda: "/fake/conda")
    monkeypatch.setattr("reproagent.audit.subprocess.run", lambda *args, **kwargs: Result())

    audit = audit_environment(state)

    assert audit.success
    assert not audit.has_warnings
    assert not audit.requires_repair
    assert audit.summary == "Environment audit passed."

def test_audit_requires_repair_for_numpy_abi_warning(tmp_path, monkeypatch):
    env_prefix = "/home/cyl/miniconda3/envs/repro_demo"
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path)
    state = ReproState(
        task=task,
        repo_context=RepoContext(repo_path=tmp_path),
        environment=EnvironmentInfo(env_name="repro_demo"),
    )

    class Result:
        returncode = 0
        stdout = '{"sys_executable":"' + env_prefix + '/bin/python","pip_version":"pip 1 from ' + env_prefix + '/lib/python3.10/site-packages/pip","torch":{"version":"2.1.0+cu121","file":"' + env_prefix + '/lib/python3.10/site-packages/torch/__init__.py","cuda_compiled":"12.1","cuda_available":true,"device_count":1}}'
        stderr = "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6\n_ARRAY_API not found"

    monkeypatch.setattr("reproagent.audit.find_conda", lambda: "/fake/conda")
    monkeypatch.setattr("reproagent.audit.subprocess.run", lambda *args, **kwargs: Result())

    audit = audit_environment(state)

    assert audit.success
    assert audit.has_warnings
    assert audit.requires_repair
    assert audit.summary == "Environment audit requires repair."
    assert any("NumPy ABI" in item for item in audit.details)

def test_audit_passes_with_custom_conda_env_dir(tmp_path, monkeypatch):
    env_prefix = "/root/autodl-tmp/conda-envs/repro_demo"
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path)
    state = ReproState(
        task=task,
        repo_context=RepoContext(repo_path=tmp_path),
        environment=EnvironmentInfo(env_name="repro_demo"),
    )

    class Result:
        returncode = 0
        stdout = '{"sys_executable":"' + env_prefix + '/bin/python","sys_prefix":"' + env_prefix + '","pip_version":"pip 1 from ' + env_prefix + '/lib/python3.10/site-packages/pip","torch":{"version":"2.6.0+cu124","file":"' + env_prefix + '/lib/python3.10/site-packages/torch/__init__.py","cuda_compiled":"12.4","cuda_available":true,"device_count":1}}'
        stderr = ""

    monkeypatch.setattr("reproagent.audit.find_conda", lambda: "/fake/conda")
    monkeypatch.setattr("reproagent.audit.subprocess.run", lambda *args, **kwargs: Result())

    audit = audit_environment(state)

    assert audit.success
    assert audit.summary == "Environment audit passed."
    assert not any("Mismatch" in item for item in audit.details)


def test_audit_uses_sanitized_command_environment(tmp_path, monkeypatch):
    env_prefix = "/home/cyl/miniconda3/envs/repro_demo"
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path)
    state = ReproState(
        task=task,
        repo_context=RepoContext(repo_path=tmp_path),
        environment=EnvironmentInfo(env_name="repro_demo"),
    )
    captured = {}

    class Result:
        returncode = 0
        stdout = '{"sys_executable":"' + env_prefix + '/bin/python","sys_prefix":"' + env_prefix + '","pip_version":"pip 1 from ' + env_prefix + '/lib/python3.10/site-packages/pip"}'
        stderr = ""

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr("reproagent.audit.find_conda", lambda: "/fake/conda")
    monkeypatch.setenv("OMP_NUM_THREADS", "bad")
    monkeypatch.setattr("reproagent.audit.subprocess.run", fake_run)

    audit = audit_environment(state)

    assert audit.success
    assert captured["env"]["OMP_NUM_THREADS"] == "16"
    assert captured["env"]["TMPDIR"] == str(tmp_path / ".tmp")
    assert captured["env"]["PIP_CACHE_DIR"] == str(tmp_path / ".cache" / "pip")
