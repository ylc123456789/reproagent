from pathlib import Path

import pytest

from reproagent.integrations.codingagent import configured_codingagent_path, run_code_task, validate_codingagent_path


def _fake_codingagent_checkout(path: Path, marker: str = "external") -> Path:
    package = path / "src" / "coding_agent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
class CodeTaskSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Result:
    pass

def run_code_task(spec):
    result = Result()
    result.status = 'completed'
    result.summary = 'ran {marker}: ' + spec.task_goal
    result.changed_files = []
    result.diff_path = None
    result.verification_results = [type('Verify', (), {'command': command})() for command in spec.verify_commands]
    result.residual_risks = []
    return result
""".replace("{marker}", marker),
        encoding="utf-8",
    )
    return path


def test_validate_codingagent_path_accepts_src_layout(tmp_path):
    checkout = _fake_codingagent_checkout(tmp_path / "CodingAgent")

    resolved = validate_codingagent_path(checkout)

    assert resolved == checkout.resolve()


def test_invalid_codingagent_path_has_clear_error(tmp_path):
    with pytest.raises(ValueError, match="does not look like a CodingAgent checkout"):
        validate_codingagent_path(tmp_path)


def test_configured_path_priority_cli_env_config(tmp_path):
    cli_checkout = _fake_codingagent_checkout(tmp_path / "cli")
    env_checkout = _fake_codingagent_checkout(tmp_path / "env")
    config_checkout = _fake_codingagent_checkout(tmp_path / "config")
    config = tmp_path / "reproagent.yaml"
    config.write_text(f"agents:\n  codingagent_path: {config_checkout}\n", encoding="utf-8")

    resolved = configured_codingagent_path(cli_checkout, config, env={"CODINGAGENT_PATH": str(env_checkout)})

    assert resolved == cli_checkout.resolve()


def test_configured_path_uses_env_before_config(tmp_path):
    env_checkout = _fake_codingagent_checkout(tmp_path / "env")
    config_checkout = _fake_codingagent_checkout(tmp_path / "config")
    config = tmp_path / "reproagent.yaml"
    config.write_text(f"agents:\n  codingagent_path: {config_checkout}\n", encoding="utf-8")

    resolved = configured_codingagent_path(None, config, env={"CODINGAGENT_PATH": str(env_checkout)})

    assert resolved == env_checkout.resolve()


def test_config_relative_path_resolves_against_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    checkout = _fake_codingagent_checkout(config_dir / "agents" / "CodingAgent")
    config_dir.mkdir(exist_ok=True)
    config = config_dir / "reproagent.yaml"
    config.write_text("agents:\n  codingagent_path: agents/CodingAgent\n", encoding="utf-8")

    resolved = configured_codingagent_path(None, config, env={})

    assert resolved == checkout.resolve()


def test_run_code_task_imports_from_explicit_checkout(tmp_path):
    checkout = _fake_codingagent_checkout(tmp_path / "CodingAgent", marker="chosen")

    report = run_code_task(
        codingagent_path=checkout,
        repo_path=tmp_path / "repo",
        task_goal="patch loss logging",
        constraints=["keep behavior"],
        verify_commands=["python train.py --help"],
        max_steps=3,
        timeout_seconds=30,
        api_base="https://api.example.test/v1",
        api_key_env="API_KEY",
        model="model-x",
        output_dir=tmp_path / "out",
    )

    assert report.status == "completed"
    assert "ran chosen" in report.summary
    assert [result.command for result in report.verification_results] == ["python train.py --help"]
