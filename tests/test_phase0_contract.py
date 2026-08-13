"""Phase 0 compatibility locks for behavior-preserving refactors."""

from __future__ import annotations

import hashlib
import argparse
from pathlib import Path

import reproagent
from reproagent.main import build_parser
from reproagent.models import AgentState, EnvironmentInfo, RepoContext, ReproTask
from reproagent.prompts import SYSTEM_PROMPT, build_initial_context


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _subcommands(parser) -> dict:
    action = next(
        item for item in parser._actions
        if isinstance(item, argparse._SubParsersAction)
    )
    return action.choices


def _options(parser) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}


def test_public_package_contract() -> None:
    assert reproagent.__all__ == ["ReproTask", "ReproState", "CommandPlan", "CommandResult"]
    assert reproagent.__version__ == "0.1.0"


def test_cli_contract() -> None:
    commands = _subcommands(build_parser())
    assert set(commands) == {"run", "resume", "list", "status"}
    assert _options(commands["run"]) == {
        "-h", "--help", "--paper", "--repo", "--workspace", "--repo-cache-dir",
        "--mock-llm", "--model", "--api-base", "--api-key-env", "--backend",
        "--python-version", "--timeout", "--max-steps", "--experiment-goal",
        "--confirm-before-experiment", "--dataset-cache", "--enable-coding-agent",
        "--max-coding-agent-steps", "--codingagent-path", "--config",
        "--mirror-profile", "--mirror-strict", "--env-namespace", "--isolate-env",
    }


def test_prompt_contracts() -> None:
    task = ReproTask(
        paper_url="https://example.test/paper",
        repo_url="https://example.test/repo.git",
        workspace_dir=Path("/tmp/phase0"),
        experiment_goal="verify phase0",
        task_id="repro-phase0",
    )
    repo = RepoContext(
        repo_path=Path("/tmp/phase0/repo"), commit_hash="abc123",
        file_tree="README.md", readme_text="Phase 0 README", hardware_text="CPU test",
    )
    rendered = build_initial_context(task, repo, EnvironmentInfo(env_name="phase0", created=True))
    assert _sha256(SYSTEM_PROMPT) == "f79538efb77bf01656dcdbe2315506d08091cc8cd08cd1832fad226edfd39a2a"
    assert _sha256(rendered) == "2f22ea512be045e406479ae55b2bff027eae99af677539089714a6cedc07d8d3"


def test_persisted_model_field_contracts(tmp_path) -> None:
    assert list(ReproTask.model_fields) == [
        "paper_url", "repo_url", "workspace_dir", "repo_cache_dir", "task_id",
        "timeout_seconds", "max_steps", "mock_llm", "model", "api_base",
        "api_key_env", "backend", "python_version", "experiment_goal",
        "enable_coding_agent", "max_coding_agent_steps", "codingagent_path",
        "config_path", "mirror_profile", "mirror_strict",
        "confirm_before_experiment", "dataset_cache_dir", "parent_run",
        "env_namespace", "isolate_env",
        # experiment-operator contract v1 additions (deliberate schema change)
        "copy_from", "external_repo_path", "setup_only", "allow_code_delegation",
    ]
    assert list(AgentState.model_fields) == [
        "task", "repo_context", "environment", "last_audit", "coding_results",
        "steps", "status", "final_summary", "result_path", "file_cache",
        "produced_files", "attempt_count", "dataset_links",
    ]
    task = ReproTask(
        paper_url="paper", repo_url="repo", workspace_dir=tmp_path,
        experiment_goal="goal", task_id="repro-phase0",
    )
    assert ReproTask.model_validate_json(task.model_dump_json()) == task
