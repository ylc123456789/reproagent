"""Small shared Pydantic models for reproagent."""
from __future__ import annotations
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

def _task_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"repro-{stamp}-{uuid.uuid4().hex[:6]}"

class ReproTask(BaseModel):
    paper_url: str
    repo_url: str
    workspace_dir: Path
    task_id: str = Field(default_factory=_task_id)
    max_env_attempts: int = 3
    max_run_attempts: int = 3
    timeout_seconds: int = 1800
    mock_llm: bool = False
    model: str | None = None
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    backend: Literal["conda"] = "conda"
    python_version: str = "3.10"

class RepoContext(BaseModel):
    repo_path: Path
    commit_hash: str | None = None
    file_tree: str = ""
    readme_text: str = ""
    hardware_text: str = ""
    paper_url: str = ""
    summary_path: Path | None = None

class EnvironmentInfo(BaseModel):
    backend: Literal["conda"] = "conda"
    env_name: str
    created: bool = False
    used_environment_yml: bool = False
    setup_command: str | None = None
    setup_stdout_path: Path | None = None
    setup_stderr_path: Path | None = None

class EnvironmentAudit(BaseModel):
    success: bool
    summary: str
    details: list[str] = Field(default_factory=list)
    has_warnings: bool = False
    requires_repair: bool = False
    stdout_path: Path | None = None
    stderr_path: Path | None = None

class CommandPlan(BaseModel):
    stage: Literal["environment", "experiment"]
    summary: str
    commands: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    stop_reason: str | None = None

class CommandResult(BaseModel):
    command: str
    exit_code: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    backend_command: list[str] = Field(default_factory=list)
    @property
    def success(self) -> bool:
        return self.exit_code == 0

class StageResult(BaseModel):
    stage: Literal["environment", "experiment"]
    attempt: int
    plan: CommandPlan
    results: list[CommandResult] = Field(default_factory=list)
    @property
    def success(self) -> bool:
        return bool(self.results) and all(r.success for r in self.results)

class ReproState(BaseModel):
    task: ReproTask
    status: str = "created"
    repo_context: RepoContext | None = None
    environment: EnvironmentInfo | None = None
    environment_audit: EnvironmentAudit | None = None
    environment_attempts: list[StageResult] = Field(default_factory=list)
    experiment_attempts: list[StageResult] = Field(default_factory=list)
    final_summary: str = ""
    result_path: Path | None = None
