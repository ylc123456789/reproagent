from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodeTaskSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_path: Path
    task_goal: str
    constraints: list[str] = Field(default_factory=list)
    verify_commands: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=list)
    max_iterations: int = 3
    max_steps: int = 12
    patch_repair_attempts: int = 2
    timeout_seconds: int = 900
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1"
    output_dir: Path | None = None

    @field_validator("repo_path")
    @classmethod
    def repo_path_must_exist(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"repo_path does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"repo_path is not a directory: {resolved}")
        return resolved

    @field_validator("max_iterations")
    @classmethod
    def max_iterations_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_iterations must be >= 1")
        return value

    @field_validator("max_steps")
    @classmethod
    def max_steps_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_steps must be >= 1")
        return value

    @field_validator("patch_repair_attempts")
    @classmethod
    def patch_repair_attempts_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("patch_repair_attempts must be >= 0")
        return value


class FileSnippet(BaseModel):
    path: str
    text: str
    truncated: bool = False


class RepoContext(BaseModel):
    repo_path: Path
    tree: list[str]
    snippets: list[FileSnippet]
    initial_diff: str = ""


class EditPlan(BaseModel):
    summary: str
    target_files: list[str]
    allowed_edit_type: Literal[
        "logging_only",
        "config_only",
        "bugfix",
        "new_file",
        "general",
    ] = "general"
    risks: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    needs_user_input: list[str] = Field(default_factory=list)
    feasibility: Literal["ready_to_edit", "needs_context", "blocked", "unsafe"]


class CommandResult(BaseModel):
    command: str
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class ControllerAction(BaseModel):
    action: Literal[
        "list_tree",
        "read_file",
        "search",
        "replace_text",
        "insert_before",
        "insert_after",
        "apply_patch",
        "run_command",
        "finish",
        "ask_user",
    ]
    reasoning: str
    path: str | None = None
    query: str | None = None
    command: str | None = None
    patch: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    anchor_text: str | None = None
    insert_text: str | None = None
    status: Literal["completed", "failed", "blocked", "needs_user_input"] | None = None
    summary: str | None = None
    residual_risks: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    step: int
    action: ControllerAction
    observation: str
    changed_files: list[str] = Field(default_factory=list)
    verification_results: list[CommandResult] = Field(default_factory=list)
    error: str | None = None


class PatchAttempt(BaseModel):
    iteration: int
    plan: EditPlan | None = None
    patch_text: str = ""
    applied: bool = False
    error: str | None = None
    verification_results: list[CommandResult] = Field(default_factory=list)


class PatchReport(BaseModel):
    status: Literal["completed", "failed", "blocked", "needs_user_input"]
    changed_files: list[str]
    diff_path: Path | None
    verification_results: list[CommandResult]
    summary: str
    residual_risks: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    task: CodeTaskSpec
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attempts: list[PatchAttempt] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    report: PatchReport | None = None
