"""Define shared data models for tasks, actions, state, and reports."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodeTaskSpec(BaseModel):
    """Configuration for one coding agent run."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace_path: Path
    task_goal: str
    constraints: list[str] = Field(default_factory=list)
    verify_commands: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=list)
    max_steps: int = 24
    max_extra_steps_after_progress: int = 8
    patch_repair_attempts: int = 2
    timeout_seconds: int = 900
    max_context_tokens: int | None = None
    model_context_window_tokens: int | None = None
    context_margin_ratio: float = 0.20
    context_output_reserve_tokens: int = 16_384
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1"
    output_dir: Path

    @field_validator("workspace_path")
    @classmethod
    def _workspace_path_setup(cls, value: Path) -> Path:
        """Resolve the workspace path, creating it if it does not exist."""
        resolved = value.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @field_validator("max_steps")
    @classmethod
    def max_steps_positive(cls, value: int) -> int:
        """Validate that the base step budget is positive."""
        if value < 1:
            raise ValueError("max_steps must be >= 1")
        return value

    @field_validator("max_extra_steps_after_progress")
    @classmethod
    def max_extra_steps_after_progress_non_negative(cls, value: int) -> int:
        """Validate the grace step budget."""
        if value < 0:
            raise ValueError("max_extra_steps_after_progress must be >= 0")
        return value

    @field_validator("max_context_tokens", "model_context_window_tokens")
    @classmethod
    def optional_token_limits_positive(cls, value: int | None) -> int | None:
        """Validate optional context token limits."""
        if value is not None and value < 8_000:
            raise ValueError("context token limits must be >= 8000")
        return value

    @field_validator("context_margin_ratio")
    @classmethod
    def context_margin_ratio_valid(cls, value: float) -> float:
        """Validate the context margin fraction."""
        if value < 0 or value >= 0.5:
            raise ValueError("context_margin_ratio must be >= 0 and < 0.5")
        return value

    @field_validator("context_output_reserve_tokens")
    @classmethod
    def context_output_reserve_tokens_positive(cls, value: int) -> int:
        """Validate output token reserve."""
        if value < 1_000:
            raise ValueError("context_output_reserve_tokens must be >= 1000")
        return value

    @field_validator("patch_repair_attempts")
    @classmethod
    def patch_repair_attempts_non_negative(cls, value: int) -> int:
        """Validate patch repair attempt count."""
        if value < 0:
            raise ValueError("patch_repair_attempts must be >= 0")
        return value


class FileSnippet(BaseModel):
    """Small text excerpt from a repository file."""
    path: str
    text: str
    truncated: bool = False


class RepoContext(BaseModel):
    """Snapshot of repository tree, snippets, and initial diff."""
    workspace_path: Path
    tree: list[str]
    snippets: list[FileSnippet]
    initial_diff: str = ""


class CommandResult(BaseModel):
    """Result metadata for one verification command."""
    command: str
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        """Return whether the command completed successfully."""
        return self.returncode == 0 and not self.timed_out


class ControllerAction(BaseModel):
    """One model-selected controller action."""
    action: Literal[
        "list_tree",
        "read_file",
        "search",
        "replace_text",
        "insert_before",
        "insert_after",
        "apply_patch",
        "write_file",
        "run_command",
        "finish",
        "ask_user",
    ]
    reasoning: str = ""
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    query: str | None = None
    command: str | None = None
    patch: str | None = None
    content: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    anchor_text: str | None = None
    insert_text: str | None = None
    occurrence_index: int | None = None
    status: Literal["completed", "failed", "blocked", "needs_user_input"] | None = None
    summary: str | None = None
    residual_risks: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    """Recorded outcome of one controller step."""
    step: int
    action: ControllerAction
    observation: str
    changed_files: list[str] = Field(default_factory=list)
    verification_results: list[CommandResult] = Field(default_factory=list)
    error: str | None = None


class PatchReport(BaseModel):
    """Final user-facing outcome of a coding task."""
    status: Literal["completed", "failed", "blocked", "needs_user_input"]
    changed_files: list[str]
    diff_path: Path | None
    verification_results: list[CommandResult]
    summary: str
    residual_risks: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Persisted state for an agent run."""
    task: CodeTaskSpec
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    steps: list[StepRecord] = Field(default_factory=list)
    report: PatchReport | None = None
