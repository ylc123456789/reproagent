"""Small shared Pydantic models for reproagent."""
from __future__ import annotations
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

StageName = Literal["environment", "probe", "experiment"]
Feasibility = Literal["ready_to_run", "needs_config", "needs_patch", "blocked", "unsafe_or_too_expensive"]
MirrorProfile = Literal["none", "cn", "autodl"]
from pydantic import BaseModel, Field


def _task_id() -> str:
    """Create a short unique task identifier."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"repro-{stamp}-{uuid.uuid4().hex[:6]}"


class ReproTask(BaseModel):
    """Experiment-operator task configuration.

    A new task must declare exactly one repository source (see
    repository.context.setup_workspace): repo_url clones an isolated copy,
    copy_from copies a local worktree (uncommitted changes preserved), and
    external_repo_path operates on an existing repository in place.  All
    three empty is the resume case — the existing workspace/repo is reused.
    """
    paper_url: str = ""
    repo_url: str = ""
    workspace_dir: Path
    repo_cache_dir: Path | None = None
    task_id: str = Field(default_factory=_task_id)
    timeout_seconds: int = 3600
    max_steps: int = 30
    mock_llm: bool = False
    model: str | None = None
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    backend: Literal["conda"] = "conda"
    python_version: str = "3.10"
    experiment_goal: str = ""
    enable_coding_agent: bool = False
    max_coding_agent_steps: int = 24
    codingagent_path: Path | None = None
    config_path: Path | None = None
    mirror_profile: MirrorProfile = "none"
    mirror_strict: bool = False
    confirm_before_experiment: bool = False
    dataset_cache_dir: str = ""
    parent_run: dict | None = None
    env_namespace: str = ""
    isolate_env: bool = False
    copy_from: str = ""
    external_repo_path: str = ""
    setup_only: bool = False
    allow_code_delegation: bool = True
    # Explicit environment binding: a conda env NAME or absolute PREFIX.
    # When set, the environment is used unchanged for every command stage;
    # a missing environment is an error, never a substitute creation.
    # env_namespace remains the default creation mechanism when unset.
    env_name: str = ""
    # Structured upstream artifacts (prior measurements/files from other
    # modules). Rendered into prompts verbatim; optional.
    input_artifacts: list[dict] = Field(default_factory=list)
    # Milestone-2 resource management (additive; legacy stays the default).
    # project_ref is the orchestrated-mode slug source (§4.3); resource_root
    # hosts manifests/locks/conda envs when reuse_mode=content_addressed.
    project_ref: str = ""
    resource_root: str = ""
    reuse_mode: Literal["legacy", "content_addressed"] = "legacy"
    # GPU required by the task (spec collection: accelerator type identity).
    requires_gpu: bool = False


class RepoContext(BaseModel):
    """Cloned repository context."""
    repo_path: Path
    commit_hash: str | None = None
    file_tree: str = ""
    readme_text: str = ""
    hardware_text: str = ""
    paper_url: str = ""
    summary_path: Path | None = None


class ReproAgentVersion(BaseModel):
    """Reproagent code version used for a run."""
    source_path: Path
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    git_remote: str | None = None


class EnvironmentInfo(BaseModel):
    """Conda environment metadata."""
    backend: Literal["conda"] = "conda"
    env_name: str
    created: bool = False
    used_environment_yml: bool = False
    setup_command: str | None = None
    setup_stdout_path: Path | None = None
    setup_stderr_path: Path | None = None


class EnvironmentAudit(BaseModel):
    """Result of environment audit."""
    success: bool
    summary: str
    details: list[str] = Field(default_factory=list)
    has_warnings: bool = False
    requires_repair: bool = False
    stdout_path: Path | None = None
    stderr_path: Path | None = None


class CommandResult(BaseModel):
    """Result of a single shell command."""
    command: str
    exit_code: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    backend_command: list[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True if the command exited with code 0."""
        return self.exit_code == 0


class CodingAgentResult(BaseModel):
    """Result of a CodingAgent patch attempt."""
    status: str
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    diff_path: Path | None = None
    report_path: Path | None = None
    output_dir: Path | None = None
    environment_summary: str = ""
    verification_commands: list[str] = Field(default_factory=list)
    residual_risks: list[str] = Field(default_factory=list)


# ── Agent-loop types ──────────────────────────────────────────────

class AgentAction(BaseModel):
    """An action decided by the LLM."""
    thinking: str
    action: Literal["run_commands", "audit_env", "call_coding_agent", "finish"]
    stage_hint: str = ""
    commands: list[str] = Field(default_factory=list)
    coding_goal: str = ""
    coding_issues: list[str] = Field(default_factory=list)
    finish_status: str = ""
    finish_summary: str = ""
    finish_metrics: dict[str, Any] = Field(default_factory=dict)
    finish_parameters: dict[str, Any] = Field(default_factory=dict)
    finish_deviations: list[str] = Field(default_factory=list)
    evidence_files: list[str] = Field(default_factory=list)


class AgentObservation(BaseModel):
    """Result of executing one agent action."""
    step: int
    action: str
    stage_hint: str
    command_results: list[CommandResult] = Field(default_factory=list)
    audit: EnvironmentAudit | None = None
    coding_result: CodingAgentResult | None = None
    # Structured issues for blocked code-delegation exits — orchestrators
    # read this instead of parsing the text summary.
    coding_issues: list[str] = Field(default_factory=list)
    error: str = ""


class AgentState(BaseModel):
    """Full agent state, persisted to state.json."""
    task: ReproTask
    repo_context: RepoContext | None = None
    environment: EnvironmentInfo | None = None
    last_audit: EnvironmentAudit | None = None
    coding_results: list[CodingAgentResult] = Field(default_factory=list)
    steps: list[AgentObservation] = Field(default_factory=list)
    status: str = "running"
    final_summary: str = ""
    structured_result: dict[str, Any] = Field(default_factory=dict)
    result_path: Path | None = None
    file_cache: dict[str, str] = Field(default_factory=dict)
    produced_files: dict[str, Path] = Field(default_factory=dict)
    attempt_count: int = 1
    # Dataset-cache bridging: resolved hardcoded dataset roots + symlink status
    # (see dataset_cache.py). Rendered into every turn prompt.
    dataset_links: list[dict] = Field(default_factory=list)


# ── Legacy types (used by runner / coding / report) ───────────────

class CommandPlan(BaseModel):
    """Legacy command plan, kept for runner and CodingAgent compatibility."""
    stage: StageName
    summary: str
    commands: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    feasibility: Feasibility | None = None
    expected_runtime: str | None = None
    needs_user_input: list[str] = Field(default_factory=list)
    stop_reason: str | None = None


class StageResult(BaseModel):
    """Legacy stage result, kept for report compatibility."""
    stage: StageName
    attempt: int
    plan: CommandPlan
    results: list[CommandResult] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True if all commands in this stage succeeded."""
        return bool(self.results) and all(r.success for r in self.results)


class ReproState(BaseModel):
    """Legacy state model, kept for infrastructure module compatibility."""
    task: ReproTask
    status: str = "created"
    reproagent_version: ReproAgentVersion | None = None
    repo_context: RepoContext | None = None
    environment: EnvironmentInfo | None = None
    environment_audit: EnvironmentAudit | None = None
    environment_attempts: list[StageResult] = Field(default_factory=list)
    probe_attempts: list[StageResult] = Field(default_factory=list)
    planned_experiment: CommandPlan | None = None
    coding_agent_results: list[CodingAgentResult] = Field(default_factory=list)
    experiment_attempts: list[StageResult] = Field(default_factory=list)
    final_summary: str = ""
    result_path: Path | None = None
