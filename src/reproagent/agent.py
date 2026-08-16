"""Top-level public API for running and resuming reproduction tasks.

main.py (CLI) and external orchestrators both go through this module:
run_task for fresh runs, resume_task for continuing a previous workspace.
"""
from __future__ import annotations

import json
from pathlib import Path

from .controller import run_controller
from .models import AgentState, ReproTask


def run_task(task: ReproTask) -> AgentState:
    """Run a reproduction task to completion in its workspace."""
    return run_controller(task)


def resume_task(
    workspace: Path,
    instruction: str,
    *,
    max_steps: int | None = None,
    model: str | None = None,
    api_base: str | None = None,
    api_key_env: str | None = None,
    mock_llm: bool = False,
    timeout: int | None = None,
) -> AgentState:
    """Resume a previous task in the same workspace.

    Loads state.json, rebuilds the task with the same task_id (so the conda
    environment is reused), folds the new instruction into the goal, and
    continues the agent loop with the previous steps preserved.
    """
    ws = Path(workspace)
    state_path = ws / "state.json"
    if not state_path.exists():
        raise SystemExit(f"No state.json found in {ws} — cannot resume.")

    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    orig_task = state_data.get("task", {})
    prev_summary = state_data.get("final_summary", "")

    task = ReproTask(
        paper_url=orig_task.get("paper_url", ""),
        repo_url=orig_task.get("repo_url", ""),
        workspace_dir=ws,
        repo_cache_dir=Path(orig_task["repo_cache_dir"]) if orig_task.get("repo_cache_dir") else None,
        task_id=orig_task.get("task_id", ""),  # same task_id → env reused
        timeout_seconds=timeout or orig_task.get("timeout_seconds", 1800),
        max_steps=max_steps or orig_task.get("max_steps", 30),
        mock_llm=mock_llm or orig_task.get("mock_llm", False),
        model=model or orig_task.get("model"),
        api_base=api_base or orig_task.get("api_base", "https://api.openai.com/v1"),
        api_key_env=api_key_env or orig_task.get("api_key_env", "OPENAI_API_KEY"),
        experiment_goal=_build_resume_goal(orig_task.get("experiment_goal", ""), instruction, prev_summary),
        enable_coding_agent=orig_task.get("enable_coding_agent", False),
        max_coding_agent_steps=orig_task.get("max_coding_agent_steps", 24),
        codingagent_path=Path(orig_task["codingagent_path"]) if orig_task.get("codingagent_path") else None,
        mirror_profile=orig_task.get("mirror_profile", "none"),
        mirror_strict=orig_task.get("mirror_strict", False),
        dataset_cache_dir=orig_task.get("dataset_cache_dir", ""),
        env_namespace=orig_task.get("env_namespace", ""),
        isolate_env=orig_task.get("isolate_env", False),
        parent_run=orig_task.get("parent_run"),
        copy_from=orig_task.get("copy_from", ""),
        external_repo_path=orig_task.get("external_repo_path", ""),
        setup_only=orig_task.get("setup_only", False),
        allow_code_delegation=orig_task.get("allow_code_delegation", True),
        env_name=orig_task.get("env_name", ""),
        input_artifacts=orig_task.get("input_artifacts", []),
        project_ref=orig_task.get("project_ref", ""),
        resource_root=orig_task.get("resource_root", ""),
        reuse_mode=orig_task.get("reuse_mode", "legacy"),
    )
    old_state = AgentState.model_validate(state_data)
    old_state.attempt_count = getattr(old_state, "attempt_count", 1) + 1
    return run_controller(task, resume_state=old_state)


def _build_resume_goal(original_goal: str, instruction: str, prev_summary: str) -> str:
    """Build the experiment goal for a resume run, incorporating previous results."""
    parts = [f"## Continuation of previous task\n\nNew instruction: {instruction}"]
    if original_goal:
        parts.append(f"\nOriginal goal: {original_goal}")
    if prev_summary:
        parts.append(f"\nPrevious results (summary): {prev_summary[:2000]}")
    return "\n".join(parts)
