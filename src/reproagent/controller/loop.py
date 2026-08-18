"""Agent-loop controller — the core of reproagent.

Replaces the old linear stage-workflow.  The LLM observes state, chooses an
action, the controller executes it and feeds the result back.  This repeats
until the LLM calls finish or the step budget is exhausted.

Action parsing and execution live in actions.py; prompt construction lives
in prompts.py.  This module only drives the state loop.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..context.policy import ContextPolicy
from ..llm import call_llm
from ..models import (
    AgentObservation,
    AgentState,
    EnvironmentInfo,
    RepoContext,
    ReproAgentVersion,
    ReproTask,
)
from ..report import write_agent_result
from ..repository.context import collect_context
from ..runtime.environment import ensure_environment
from ..session import write_session_card
from .actions import (
    _parse_action,
    _tool_audit_env,
    _tool_call_coding_agent,
    _tool_run_commands,
    _update_file_cache,
)
from .prompts import (
    SETUP_ONLY_DIRECTIVE,
    SYSTEM_PROMPT,
    build_initial_context,
    build_turn_prompt,
    convergence_guard,
)


def _experiment_run_count(state: AgentState) -> int:
    """Number of executed run_commands actions tagged as experiment."""
    return sum(
        1 for s in state.steps
        if s.action == "run_commands" and s.stage_hint == "experiment"
    )


def _log(message: str) -> None:
    """Print a prefixed workflow log message."""
    print(f"[reproagent] {message}", flush=True)


def _current_reproagent_version() -> ReproAgentVersion:
    """Collect git metadata for the running reproagent checkout."""
    source_path = Path(__file__).resolve().parents[3]
    return ReproAgentVersion(
        source_path=source_path,
        git_commit=_git_output(source_path, "rev-parse", "HEAD"),
        git_branch=_git_output(source_path, "branch", "--show-current"),
        git_dirty=_git_dirty(source_path),
        git_remote=_git_output(source_path, "config", "--get", "remote.origin.url"),
    )


def _git_output(repo_path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _git_dirty(repo_path: Path) -> bool | None:
    try:
        wt = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--quiet", "--"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
        ix = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--cached", "--quiet", "--"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        return None
    if wt.returncode not in {0, 1} or ix.returncode not in {0, 1}:
        return None
    return wt.returncode == 1 or ix.returncode == 1


def _format_version(version: ReproAgentVersion | None) -> str:
    if version is None:
        return "reproagent version: unknown"
    commit = (version.git_commit or "unknown")[:12]
    branch = version.git_branch or "unknown-branch"
    dirty = "dirty" if version.git_dirty else "clean" if version.git_dirty is False else "dirty-unknown"
    remote = f", remote={version.git_remote}" if version.git_remote else ""
    return f"reproagent version: {branch}@{commit} ({dirty}){remote}"


def run_controller(task: ReproTask, *, resume_state: AgentState | None = None) -> AgentState:
    """Run the agent loop until finish or step budget exhausted.

    When resume_state is provided, the existing workspace is reused
    (no re-clone, no re-create conda env) and new steps are appended.
    """
    task.workspace_dir.mkdir(parents=True, exist_ok=True)
    version = _current_reproagent_version()
    _log(f"workspace: {task.workspace_dir}")
    _log(_format_version(version))
    _prev_cache = os.environ.get("REPROAGENT_DATASET_CACHE")
    if task.dataset_cache_dir:
        os.environ["REPROAGENT_DATASET_CACHE"] = task.dataset_cache_dir

    # ── init or resume ──────────────────────────────────────
    if resume_state is not None:
        state = resume_state
        state.task = task
        state.status = "running"
        repo_context = state.repo_context
        environment = state.environment
    else:
        if task.mock_llm:
            repo_context = RepoContext(
                repo_path=task.workspace_dir / "repo",
                file_tree="README.md\nsetup.py",
                readme_text="Mock README",
                hardware_text="CPU only",
            )
            repo_context.repo_path.mkdir(parents=True, exist_ok=True)
            (repo_context.repo_path / "README.md").write_text("# Mock", encoding="utf-8")
            environment = EnvironmentInfo(env_name="mock_env", created=True)
        else:
            _log("collecting repository, README, hardware, and paper context")
            repo_context = collect_context(task)
            _log("preparing conda environment")
            environment = ensure_environment_for_controller(task, repo_context)
        state = AgentState(
            task=task,
            repo_context=repo_context,
            environment=environment,
        )

    # Bridge hardcoded dataset roots to the shared cache BEFORE the loop:
    # the LLM cannot do this itself (runner blocks ../ in commands) and
    # relative-path reasoning is where it fails. Best-effort, never fatal.
    if task.dataset_cache_dir:
        try:
            from ..runtime.dataset_cache import prepare_dataset_links
            # Shared mode operates on an external repo: symlinks may only be
            # created inside that repo (never beside it).  Isolated/copy modes
            # own the whole task workspace.
            allowed_root = repo_context.repo_path if task.external_repo_path else task.workspace_dir
            state.dataset_links = prepare_dataset_links(
                repo_path=repo_context.repo_path,
                workspace_dir=task.workspace_dir,
                cache_root=task.dataset_cache_dir,
                allowed_write_root=allowed_root,
            )
            created = sum(1 for r in state.dataset_links if r.get("link") == "created")
            _log(f"dataset cache: {len(state.dataset_links)} root(s) detected, "
                 f"{created} symlink(s) pre-created")
        except Exception as exc:
            _log(f"dataset cache preparation skipped: {exc}")

    policy = ContextPolicy.for_model(task.model)

    # ── loop ──────────────────────────────────────────────────
    for _step in range(task.max_steps):
        # build prompt fresh from current state
        if len(state.steps) == 0:
            user_prompt = build_initial_context(task, repo_context, environment, policy,
                                                dataset_links=state.dataset_links)
        else:
            user_prompt = build_turn_prompt(state, policy)
        guard = convergence_guard(_experiment_run_count(state))
        if guard:
            user_prompt += guard
        if task.setup_only:
            user_prompt += SETUP_ONLY_DIRECTIVE

        try:
            raw = call_llm(task, SYSTEM_PROMPT, user_prompt,
                           trace_label=f"step_{len(state.steps) + 1:02d}")
        except Exception as exc:
            _log(f"LLM call failed: {exc}")
            state.steps.append(AgentObservation(
                step=len(state.steps) + 1, action="llm_error", stage_hint="error",
                error=f"LLM API call failed: {exc}",
            ))
            continue

        action = _parse_action(raw)
        if action is None:
            state.steps.append(AgentObservation(
                step=len(state.steps) + 1, action="parse_error", stage_hint="error",
                error="Could not parse JSON action. Return a valid JSON object with an 'action' field.",
            ))
            continue

        _log(f"step {len(state.steps) + 1}: {action.action} — {action.thinking[:120]}")

        # execute
        observation: AgentObservation
        if action.action == "run_commands":
            observation = _tool_run_commands(action, state)
        elif action.action == "audit_env":
            observation = _tool_audit_env(state)
        elif action.action == "call_coding_agent":
            observation = _tool_call_coding_agent(action, state)
            if not state.task.allow_code_delegation:
                # Structured exit: the orchestrator routes a CodingAgent task
                # against this repo and resumes this session afterwards.
                # The issues list is persisted on the observation
                # (observation.coding_issues) for stable programmatic reads.
                state.status = "blocked"
                issues = observation.coding_issues
                state.final_summary = (
                    "Code changes are required but code delegation is disabled "
                    "by the caller.\nCoding issues:\n"
                    + "\n".join(f"- {issue}" for issue in issues)
                )
                state.steps.append(observation)
                _log(f"finish: {state.status} (code delegation disabled)")
                break
        else:  # finish
            state.status = action.finish_status or "completed"
            state.final_summary = action.finish_summary
            if task.setup_only:
                # Deterministic gate: a setup-only task may only complete
                # after a successful environment audit.
                if state.last_audit is None or not state.last_audit.success:
                    state.status = "failed"
                    state.final_summary += (
                        "\n\nSetup-only completion requires a successful "
                        "audit_env; the audit is missing or failed."
                    )
                state.final_summary += _provisioning_summary(state)
            _log(f"finish: {state.status}")
            break

        # track file reads for context cache
        _update_file_cache(state, observation)
        state.steps.append(observation)

    else:
        _log("step budget exhausted — forcing finish")
        if state.status == "running":
            state.status = "completed_with_failures"
            state.final_summary = "Step budget exhausted before explicit finish."
            # one last call to generate a final summary
            user_prompt = build_turn_prompt(state, policy)
            user_prompt += "\n\nStep budget exhausted. The run ends here. Write a brief final summary."
            final_raw = call_llm(task, SYSTEM_PROMPT, user_prompt, trace_label="force_finish")
            state.final_summary = final_raw[:5000]
            if task.setup_only:
                state.final_summary += _provisioning_summary(state)

    # ── write report ──────────────────────────────────────────
    result_path = write_agent_result(state, version)
    state.result_path = result_path
    _log(f"result: {result_path}")
    try:
        card_path = write_session_card(state)
        _log(f"session card: {card_path}")
    except Exception as exc:
        _log(f"session card skipped: {exc}")
    # restore env var that was overridden for this task
    if _prev_cache is None:
        os.environ.pop("REPROAGENT_DATASET_CACHE", None)
    else:
        os.environ["REPROAGENT_DATASET_CACHE"] = _prev_cache
    return state


def ensure_environment_for_controller(task: ReproTask, repo_context: RepoContext) -> EnvironmentInfo:
    """Create or reuse the conda env for the resolved repository.

    The environment is created with the RESOLVED repo path as cwd — never a
    rebuilt workspace/repo path.  In shared mode that is the external repo,
    where environment.yml lives and setup commands must run.
    """
    env_state = AgentState(task=task, repo_context=repo_context)
    return ensure_environment(env_state)  # type: ignore[arg-type]


def _provisioning_summary(state: AgentState) -> str:
    """Deterministic environment summary appended to setup_only reports."""
    lines = ["\n\n## Environment Provisioning"]
    env = state.environment
    if env is None:
        return "\n".join(lines + ["- No environment prepared."])
    lines.append(f"- Conda env: {env.env_name} ({'created this run' if env.created else 'reused'})")
    audit = state.last_audit
    if audit is None:
        lines.append("- Audit: not run.")
    else:
        lines.append(f"- Audit: {'PASSED' if audit.success else 'FAILED'} — {audit.summary}")
        lines.extend(f"  - {detail}" for detail in (audit.details or []))
    return "\n".join(lines)
