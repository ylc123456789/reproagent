"""Agent-loop controller — the core of reproagent.

Replaces the old linear stage-workflow.  The LLM observes state, chooses an
action, the controller executes it and feeds the result back.  This repeats
until the LLM calls finish or the step budget is exhausted.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .audit import audit_environment
from .coding import run_coding_agent_for_patch
from .context import collect_context
from .env import ensure_environment, find_conda
from .integrations.codingagent import configured_codingagent_path
from .llm import SYSTEM_PROMPT, build_initial_context, build_turn_prompt, call_llm
from .models import (
    AgentAction,
    AgentObservation,
    AgentState,
    CommandPlan,
    ContextPolicy,
    EnvironmentInfo,
    RepoContext,
    ReproAgentVersion,
    ReproTask,
)
from .report import write_agent_result
from .runner import run_commands


# ── helpers ───────────────────────────────────────────────────────

def _log(message: str) -> None:
    print(f"[reproagent] {message}", flush=True)


def _confirm_experiment(commands: list[str]) -> bool:
    """Ask the user to confirm experiment commands."""
    _log("experiment commands:")
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}", flush=True)
    try:
        answer = input("[reproagent] Run these experiment commands? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _current_reproagent_version() -> ReproAgentVersion:
    source_path = Path(__file__).resolve().parents[2]
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


def _parse_action(text: str) -> AgentAction | None:
    """Extract a valid JSON action from LLM response text. Returns None on parse failure."""
    match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    action_type = data.get("action", "")
    if action_type not in {"run_commands", "audit_env", "call_coding_agent", "finish"}:
        return None
    return AgentAction(
        thinking=str(data.get("thinking", "")),
        action=action_type,
        stage_hint=str(data.get("stage_hint", "")),
        commands=[str(c) for c in data.get("commands", [])],
        coding_goal=str(data.get("coding_goal", "")),
        coding_issues=[str(i) for i in data.get("coding_issues", [])],
        finish_status=str(data.get("finish_status", "")),
        finish_summary=str(data.get("finish_summary", "")),
    )


def _update_file_cache(state: AgentState, observation: AgentObservation) -> None:
    """Cache file reads for context reuse — avoid repeating full file dumps."""
    for r in observation.command_results:
        cmd = r.command.strip()
        # match common file-reading patterns: cat/head/tail/sed on a specific file
        m = re.match(r"(?:cat|head(?:\s+-n\s+\d+)?|tail(?:\s+-n\s+\d+)?|sed\s+[^ ]+|grep(?:\s+[^ ]+)*)\s+(.+)", cmd)
        if not m:
            continue
        path = m.group(1).strip().strip("'\"")
        # only cache repo-relative paths
        if not path.startswith("/"):
            full = (state.repo_context.repo_path if state.repo_context else state.task.workspace_dir) / path
            if full.exists():
                state.file_cache[path] = full.read_text(encoding="utf-8", errors="replace")


# ── tools ─────────────────────────────────────────────────────────

def _tool_run_commands(action: AgentAction, state: AgentState) -> AgentObservation:
    """Execute shell commands via the runner, respecting safety and confirm-before-experiment."""
    commands = action.commands
    if not commands:
        return AgentObservation(
            step=len(state.steps) + 1,
            action=action.action,
            stage_hint=action.stage_hint,
            error="no commands provided",
        )
    if (state.task.confirm_before_experiment and action.stage_hint == "experiment"
            and not _confirm_experiment(commands)):
        return AgentObservation(
            step=len(state.steps) + 1,
            action=action.action,
            stage_hint=action.stage_hint,
            error="experiment cancelled by user before command execution",
        )
    if state.task.mock_llm:
        # In mock mode, skip conda wrapping — run commands directly
        from .models import CommandResult
        results = _mock_run_commands(commands, state)
    else:
        env_name = state.environment.env_name if state.environment else ""
        results = run_commands(
            commands,
            cwd=state.repo_context.repo_path if state.repo_context else state.task.workspace_dir,
            workspace=state.task.workspace_dir,
            stage=action.stage_hint or "agent",
            attempt=len(state.steps) + 1,
            timeout=state.task.timeout_seconds,
            env_name=env_name,
        )
    return AgentObservation(
        step=len(state.steps) + 1,
        action=action.action,
        stage_hint=action.stage_hint,
        command_results=results,
    )


def _tool_audit_env(state: AgentState) -> AgentObservation:
    """Run environment audit: check Python, pip, torch/tf/jax, GPU availability."""
    from .models import EnvironmentAudit as EA  # noqa

    try:
        audit = audit_environment(state)
    except Exception as exc:
        return AgentObservation(
            step=len(state.steps) + 1,
            action="audit_env",
            stage_hint="audit",
            audit=EA(success=False, summary=str(exc)),
        )
    state.last_audit = audit
    return AgentObservation(
        step=len(state.steps) + 1,
        action="audit_env",
        stage_hint="audit",
        audit=audit,
    )


def _tool_call_coding_agent(action: AgentAction, state: AgentState) -> AgentObservation:
    """Delegate a code modification to CodingAgent and collect the result."""
    if not state.task.enable_coding_agent:
        return AgentObservation(
            step=len(state.steps) + 1,
            action="call_coding_agent",
            stage_hint="coding",
            error="CodingAgent is not enabled (use --enable-coding-agent)",
        )
    help_cmds: list[str] = []
    for step in state.steps:
        for r in step.command_results:
            if r.exit_code == 0 and "--help" in r.command:
                help_cmds.append(r.command)
    plan = CommandPlan(
        stage="experiment",
        summary=action.coding_goal or "patch",
        commands=help_cmds[:3],
        needs_user_input=action.coding_issues,
    )
    try:
        result = run_coding_agent_for_patch(state, plan)
    except Exception as exc:
        return AgentObservation(
            step=len(state.steps) + 1,
            action="call_coding_agent",
            stage_hint="coding",
            error=str(exc),
        )
    state.coding_results.append(result)
    return AgentObservation(
        step=len(state.steps) + 1,
        action="call_coding_agent",
        stage_hint="coding",
        coding_result=result,
    )


def _mock_run_commands(commands: list[str], state: AgentState):
    """Run commands directly without conda for mock tests."""
    import subprocess as sp
    from .models import CommandResult
    results = []
    logs = state.task.workspace_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for i, cmd in enumerate(commands):
        out_path = logs / f"mock_{len(state.steps) + 1:02d}_{i + 1:02d}.stdout"
        err_path = logs / f"mock_{len(state.steps) + 1:02d}_{i + 1:02d}.stderr"
        try:
            r = sp.run(cmd, shell=True, capture_output=True, text=True, timeout=30,
                       cwd=str(state.repo_context.repo_path) if state.repo_context else None)
            out_path.write_text(r.stdout or "", encoding="utf-8", errors="replace")
            err_path.write_text(r.stderr or "", encoding="utf-8", errors="replace")
            results.append(CommandResult(
                command=cmd, exit_code=r.returncode,
                stdout_path=out_path, stderr_path=err_path, duration_seconds=0.1,
                backend_command=[cmd],
            ))
        except Exception as exc:
            err_path.write_text(str(exc), encoding="utf-8")
            results.append(CommandResult(
                command=cmd, exit_code=-1,
                stdout_path=out_path, stderr_path=err_path, duration_seconds=0.1,
                backend_command=[cmd],
            ))
    return results


# ── main loop ─────────────────────────────────────────────────────

def run_controller(task: ReproTask) -> AgentState:
    """Run the agent loop until finish or step budget exhausted."""
    task.workspace_dir.mkdir(parents=True, exist_ok=True)
    version = _current_reproagent_version()
    _log(f"workspace: {task.workspace_dir}")
    _log(_format_version(version))
    if task.dataset_cache_dir:
        import os as _os
        _os.environ["REPROAGENT_DATASET_CACHE"] = task.dataset_cache_dir

    # ── init ──────────────────────────────────────────────────
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
        environment = ensure_environment_for_controller(task)

    state = AgentState(
        task=task,
        repo_context=repo_context,
        environment=environment,
    )
    policy = ContextPolicy.for_model(task.model)

    # ── loop ──────────────────────────────────────────────────
    for _step in range(task.max_steps):
        # build prompt fresh from current state
        if len(state.steps) == 0:
            user_prompt = build_initial_context(task, repo_context, environment, policy)
        else:
            user_prompt = build_turn_prompt(state, policy)

        raw = call_llm(task, SYSTEM_PROMPT, user_prompt,
                       trace_label=f"step_{len(state.steps) + 1:02d}")

        action = _parse_action(raw)
        if action is None:
            # Add a fake step so the next turn's prompt reflects the error
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
        else:  # finish
            state.status = action.finish_status or "completed"
            state.final_summary = action.finish_summary
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

    # ── write report ──────────────────────────────────────────
    result_path = write_agent_result(state, version)
    state.result_path = result_path
    _log(f"result: {result_path}")
    return state


def ensure_environment_for_controller(task: ReproTask):
    """Create or reuse conda env, returning EnvironmentInfo."""
    env_state = AgentState(
        task=task,
        repo_context=RepoContext(repo_path=task.workspace_dir / "repo"),
    )
    return ensure_environment(env_state)  # type: ignore[arg-type]
