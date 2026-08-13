"""Parse and execute the agent actions decided by the LLM.

run_commands, audit_env, and call_coding_agent handlers, plus the helpers
they need: JSON action parsing, file-read caching, and experiment
confirmation.  Imported by loop.py; kept separate so the loop stays focused
on state progression and budgeting.
"""

from __future__ import annotations

import json
import re
import subprocess

from ..integrations.codingagent import run_coding_agent_for_patch
from ..models import (
    AgentAction,
    AgentObservation,
    AgentState,
    CommandPlan,
    CommandResult,
    EnvironmentAudit,
)
from ..runtime.audit import audit_environment
from ..runtime.runner import _is_setup_command, run_commands


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


def _confirm_experiment(commands: list[str]) -> bool:
    """Ask the user to confirm experiment commands."""
    print("[reproagent] experiment commands:", flush=True)
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}", flush=True)
    try:
        answer = input("[reproagent] Run these experiment commands? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


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
    if state.task.setup_only:
        # Deterministic whitelist — does NOT rely on the LLM's stage_hint:
        # script/module execution is blocked whatever label the model used.
        blocked = [cmd for cmd in commands if not _is_setup_command(cmd)]
        if blocked:
            return AgentObservation(
                step=len(state.steps) + 1,
                action=action.action,
                stage_hint=action.stage_hint,
                error="setup_only task: command not allowed (provisioning and "
                      "inspection only): " + "; ".join(blocked),
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
        results = _mock_run_commands(commands, state)
    else:
        env_name = state.environment.env_name if state.environment else ""
        results = run_commands(
            commands,
            cwd=state.repo_context.repo_path if state.repo_context else state.task.workspace_dir,
            workspace=state.task.workspace_dir,
            stage="setup" if state.task.setup_only else (action.stage_hint or "agent"),
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
    try:
        audit = audit_environment(state)
    except Exception as exc:
        return AgentObservation(
            step=len(state.steps) + 1,
            action="audit_env",
            stage_hint="audit",
            audit=EnvironmentAudit(success=False, summary=str(exc)),
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
    if not state.task.allow_code_delegation:
        issues = action.coding_issues or [action.coding_goal or "unspecified"]
        return AgentObservation(
            step=len(state.steps) + 1,
            action="call_coding_agent",
            stage_hint="coding",
            error="blocked: code delegation is disabled by the caller — "
                  "the orchestrator routes a CodingAgent task and resumes this "
                  "session. Issues: " + "; ".join(issues),
        )
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


def _mock_run_commands(commands: list[str], state: AgentState) -> list[CommandResult]:
    """Run commands directly without conda for mock tests."""
    results: list[CommandResult] = []
    logs = state.task.workspace_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for i, cmd in enumerate(commands):
        out_path = logs / f"mock_{len(state.steps) + 1:02d}_{i + 1:02d}.stdout"
        err_path = logs / f"mock_{len(state.steps) + 1:02d}_{i + 1:02d}.stderr"
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30,
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
