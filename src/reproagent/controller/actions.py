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

from pydantic import ValidationError

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
from ..runtime.runner import analyze_command, _is_setup_command, run_commands


def _not_setup_allowed(commands: list[str]) -> list[str]:
    """Commands outside the deterministic provisioning/inspection whitelist."""
    return [cmd for cmd in commands if not _is_setup_command(cmd)]


def _is_env_mutating_command(command: str) -> bool:
    """Whether the command changes installed packages (invalidates the audit)."""
    return analyze_command(command).mutates_environment


def _parse_action(text: str) -> AgentAction | None:
    """Extract a valid JSON action from LLM response text. Returns None on parse failure."""
    data = None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "action" in candidate:
            data = candidate
            break
    if data is None:
        return None
    try:
        return AgentAction.model_validate(data)
    except ValidationError:
        return None


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
        blocked = _not_setup_allowed(commands)
        if blocked:
            return AgentObservation(
                step=len(state.steps) + 1,
                action=action.action,
                stage_hint=action.stage_hint,
                error="setup_only task: command not allowed (provisioning and "
                      "inspection only): " + "; ".join(blocked),
            )
    elif state.last_audit is None or not state.last_audit.success:
        # Certification gate: a successful audit_env must precede experiment
        # commands. Deterministic (whitelist-based) — a training command is
        # refused whatever stage label the model used; dependency repair
        # (pip) and inspection stay available for the bounded repair loop.
        blocked = _not_setup_allowed(commands)
        if blocked:
            return AgentObservation(
                step=len(state.steps) + 1,
                action=action.action,
                stage_hint=action.stage_hint,
                error="environment is not yet certified: a successful audit_env "
                      "is required before experiment commands. Install or repair "
                      "dependencies, run audit_env, then retry. Blocked: "
                      + "; ".join(blocked),
            )
    # Mutation policy: an action that changes installed packages must not
    # smuggle in experiment commands — not in the same list item (compound
    # shell operators) and not in the same action (a later item would run
    # against the un-audited mutated environment).
    analyses = {cmd: analyze_command(cmd) for cmd in commands}
    mutating = [cmd for cmd in commands if analyses[cmd].mutates_environment]
    if mutating:
        compound = [cmd for cmd in mutating if analyses[cmd].has_shell_control]
        if compound:
            return AgentObservation(
                step=len(state.steps) + 1,
                action=action.action,
                stage_hint=action.stage_hint,
                error="environment-mutating commands must be single commands "
                      "without shell operators: " + "; ".join(compound),
            )
        experiments = [cmd for cmd in commands
                       if not _is_env_mutating_command(cmd) and not _is_setup_command(cmd)]
        if experiments:
            return AgentObservation(
                step=len(state.steps) + 1,
                action=action.action,
                stage_hint=action.stage_hint,
                error="environment-mutating and experiment commands cannot share "
                      "one action — install first, run audit_env, then run "
                      "experiments in a later action. Experiment commands: "
                      + "; ".join(experiments),
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
    # Package changes invalidate the certification immediately when the
    # action completes: the next action's experiment commands are refused
    # until a fresh audit_env succeeds.
    if results and mutating:
        state.last_audit = None
        # Content-addressed bookkeeping: the manifest's recorded inventory
        # no longer describes the mutated env — reset it until the next
        # passing audit finalizes the new inventory.
        try:
            from ..runtime import env_manager
            env_manager.invalidate_task_inventory(state)
        except Exception:
            pass  # bookkeeping only; the certification gate is the hard stop
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
    error = ""
    if audit.success:
        # Content-addressed mode: verify spec-declared distributions exist
        # (second drift-detection line), then finalize the manifest —
        # resolved inventory, experiment certification, audit artifact.
        try:
            from ..runtime import env_manager
            violations = env_manager.check_task_spec_compliance(state)
            if violations:
                audit = EnvironmentAudit(
                    success=False,
                    summary="spec compliance failed",
                    details=list(violations),
                    stdout_path=audit.stdout_path,
                    stderr_path=audit.stderr_path,
                )
                state.last_audit = audit
                return AgentObservation(
                    step=len(state.steps) + 1,
                    action="audit_env",
                    stage_hint="audit",
                    audit=audit,
                    error="spec compliance failed: " + "; ".join(violations),
                )
            env_manager.finalize_manifest_after_audit(state, audit)
        except Exception as exc:
            error = f"manifest finalization failed after successful audit: {exc}"
    return AgentObservation(
        step=len(state.steps) + 1,
        action="audit_env",
        stage_hint="audit",
        audit=audit,
        error=error,
    )


def _tool_call_coding_agent(action: AgentAction, state: AgentState) -> AgentObservation:
    """Delegate a code modification to CodingAgent and collect the result."""
    if not state.task.allow_code_delegation:
        issues = action.coding_issues or [action.coding_goal or "unspecified"]
        return AgentObservation(
            step=len(state.steps) + 1,
            action="call_coding_agent",
            stage_hint="coding",
            coding_issues=issues,
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
