"""Run the step-based coding agent controller main loop."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..apply import current_diff
from ..context import build_repo_context
from ..context_policy import resolve_context_policy
from ..llm import LLMClient
from ..models import AgentState, CodeTaskSpec, ControllerAction, PatchReport, StepRecord
from ..report import prepare_output_dir, write_diff, write_initial_diff, write_patch_report, write_state
from ..reviewer import review_outcome
from .actions import execute_action, _normalize_action, _run_missing_finish_verification
from .prompts import choose_next_action

def run_step_controller(spec: CodeTaskSpec) -> PatchReport:
    """Run the controller loop until finish, failure, or budget exhaustion."""
    output_dir = prepare_output_dir(spec)
    log_dir = output_dir / "logs"
    state = AgentState(task=spec)
    policy = resolve_context_policy(spec)
    context = _build_context(spec, policy)
    write_initial_diff(context.initial_diff, output_dir)
    client = LLMClient(api_base=spec.api_base, api_key_env=spec.api_key_env, model=spec.model)

    changed_files: list[str] = []
    verification_results = []
    final_error = ""

    hard_step_limit = spec.max_steps + spec.max_extra_steps_after_progress
    for step in range(1, hard_step_limit + 1):
        if step > spec.max_steps and not _should_continue_past_base_limit(spec, state.steps):
            final_error = "max_steps reached"
            break
        try:
            action = choose_next_action(spec, state, context, client)
        except Exception as llm_exc:
            state.steps.append(StepRecord(
                step=step, action=ControllerAction(action="finish", reasoning="LLM call failed."),
                observation="", error=f"LLM API call failed: {llm_exc}",
            ))
            write_state(state, output_dir)
            continue
        try:
            action = _normalize_action(spec, state.steps, action)
            (log_dir / f"action_{step:02d}.json").write_text(action.model_dump_json(indent=2), encoding="utf-8")
            record = execute_action(spec, action, output_dir, step, client)
            state.steps.append(record)
            changed_files.extend(path for path in record.changed_files if path not in changed_files)
            verification_results.extend(record.verification_results)
            write_state(state, output_dir)

            if action.action in {"replace_text", "insert_before", "insert_after", "apply_patch", "write_file", "run_command"}:
                context = _build_context(spec, policy)

            if action.action == "finish":
                auto_verification = _run_missing_finish_verification(spec, state.steps, output_dir, step)
                if auto_verification:
                    record.verification_results.extend(auto_verification)
                    record.observation = _append_observation(
                        record.observation,
                        f"Auto-ran {len(auto_verification)} verification command(s) before finish.",
                    )
                    verification_results.extend(auto_verification)
                diff_path = write_diff(current_diff(spec.workspace_path), output_dir)
                report = PatchReport(
                    status=_final_status(action.status, changed_files, verification_results),
                    changed_files=changed_files,
                    diff_path=diff_path,
                    verification_results=verification_results,
                    summary=action.summary or "Controller finished the coding task.",
                    residual_risks=action.residual_risks,
                )
                state.report = report
                write_patch_report(spec, report, output_dir)
                write_state(state, output_dir)
                return report

            if action.action == "ask_user":
                report = PatchReport(
                    status="needs_user_input",
                    changed_files=changed_files,
                    diff_path=write_diff(current_diff(spec.workspace_path), output_dir),
                    verification_results=verification_results,
                    summary=action.summary or "Controller needs user input before continuing.",
                    residual_risks=action.residual_risks,
                )
                state.report = report
                write_patch_report(spec, report, output_dir)
                write_state(state, output_dir)
                return report
        except Exception as exc:
            final_error = str(exc)
            error_action = ControllerAction(action="finish", reasoning="An unrecoverable controller error occurred.")
            state.steps.append(StepRecord(step=step, action=error_action, observation="", error=final_error))
            write_state(state, output_dir)
            break

    diff_path = write_diff(current_diff(spec.workspace_path), output_dir)
    if changed_files and verification_results:
        report = review_outcome(spec, changed_files, diff_path, verification_results, [final_error] if final_error else [])
    else:
        report = PatchReport(
            status="failed",
            changed_files=changed_files,
            diff_path=diff_path,
            verification_results=verification_results,
            summary=f"Controller stopped before completion: {final_error or 'max_steps reached'}",
            residual_risks=[final_error] if final_error else ["max_steps reached"],
        )
    state.report = report
    write_patch_report(spec, report, output_dir)
    write_state(state, output_dir)
    return report


def _append_observation(existing: str, addition: str) -> str:
    """Append text to an observation string."""
    return f"{existing}\n{addition}" if existing else addition


def _build_context(spec: CodeTaskSpec, policy: ContextPolicy):
    """Build repository context using a resolved policy."""
    return build_repo_context(
        spec,
        max_files=policy.snippet_count,
        max_bytes=policy.snippet_chars,
        tree_limit=policy.repo_tree_limit,
    )


def _should_continue_past_base_limit(spec: CodeTaskSpec, steps: list[StepRecord]) -> bool:
    """Allow grace steps only after unverified progress."""
    if spec.max_extra_steps_after_progress <= 0:
        return False
    last_change_step = max((step.step for step in steps if step.changed_files), default=0)
    if last_change_step == 0:
        return False
    last_verify_step = max((step.step for step in steps if step.verification_results), default=0)
    return last_verify_step < last_change_step


def _final_status(requested_status: str | None, changed_files: list[str], verification_results) -> str:
    """Return the finish status.

    The agent's explicit finish status is authoritative --- it has semantic
    understanding of whether the task succeeded.  Verification results are
    evidence for the caller to inspect but do not override the agent's judgment.

    When the agent did *not* provide a status (error, budget exhaustion, or
    ask_user without an explicit marker), fall back to a conservative inference
    from changed files and verification results.
    """
    if requested_status:
        return requested_status
    return _status_from_verification(changed_files, verification_results)


def _status_from_verification(changed_files: list[str], verification_results) -> str:
    """Infer status from changed files and verification results."""
    if not changed_files:
        return "failed"
    if verification_results and all(result.succeeded for result in verification_results):
        return "completed"
    return "failed"
