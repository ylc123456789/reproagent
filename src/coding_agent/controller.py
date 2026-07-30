from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .apply import PatchApplyError, apply_patch_text, current_diff, extract_patch_paths, normalize_patch_text
from .context import build_repo_context
from .llm import LLMClient
from .models import AgentState, CodeTaskSpec, ControllerAction, PatchReport, StepRecord
from .report import prepare_output_dir, write_diff, write_initial_diff, write_patch_report, write_state
from .reviewer import review_outcome
from .runner import run_verify_commands
from .safety import SafetyError, ensure_path_allowed


ACTION_SCHEMA = {
    "action": "list_tree|read_file|search|apply_patch|run_command|finish|ask_user",
    "reasoning": "brief reason for this next action",
    "path": "relative file path for read_file, optional",
    "query": "search query for search, optional",
    "command": "verification command for run_command, optional",
    "patch": "unified diff for apply_patch, optional",
    "status": "completed|failed|blocked|needs_user_input for finish/ask_user",
    "summary": "final or user-facing summary, optional",
    "residual_risks": ["risk strings for finish/ask_user"],
}


class PatchRepairResponse(BaseModel):
    patch: str
    notes: list[str] = Field(default_factory=list)


def run_step_controller(spec: CodeTaskSpec) -> PatchReport:
    output_dir = prepare_output_dir(spec)
    log_dir = output_dir / "logs"
    state = AgentState(task=spec)
    context = build_repo_context(spec)
    write_initial_diff(context.initial_diff, output_dir)
    client = LLMClient(api_base=spec.api_base, api_key_env=spec.api_key_env, model=spec.model)

    changed_files: list[str] = []
    verification_results = []
    final_error = ""

    for step in range(1, spec.max_steps + 1):
        try:
            action = choose_next_action(spec, state, context, client)
            (log_dir / f"action_{step:02d}.json").write_text(action.model_dump_json(indent=2), encoding="utf-8")
            record = execute_action(spec, action, output_dir, step, client)
            state.steps.append(record)
            changed_files.extend(path for path in record.changed_files if path not in changed_files)
            verification_results.extend(record.verification_results)
            write_state(state, output_dir)

            if action.action in {"apply_patch", "run_command"}:
                context = build_repo_context(spec)

            if action.action == "finish":
                diff_path = write_diff(current_diff(spec.repo_path), output_dir)
                report = PatchReport(
                    status=_final_status(action.status, changed_files, verification_results, bool(spec.verify_commands)),
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
                    diff_path=write_diff(current_diff(spec.repo_path), output_dir),
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

    diff_path = write_diff(current_diff(spec.repo_path), output_dir)
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


def choose_next_action(spec: CodeTaskSpec, state: AgentState, context, client: LLMClient) -> ControllerAction:
    system = (
        "You are a coding agent controller inspired by modern agentic coding tools. "
        "Choose exactly one next action from the allowed action set. "
        "Use read/search/run before editing when useful. Use apply_patch for repo changes. "
        "Use finish only after the diff and verification evidence satisfy the task, or when failure is clear. "
        "Never silently change research semantics such as model architecture, loss, optimizer, dataset split, or metric. "
        "Return only JSON matching the schema."
    )
    user = {
        "task_goal": spec.task_goal,
        "constraints": spec.constraints,
        "verify_commands": spec.verify_commands,
        "allowed_paths": spec.allowed_paths,
        "repo_tree": context.tree[:300],
        "snippets": [snippet.model_dump() for snippet in context.snippets[:8]],
        "current_diff_tail": current_diff(spec.repo_path)[-8000:],
        "steps": [_compact_step(step) for step in state.steps[-10:]],
        "available_actions": ACTION_SCHEMA,
    }
    return ControllerAction.model_validate(client.complete_json(system, json.dumps(user, indent=2)))


def execute_action(spec: CodeTaskSpec, action: ControllerAction, output_dir: Path, step: int, client: LLMClient) -> StepRecord:
    if action.action == "list_tree":
        context = build_repo_context(spec)
        return StepRecord(step=step, action=action, observation="\n".join(context.tree[:300]))
    if action.action == "read_file":
        if not action.path:
            raise ValueError("read_file requires path")
        path = ensure_path_allowed(spec.repo_path, action.path, spec.allowed_paths or None)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return StepRecord(step=step, action=action, observation=text[:20_000])
    if action.action == "search":
        if not action.query:
            raise ValueError("search requires query")
        return StepRecord(step=step, action=action, observation=_search_repo(spec.repo_path, action.query))
    if action.action == "apply_patch":
        if not action.patch:
            raise ValueError("apply_patch requires patch")
        changed, observation = _apply_patch_with_repair(spec, action.patch, output_dir, step, client)
        return StepRecord(step=step, action=action, observation=observation, changed_files=changed)
    if action.action == "run_command":
        command = action.command or (spec.verify_commands[0] if spec.verify_commands else None)
        if not command:
            raise ValueError("run_command requires command")
        results = run_verify_commands(spec.repo_path, [command], output_dir / "logs" / f"step_{step:02d}", spec.timeout_seconds)
        result = results[0]
        stdout_tail = result.stdout_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        stderr_tail = result.stderr_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        return StepRecord(
            step=step,
            action=action,
            observation=(
                f"returncode={result.returncode} timed_out={result.timed_out}\n"
                f"stdout_tail:\n{stdout_tail}\n"
                f"stderr_tail:\n{stderr_tail}"
            ),
            verification_results=results,
        )
    if action.action in {"finish", "ask_user"}:
        return StepRecord(step=step, action=action, observation=action.summary or action.reasoning)
    raise ValueError(f"unsupported action: {action.action}")


def _apply_patch_with_repair(
    spec: CodeTaskSpec,
    patch_text: str,
    output_dir: Path,
    step: int,
    client: LLMClient,
) -> tuple[list[str], str]:
    patch = normalize_patch_text(patch_text)
    errors: list[str] = []
    repair_notes: list[str] = []
    max_attempts = spec.patch_repair_attempts + 1

    for attempt in range(1, max_attempts + 1):
        try:
            changed = apply_patch_text(spec.repo_path, patch, spec.allowed_paths)
            diff_path = write_diff(current_diff(spec.repo_path), output_dir)
            suffix = "" if attempt == 1 else f" after {attempt - 1} repair attempt(s)"
            notes = f" Repair notes: {'; '.join(repair_notes)}" if repair_notes else ""
            return changed, f"Patch applied{suffix}. Current diff written to {diff_path}.{notes}"
        except (PatchApplyError, SafetyError) as exc:
            stderr = getattr(exc, "stderr", "") or str(exc)
            errors.append(stderr)
            _save_failed_patch(output_dir, step, attempt, patch, stderr)
            if attempt >= max_attempts:
                joined = "\n---\n".join(errors)
                raise PatchApplyError(
                    f"patch failed validation/application after {spec.patch_repair_attempts} repair attempt(s):\n{joined}",
                    joined,
                ) from exc
            repaired = repair_patch(spec, patch, stderr, client)
            patch = normalize_patch_text(repaired.patch)
            repair_notes.extend(repaired.notes)

    raise PatchApplyError("patch repair loop exited unexpectedly")


def repair_patch(spec: CodeTaskSpec, failed_patch: str, stderr: str, client: LLMClient) -> PatchRepairResponse:
    paths = extract_patch_paths(failed_patch)
    file_context = []
    for rel in paths[:6]:
        try:
            path = ensure_path_allowed(spec.repo_path, rel, spec.allowed_paths or None)
            file_context.append({"path": rel, "text": path.read_text(encoding="utf-8", errors="ignore")[:24_000]})
        except Exception as exc:
            file_context.append({"path": rel, "error": str(exc)})

    system = (
        "You repair malformed or non-applicable unified diffs for a coding agent. "
        "Return only JSON with fields patch and notes. The patch must be a valid unified diff, use repo-relative paths, "
        "and preserve the original edit intent without adding unrelated changes. Do not use markdown fences."
    )
    user = {
        "task_goal": spec.task_goal,
        "constraints": spec.constraints,
        "allowed_paths": spec.allowed_paths,
        "git_apply_check_or_apply_error": stderr,
        "failed_patch": failed_patch,
        "current_file_context": file_context,
        "schema": {"patch": "valid unified diff", "notes": ["repair note"]},
    }
    return PatchRepairResponse.model_validate(client.complete_json(system, json.dumps(user, indent=2)))


def _save_failed_patch(output_dir: Path, step: int, attempt: int, patch: str, stderr: str) -> None:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = f"failed_patch_{step:02d}_{attempt:02d}"
    (logs / f"{stem}.patch").write_text(patch, encoding="utf-8")
    (logs / f"{stem}.stderr").write_text(stderr, encoding="utf-8")


def _search_repo(repo: Path, query: str, limit: int = 80) -> str:
    matches: list[str] = []
    lowered = query.lower()
    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo).as_posix()
        try:
            ensure_path_allowed(repo, rel)
        except SafetyError:
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            if lowered in line.lower():
                matches.append(f"{rel}:{number}: {line}")
                if len(matches) >= limit:
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches."


def _compact_step(step: StepRecord) -> dict[str, object]:
    return {
        "step": step.step,
        "action": step.action.action,
        "reasoning": step.action.reasoning,
        "observation_tail": step.observation[-2000:],
        "changed_files": step.changed_files,
        "verification": [
            {"command": result.command, "returncode": result.returncode, "timed_out": result.timed_out}
            for result in step.verification_results
        ],
        "error": step.error,
    }


def _final_status(requested_status: str | None, changed_files: list[str], verification_results, verification_required: bool) -> str:
    evidence_status = _status_from_verification(changed_files, verification_results)
    if requested_status == "completed" and verification_required and evidence_status != "completed":
        return evidence_status
    return requested_status or evidence_status


def _status_from_verification(changed_files: list[str], verification_results) -> str:
    if not changed_files:
        return "failed"
    if verification_results and all(result.succeeded for result in verification_results):
        return "completed"
    return "failed"
