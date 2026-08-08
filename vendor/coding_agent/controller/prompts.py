"""Build controller prompts and compact step history."""
from __future__ import annotations

import json
from pathlib import Path

from ..apply import current_diff
from ..context_policy import ContextPolicy, resolve_context_policy
from ..llm import LLMClient
from ..models import AgentState, CodeTaskSpec, ControllerAction, StepRecord

ACTION_SCHEMA = {
    "action": "list_tree|read_file|search|replace_text|insert_before|insert_after|apply_patch|write_file|run_command|finish|ask_user",
    "reasoning": "brief reason for this next action",
    "path": "relative file path for read_file or structured edits, optional",
    "start_line": "optional 1-based start line for read_file",
    "end_line": "optional 1-based inclusive end line for read_file",
    "query": "search query for search, optional",
    "command": "verification command for run_command, optional",
    "patch": "unified diff for apply_patch, optional",
    "content": "full file content for write_file when creating or overwriting a file",
    "old_text": "exact text copied from the current file for replace_text",
    "new_text": "replacement text for replace_text",
    "anchor_text": "exact unique anchor copied from the current file for insert_before/insert_after; prefer several adjacent lines over a short common line",
    "insert_text": "text to insert before or after anchor_text",
    "occurrence_index": "optional 1-based match index only when a repeated anchor is intentional and read_file/search context proves the target occurrence",
    "status": "completed|failed|blocked|needs_user_input for finish/ask_user",
    "summary": "final or user-facing summary, optional",
    "residual_risks": ["risk strings for finish/ask_user"],
}




# Read-only action set for code question answering
QA_ACTION_SCHEMA = {
    "action": "list_tree|read_file|search|run_command|finish|ask_user",
    "reasoning": "brief reason for this next action",
    "path": "relative file path for read_file, optional",
    "start_line": "optional 1-based start line for read_file",
    "end_line": "optional 1-based inclusive end line for read_file",
    "query": "search query for search, optional",
    "command": "read-only shell command for run_command (allowed: ls, cat, head, tail, grep, rg, find, wc, file, pwd, tree)",
    "status": "completed|failed|blocked|needs_user_input for finish/ask_user",
    "summary": "final answer or user-facing summary; use markdown",
    "residual_risks": ["risk strings for finish/ask_user"],
}

QA_SYSTEM = (
    "You are a code understanding agent. Answer questions about the repository "
    "by reading files (prefer read_file over grep to get exact code context) "
    "and running read-only shell commands (grep, find, ls, cat, etc.). "
    "Always read_file before citing line numbers to confirm accuracy. "
    "Your answer MUST include: (1) file paths and line numbers for every claim, "
    "(2) relevant code snippets copied from the files, "
    "(3) explicit uncertainty statements where applicable. "
    "You CANNOT modify any files — write actions are disabled. "
    "Use finish with status=completed (or failed if you cannot answer), "
    "and a well-structured markdown answer in the summary field. "
    "Return only JSON matching the schema."
)

def choose_next_action(spec: CodeTaskSpec, state: AgentState, context, client: LLMClient) -> ControllerAction:
    """Ask the model to choose the next controller action."""
    policy = resolve_context_policy(spec)
    is_qa = getattr(spec, "read_only", False)
    system = QA_SYSTEM if is_qa else (
        "You are a coding agent controller inspired by modern agentic coding tools. "
        "Choose exactly one next action from the allowed action set. "
        "After reading a file, prefer structured edit actions (replace_text, insert_before, insert_after) for small local edits. "
        "Do not repeatedly read the same file when recent_file_observations already contain the needed text; make progress "
        "by editing, searching for a specific symbol, running verification, or finishing. "
        "Use exact old_text or anchor_text copied from the current file. For inserts, prefer a unique multi-line "
        "anchor that includes nearby context instead of a short common line. Use apply_patch only for changes that are not suitable "
        "for exact structured edits. Use finish only after the diff and verification evidence satisfy the task, or when failure is clear. "
        "Never silently change existing behavior; prefer adding new code over modifying existing logic. "
        "For insert_before/insert_after anchors: prefer 2-4 adjacent lines as anchor, including the line above the target. "
        "Never use anchors consisting only of whitespace and punctuation (e.g. a closing parenthesis alone). "
        "When nesting is deep, include the parent construct opening line in the anchor. "
        "Return only JSON matching the schema."
    )
    user = {
        "task_goal": spec.task_goal,
        "constraints": spec.constraints,
        "verify_commands": spec.verify_commands,
        "allowed_paths": spec.allowed_paths,
        "context_budget": {
            "context_window_tokens": policy.context_window_tokens,
            "input_budget_tokens": policy.input_budget_tokens,
            "margin_ratio": spec.context_margin_ratio,
            "output_reserve_tokens": spec.context_output_reserve_tokens,
        },
        "repo_tree": context.tree[:policy.repo_tree_limit],
        "snippets": [snippet.model_dump() for snippet in context.snippets[:policy.snippet_count]],
        "current_diff_tail": current_diff(spec.workspace_path)[-policy.diff_chars:],
        "remaining_base_steps": max(spec.max_steps - len(state.steps), 0),
        "remaining_hard_steps": max(spec.max_steps + spec.max_extra_steps_after_progress - len(state.steps), 0),
        "progress_hints": _progress_hints(spec, state.steps),
        "recent_file_observations": _recent_file_observations(state.steps, policy),
        "steps": [_compact_step(step, policy) for step in state.steps[-10:]],
        "available_actions": QA_ACTION_SCHEMA if getattr(spec, "read_only", False) else ACTION_SCHEMA,
    }
    return ControllerAction.model_validate(client.complete_json(system, json.dumps(user, indent=2)))


def _recent_file_observations(steps: list[StepRecord], policy: ContextPolicy | None = None) -> list[dict[str, object]]:
    """Return recent read-file observations for prompt reuse."""
    limit = policy.recent_file_count if policy else 2
    char_limit = policy.recent_file_chars if policy else 24_000
    observations = []
    seen = set()
    for step in reversed(steps):
        action = step.action
        if action.action != "read_file" or not action.path or action.path in seen:
            continue
        seen.add(action.path)
        observations.append({
            "path": action.path,
            "start_line": action.start_line,
            "end_line": action.end_line,
            "chars": len(step.observation),
            "text": step.observation[:char_limit],
        })
        if len(observations) >= limit:
            break
    return observations


def _progress_hints(spec: CodeTaskSpec, steps: list[StepRecord]) -> list[str]:
    """Build short hints that discourage stalled behavior."""
    hints = []
    if not steps:
        return hints
    last = steps[-1].action
    repeated_reads = 0
    for step in reversed(steps):
        action = step.action
        if action.action == "read_file" and last.action == "read_file" and action.path == last.path:
            repeated_reads += 1
        else:
            break
    if repeated_reads >= 2 and last.path:
        hints.append(
            f"{last.path} has already been read {repeated_reads} consecutive times; use the recent_file_observations text to edit, search a specific symbol, run verification, or finish instead of reading it again."
        )
    remaining_base = spec.max_steps - len(steps)
    if remaining_base <= 4:
        hints.append("The base step budget is nearly exhausted; prefer concrete edits, verification, or finish over broad exploration.")
    last_change_step = max((step.step for step in steps if step.changed_files), default=0)
    last_verify_step = max((step.step for step in steps if step.verification_results), default=0)
    if last_change_step and last_verify_step < last_change_step:
        hints.append("Files changed after the last verification; run verification before finish.")
    return hints


def _compact_step(step: StepRecord, policy: ContextPolicy | None = None) -> dict[str, object]:
    """Compact a step record for the next prompt."""
    observation_chars = policy.step_observation_chars if policy else 2_000
    return {
        "step": step.step,
        "action": step.action.action,
        "reasoning": step.action.reasoning,
        "observation_tail": step.observation[-observation_chars:],
        "changed_files": step.changed_files,
        "verification": [
            {"command": result.command, "returncode": result.returncode, "timed_out": result.timed_out}
            for result in step.verification_results
        ],
        "error": step.error,
    }


