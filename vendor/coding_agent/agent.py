"""Provide the top-level entrypoint for running a coding task."""
from __future__ import annotations

from .controller import run_step_controller
from .models import CodeTaskSpec, PatchReport






def _extract_evidence(output_dir):
    """Extract file evidence from step records.

    Collects paths from read_file actions, search results (parsed as
    path:line: matches), and run_command invocations that reference
    file paths in their command or observation.
    """
    import json, re
    from .models import Snippet

    state_path = output_dir / "state.json"
    if not state_path.exists():
        return [], []

    state = json.loads(state_path.read_text())
    evidence_files = []
    raw_snippets = []
    seen_paths = set()
    seen_snippet_paths = set()

    _file_re = re.compile(r"^(?P<path>[^\s:]+?\.[a-z]{1,6}):\d+:", re.MULTILINE)

    for step in state.get("steps", []):
        action = step.get("action", {})
        obs = step.get("observation", "")

        # 1. read_file — primary evidence
        if action.get("action") == "read_file" and action.get("path"):
            path = action["path"]
            if path not in seen_paths:
                seen_paths.add(path)
                evidence_files.append(path)
            if obs and path not in seen_snippet_paths:
                seen_snippet_paths.add(path)
                start = action.get("start_line")
                end = action.get("end_line")
                if start:
                    raw_snippets.append({
                        "path": path, "content": obs[:2000],
                        "start_line": start,
                        "end_line": end or (start + obs.count("\n")),
                        "why": "",
                    })
                else:
                    raw_snippets.append({
                        "path": path, "content": obs[:2000],
                        "start_line": 1,
                        "end_line": obs.count("\n") + 1,
                        "why": "",
                    })

        # 2. search / grep output — extract path:line: matches
        if action.get("action") in ("search", "run_command") and obs:
            for m in _file_re.finditer(obs):
                path = m.group("path")
                if path not in seen_paths:
                    seen_paths.add(path)
                    evidence_files.append(path)

    return evidence_files, [Snippet(**s) for s in raw_snippets[:10]]

def _detect_uncertainty(text):
    """Scan answer text for uncertainty signals."""
    signals = []
    lowered = text.lower()
    if "uncertain" in lowered or "not sure" in lowered:
        signals.append("Answer contains explicit uncertainty markers.")
    if "may " in lowered or "might " in lowered or "could be" in lowered:
        signals.append("Answer uses hedging language (may/might/could).")
    if "assume" in lowered or "likely" in lowered:
        signals.append("Answer contains assumptions or likelihood statements.")
    return "; ".join(signals) if signals else ""


def run_code_question(spec):
    """Run a read-only code understanding question.

    Internally wraps CodeQuestionSpec into a read_only CodeTaskSpec
    and reuses the step controller.  The result is a CodeExplanation
    (answer + evidence) instead of a PatchReport.
    """
    from .controller.loop import run_step_controller
    from .models import CodeTaskSpec, CodeExplanation

    task_spec = CodeTaskSpec(
        workspace_path=spec.workspace_path,
        task_goal=(
            f"Question: {spec.question}"
            + (f"\n\nContext hint: {spec.context_hint}" if spec.context_hint else "")
        ),
        constraints=spec.constraints + [
            "Do NOT modify any files. Read-only access only.",
        ],
        output_dir=spec.output_dir,
        read_only=True,
        max_steps=spec.max_steps,
        timeout_seconds=spec.timeout_seconds,
        model=spec.model,
        api_base=spec.api_base,
        api_key_env=spec.api_key_env,
        max_context_tokens=spec.max_context_tokens,
        model_context_window_tokens=spec.model_context_window_tokens,
        context_margin_ratio=spec.context_margin_ratio,
        context_output_reserve_tokens=spec.context_output_reserve_tokens,
    )
    report = run_step_controller(task_spec)

    evidence_files, snippets = _extract_evidence(spec.output_dir)
    uncertainty = _detect_uncertainty(report.summary)

    return CodeExplanation(
        status=report.status,
        answer=report.summary,
        evidence_files=evidence_files,
        relevant_snippets=snippets,
        uncertainty=uncertainty,
        commands_run=report.verification_results,
    )


def run_code_task(spec: CodeTaskSpec) -> PatchReport:
    """Run a coding task through the step controller."""
    return run_step_controller(spec)
