"""CLI and top-level workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

from .context import collect_context
from .coding import run_coding_agent_for_patch
from .audit import audit_environment
from .env import ensure_environment
from .llm import final_review, plan_environment, plan_experiment, plan_probe, revise_after_failure
from .models import ReproState, ReproTask, StageResult
from .report import save_state, write_result
from .runner import run_commands
from .text import normalize_text
from .validation import validate_experiment_plan


def _log(message: str) -> None:
    print(f"[reproagent] {message}", flush=True)


def run_task(task: ReproTask) -> ReproState:
    task.workspace_dir.mkdir(parents=True, exist_ok=True)
    state = ReproState(task=task, status="started")
    save_state(state)
    _log(f"workspace: {task.workspace_dir}")
    _log(f"experiment goal: {task.experiment_goal}")

    state.status = "collecting_context"
    _log("collecting repository, README, hardware, and paper context")
    try:
        state.repo_context = collect_context(task)
    except Exception as exc:
        state.status = "failed"
        state.final_summary = f"Context collection failed: {exc}"
        write_result(state)
        return state
    save_state(state)

    state.status = "preparing_conda_environment"
    _log("preparing conda environment")
    try:
        state.environment = ensure_environment(state)
    except Exception as exc:
        state.status = "failed"
        state.final_summary = f"Environment preparation failed: {exc}"
        write_result(state)
        return state
    save_state(state)

    env_ok = _run_stage_loop(state, stage="environment", max_attempts=task.max_env_attempts)
    probe_ok = _run_stage_loop(state, stage="probe", max_attempts=1) if env_ok else False
    if env_ok and probe_ok and task.plan_only:
        state.status = "planning_experiment"
        _log("planning final experiment without running commands")
        state.planned_experiment = _validated_experiment_plan(state)
        _print_plan("experiment", 1, state.planned_experiment)
        state.status = "planned"
        state.final_summary = "Experiment plan generated with --plan-only. No experiment commands were executed."
        write_result(state)
        return state

    exp_ok = _run_stage_loop(state, stage="experiment", max_attempts=task.max_run_attempts) if env_ok and probe_ok else False

    state.status = "reviewing"
    _log("writing final review")
    state.final_summary = _final_summary(state)
    state.status = "completed" if env_ok and probe_ok and exp_ok else "completed_with_failures"
    write_result(state)
    return state


def _final_summary(state: ReproState) -> str:
    blocked_result = _blocking_coding_agent_result(state)
    if blocked_result is not None:
        lines = [
            "Experiment commands were not executed because CodingAgent did not complete the required patch.",
            "",
            f"CodingAgent status: {blocked_result.status}",
        ]
        if state.planned_experiment:
            lines += [
                f"Experiment feasibility: {state.planned_experiment.feasibility or 'unknown'}",
                f"Experiment plan: {state.planned_experiment.summary}",
            ]
            if state.planned_experiment.needs_user_input:
                lines += ["Required decision or fix:"]
                lines += [f"- {item}" for item in state.planned_experiment.needs_user_input]
        if blocked_result.summary:
            lines += ["", f"CodingAgent summary: {blocked_result.summary}"]
        if blocked_result.report_path:
            lines += [f"CodingAgent report: {blocked_result.report_path}"]
        if blocked_result.diff_path:
            lines += [f"CodingAgent diff: {blocked_result.diff_path}"]
        if blocked_result.output_dir:
            lines += [f"CodingAgent output dir: {blocked_result.output_dir}"]
        lines += [
            "",
            "Next step: fix the CodingAgent patch/apply failure or update the required repo-local patch, then rerun reproagent.",
        ]
        return normalize_text("\n".join(lines))
    return normalize_text(final_review(state))


def _blocking_coding_agent_result(state: ReproState):
    if not state.coding_agent_results:
        return None
    latest = state.coding_agent_results[-1]
    experiment_commands_ran = any(attempt.results for attempt in state.experiment_attempts)
    if latest.status != "completed" and not experiment_commands_ran:
        return latest
    return None


def _run_stage_loop(state: ReproState, stage: str, max_attempts: int) -> bool:
    assert state.repo_context is not None
    assert state.environment is not None
    for attempt in range(1, max_attempts + 1):
        _log(f"planning {stage} attempt {attempt}/{max_attempts}")
        if attempt == 1:
            if stage == "environment":
                plan = plan_environment(state)
            elif stage == "probe":
                plan = plan_probe(state)
            else:
                plan = _validated_experiment_plan(state)
        else:
            plan = revise_after_failure(state, stage=stage)

        _print_plan(stage, attempt, plan)
        if stage == "experiment" and plan.feasibility and plan.feasibility != "ready_to_run":
            if plan.feasibility == "needs_patch" and state.task.enable_coding_agent and not state.task.plan_only:
                if _run_coding_agent_patch_cycle(state, plan):
                    plan = _validated_experiment_plan(state)
                    _print_plan(stage, attempt, plan)
                    if plan.feasibility and plan.feasibility != "ready_to_run":
                        _log(f"experiment not executed because plan feasibility is {plan.feasibility}")
                        state.experiment_attempts.append(StageResult(stage=stage, attempt=attempt, plan=plan, results=[]))
                        save_state(state)
                        return False
                else:
                    state.experiment_attempts.append(StageResult(stage=stage, attempt=attempt, plan=plan, results=[]))
                    save_state(state)
                    return False
            else:
                _log(f"experiment not executed because plan feasibility is {plan.feasibility}")
                state.experiment_attempts.append(StageResult(stage=stage, attempt=attempt, plan=plan, results=[]))
                save_state(state)
                return False
        if stage == "experiment" and state.task.confirm_before_experiment and not _confirm_experiment(plan):
            _log("experiment cancelled by user before command execution")
            return False

        results = []
        if plan.commands:
            results = run_commands(
                plan.commands,
                cwd=state.repo_context.repo_path,
                workspace=state.task.workspace_dir,
                stage=stage,
                attempt=attempt,
                timeout=state.task.timeout_seconds,
                env_name=state.environment.env_name,
            )
        stage_result = StageResult(stage=stage, attempt=attempt, plan=plan, results=results)
        if stage == "environment":
            state.environment_attempts.append(stage_result)
        elif stage == "probe":
            state.probe_attempts.append(stage_result)
        else:
            state.experiment_attempts.append(stage_result)
        save_state(state)

        if stage == "environment" and (stage_result.success or not plan.commands):
            state.status = "auditing_environment"
            _log("auditing environment")
            state.environment_audit = audit_environment(state)
            save_state(state)
            if state.environment_audit.success and not state.environment_audit.requires_repair:
                return True
            continue

        if _stage_succeeded(stage, stage_result):
            return True
    return False


def _stage_succeeded(stage: str, stage_result: StageResult) -> bool:
    if not stage_result.plan.commands:
        return True
    if stage == "probe":
        return any(result.success for result in stage_result.results)
    return stage_result.success


def _validated_experiment_plan(state: ReproState):
    plan = plan_experiment(state)
    validated = validate_experiment_plan(state, plan)
    if validated is not plan and validated.needs_user_input:
        _log("plan validation flagged issues")
    state.planned_experiment = validated
    return validated


def _run_coding_agent_patch_cycle(state: ReproState, plan) -> bool:
    _log("running CodingAgent to resolve patch-required validation issues")
    try:
        result = run_coding_agent_for_patch(state, plan)
    except Exception as exc:
        _log(f"CodingAgent failed: {exc}")
        return False
    state.coding_agent_results.append(result)
    save_state(state)
    _log(f"CodingAgent status: {result.status}")
    if result.report_path:
        _log(f"CodingAgent report: {result.report_path}")
    if result.status != "completed":
        return False
    _log("rerunning probe after CodingAgent patch")
    return _run_probe_once_after_patch(state)


def _run_probe_once_after_patch(state: ReproState) -> bool:
    assert state.repo_context is not None
    assert state.environment is not None
    attempt = len(state.probe_attempts) + 1
    plan = plan_probe(state)
    _print_plan("probe", attempt, plan)
    results = []
    if plan.commands:
        results = run_commands(
            plan.commands,
            cwd=state.repo_context.repo_path,
            workspace=state.task.workspace_dir,
            stage="probe",
            attempt=attempt,
            timeout=state.task.timeout_seconds,
            env_name=state.environment.env_name,
        )
    stage_result = StageResult(stage="probe", attempt=attempt, plan=plan, results=results)
    state.probe_attempts.append(stage_result)
    save_state(state)
    return _stage_succeeded("probe", stage_result)


def _print_plan(stage: str, attempt: int, plan) -> None:
    _log(f"{stage} attempt {attempt} plan: {plan.summary or 'no summary'}")
    if plan.feasibility:
        _log(f"feasibility: {plan.feasibility}")
    if plan.expected_runtime:
        _log(f"expected runtime: {plan.expected_runtime}")
    if plan.needs_user_input:
        _log("needs user input:")
        for item in plan.needs_user_input:
            print(f"  - {item}", flush=True)
    if plan.assumptions:
        _log("assumptions:")
        for item in plan.assumptions:
            print(f"  - {item}", flush=True)
    if plan.commands:
        _log("planned commands:")
        for index, command in enumerate(plan.commands, start=1):
            print(f"  {index}. {command}", flush=True)
    else:
        _log("no commands planned")


def _confirm_experiment(plan) -> bool:
    if not plan.commands:
        return True
    try:
        answer = input("[reproagent] Run these experiment commands? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-first ML paper repo reproduction runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--paper", required=True, help="Paper URL, e.g. arXiv link")
    run.add_argument("--repo", required=True, help="Git repository URL")
    run.add_argument("--workspace", required=True, type=Path, help="Run workspace directory")
    run.add_argument("--mock-llm", action="store_true", help="Use deterministic mock LLM for local tests")
    run.add_argument("--model", default=None, help="OpenAI-compatible model name")
    run.add_argument("--api-base", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key")
    run.add_argument("--backend", default="conda", choices=["conda"], help="Execution backend; MVP supports conda only")
    run.add_argument("--python-version", default="3.10", help="Python version for conda env when no environment.yml exists")
    run.add_argument("--max-env-attempts", type=int, default=3)
    run.add_argument("--max-run-attempts", type=int, default=3)
    run.add_argument("--timeout", type=int, default=1800)
    run.add_argument("--experiment-goal", required=True, help="Concrete reproduction goal for this run")
    run.add_argument("--confirm-before-experiment", action="store_true", help="Print the experiment plan and wait for y/N before running experiment commands")
    run.add_argument("--plan-only", action="store_true", help="Stop after probe and final experiment-plan generation; do not run experiment commands")
    run.add_argument("--enable-coding-agent", action="store_true", help="Allow CodingAgent to patch the cloned repository when validation reports needs_patch")
    run.add_argument("--max-coding-agent-steps", type=int, default=12, help="Maximum CodingAgent controller steps per patch attempt")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        task = ReproTask(
            paper_url=args.paper,
            repo_url=args.repo,
            workspace_dir=args.workspace,
            mock_llm=args.mock_llm,
            model=args.model,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            backend=args.backend,
            python_version=args.python_version,
            max_env_attempts=args.max_env_attempts,
            max_run_attempts=args.max_run_attempts,
            timeout_seconds=args.timeout,
            experiment_goal=args.experiment_goal,
            confirm_before_experiment=args.confirm_before_experiment,
            plan_only=args.plan_only,
            enable_coding_agent=args.enable_coding_agent,
            max_coding_agent_steps=args.max_coding_agent_steps,
        )
        state = run_task(task)
        print(f"status: {state.status}")
        print(f"result: {state.result_path}")


if __name__ == "__main__":
    main()
