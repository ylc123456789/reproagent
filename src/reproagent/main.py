"""CLI and top-level workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

from .context import collect_context
from .audit import audit_environment
from .env import ensure_environment
from .llm import final_review, plan_environment, plan_experiment, revise_after_failure
from .models import ReproState, ReproTask, StageResult
from .report import save_state, write_result
from .runner import run_commands


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
    exp_ok = _run_stage_loop(state, stage="experiment", max_attempts=task.max_run_attempts) if env_ok else False

    state.status = "reviewing"
    _log("writing final review")
    state.final_summary = final_review(state)
    state.status = "completed" if env_ok and exp_ok else "completed_with_failures"
    write_result(state)
    return state


def _run_stage_loop(state: ReproState, stage: str, max_attempts: int) -> bool:
    assert state.repo_context is not None
    assert state.environment is not None
    for attempt in range(1, max_attempts + 1):
        _log(f"planning {stage} attempt {attempt}/{max_attempts}")
        if attempt == 1:
            plan = plan_environment(state) if stage == "environment" else plan_experiment(state)
        else:
            plan = revise_after_failure(state, stage=stage)

        _print_plan(stage, attempt, plan)
        if stage == "experiment" and state.task.confirm_before_experiment and not _confirm_experiment(plan):
            _log("experiment cancelled by user before command execution")
            return False

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

        if not plan.commands:
            return True
        if stage_result.success:
            return True
    return False


def _print_plan(stage: str, attempt: int, plan) -> None:
    _log(f"{stage} attempt {attempt} plan: {plan.summary or 'no summary'}")
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
        )
        state = run_task(task)
        print(f"status: {state.status}")
        print(f"result: {state.result_path}")


if __name__ == "__main__":
    main()
