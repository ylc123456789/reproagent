"""CLI and top-level workflow."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .context import collect_context
from .coding import run_coding_agent_for_patch
from .audit import audit_environment
from .env import ensure_environment
from .integrations.codingagent import configured_codingagent_path
from .llm import final_review, plan_environment, plan_experiment, plan_probe, revise_after_failure, review_experiment_plan, apply_review_to_plan
from .models import ReproAgentVersion, ReproState, ReproTask, StageResult
from .report import save_state, write_result
from .runner import run_commands
from .text import normalize_text


def _log(message: str) -> None:
    """Print a prefixed workflow log message."""
    print(f"[reproagent] {message}", flush=True)


def _current_reproagent_version() -> ReproAgentVersion:
    """Collect git metadata for the running reproagent checkout."""
    source_path = Path(__file__).resolve().parents[2]
    return ReproAgentVersion(
        source_path=source_path,
        git_commit=_git_output(source_path, "rev-parse", "HEAD"),
        git_branch=_git_output(source_path, "branch", "--show-current"),
        git_dirty=_git_dirty(source_path),
        git_remote=_git_output(source_path, "config", "--get", "remote.origin.url"),
    )


def _git_output(repo_path: Path, *args: str) -> str | None:
    """Return one git command output value, or None outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _git_dirty(repo_path: Path) -> bool | None:
    """Return whether tracked reproagent files differ from HEAD."""
    try:
        worktree = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--quiet", "--"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        index = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--cached", "--quiet", "--"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    if worktree.returncode not in {0, 1} or index.returncode not in {0, 1}:
        return None
    return worktree.returncode == 1 or index.returncode == 1


def _format_reproagent_version(version: ReproAgentVersion | None) -> str:
    """Format reproagent version metadata for terminal logs."""
    if version is None:
        return "reproagent version: unknown"
    commit = (version.git_commit or "unknown")[:12]
    branch = version.git_branch or "unknown-branch"
    dirty = "dirty" if version.git_dirty else "clean" if version.git_dirty is False else "dirty-unknown"
    remote = f", remote={version.git_remote}" if version.git_remote else ""
    return f"reproagent version: {branch}@{commit} ({dirty}){remote}"


def run_task(task: ReproTask) -> ReproState:
    """Run the full reproduction workflow for a task."""
    task.workspace_dir.mkdir(parents=True, exist_ok=True)
    state = ReproState(task=task, status="started", reproagent_version=_current_reproagent_version())
    save_state(state)
    _log(f"workspace: {task.workspace_dir}")
    _log(_format_reproagent_version(state.reproagent_version))
    _log(f"experiment goal: {task.experiment_goal}")

    state.status = "collecting_context"
    _log("collecting repository, README, hardware, and paper context")
    try:
        state.repo_context = collect_context(task)
    except Exception as exc:
        state.status = "failed"
        state.final_summary = f"Context collection failed: {exc}"
        _log(state.final_summary)
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
        _log(state.final_summary)
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
    """Build the final workflow summary."""
    blocked_result = _blocking_coding_agent_result(state)
    no_experiment_results = not any(attempt.results for attempt in state.experiment_attempts)
    if no_experiment_results:
        return _no_experiment_summary(state)
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


def _no_experiment_summary(state: ReproState) -> str:
    """Explain why formal experiment commands did not run."""
    ready_to_run = state.planned_experiment and state.planned_experiment.feasibility == "ready_to_run"
    lines = [
        "Experiment plan was ready, but commands were not executed." if ready_to_run else "Experiment commands were not executed, so no reproduction metrics were produced.",
    ]
    if state.planned_experiment:
        lines += [
            "",
            f"Planned experiment feasibility: {state.planned_experiment.feasibility or 'unknown'}",
            f"Planned experiment: {state.planned_experiment.summary}",
        ]
        if state.planned_experiment.commands:
            lines += ["Planned commands:"]
            lines += [f"- {command}" for command in state.planned_experiment.commands]
        if state.planned_experiment.needs_user_input:
            lines += ["Remaining issues:"]
            lines += [f"- {item}" for item in state.planned_experiment.needs_user_input]
    if state.coding_agent_results:
        latest = state.coding_agent_results[-1]
        lines += [
            "",
            f"Latest CodingAgent status: {latest.status}",
        ]
        if latest.summary:
            lines += [f"CodingAgent summary: {latest.summary}"]
        if latest.report_path:
            lines += [f"CodingAgent report: {latest.report_path}"]
    lines += [
        "",
        "Next step: rerun and approve the ready experiment plan when you want to execute it." if ready_to_run else "Next step: fix the remaining plan validation issue or rerun with a plan that satisfies the experiment goal, then execute the experiment commands.",
    ]
    return normalize_text("\n".join(lines))


def _blocking_coding_agent_result(state: ReproState):
    """Return a CodingAgent result that blocks execution."""
    if not state.coding_agent_results:
        return None
    latest = state.coding_agent_results[-1]
    experiment_commands_ran = any(attempt.results for attempt in state.experiment_attempts)
    if latest.status != "completed" and not experiment_commands_ran:
        return latest
    return None


def _run_stage_loop(state: ReproState, stage: str, max_attempts: int) -> bool:
    """Run planning, execution, validation, and retry for a stage."""
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
                plan = plan_experiment(state)
        else:
            plan = revise_after_failure(state, stage=stage)

        if stage == "experiment":
            plan = _validate_experiment_plan(state, plan)
        _print_plan(stage, attempt, plan)
        if stage == "experiment" and plan.feasibility and plan.feasibility != "ready_to_run":
            if plan.feasibility == "needs_patch":
                if state.task.enable_coding_agent and not state.task.plan_only:
                    if _run_coding_agent_patch_cycle(state, plan):
                        plan = _validated_experiment_plan(state)
                        _print_plan(stage, attempt, plan)
                    else:
                        state.experiment_attempts.append(StageResult(stage=stage, attempt=attempt, plan=plan, results=[]))
                        save_state(state)
                        return False
                else:
                    _log("experiment needs a code patch, but CodingAgent is disabled")
                    state.experiment_attempts.append(StageResult(stage=stage, attempt=attempt, plan=plan, results=[]))
                    save_state(state)
                    return False

            # All other non-ready states (needs_config, blocked, etc.) — retry if
            # attempts remain.  The LLM will see the issues via revise_after_failure
            # and fix them in the next attempt.
            _log(f"experiment not executed because plan feasibility is {plan.feasibility}")
            state.experiment_attempts.append(StageResult(stage=stage, attempt=attempt, plan=plan, results=[]))
            save_state(state)
            if attempt < max_attempts:
                _log("retrying experiment planning with validation feedback")
                continue
            return False
        if stage == "experiment" and state.task.confirm_before_experiment and not _confirm_experiment(plan):
            _log("experiment cancelled by user before command execution")
            state.experiment_attempts.append(StageResult(stage=stage, attempt=attempt, plan=plan, results=[]))
            save_state(state)
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
    """Return whether a stage attempt succeeded."""
    if not stage_result.plan.commands:
        return True
    if stage == "probe":
        return any(result.success for result in stage_result.results)
    return stage_result.success


def _validated_experiment_plan(state: ReproState):
    """Generate and validate a fresh experiment plan."""
    plan = plan_experiment(state)
    validated = _validate_experiment_plan(state, plan)
    state.planned_experiment = validated
    return validated


def _validate_experiment_plan(state: ReproState, plan):
    """Ask the LLM to review its own plan and annotate it with any issues found."""
    try:
        review = review_experiment_plan(state, plan)
    except Exception as exc:
        _log(f"plan review skipped: {exc}")
        return plan
    validated = apply_review_to_plan(plan, review)
    if not review.get("ready"):
        _log("plan validation flagged issues")
    return validated


def _run_coding_agent_patch_cycle(state: ReproState, plan) -> bool:
    """Run CodingAgent and re-probe after a successful patch."""
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
    """Run one probe attempt after CodingAgent modifies the repo."""
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
    """Print print plan."""
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
    """Confirm confirm experiment."""
    if not plan.commands:
        return True
    try:
        answer = input("[reproagent] Run these experiment commands? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    """Build the reproagent command-line parser."""
    parser = argparse.ArgumentParser(description="LLM-first ML paper repo reproduction runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--paper", required=True, help="Paper URL, e.g. arXiv link")
    run.add_argument("--repo", required=True, help="Git repository URL")
    run.add_argument("--workspace", required=True, type=Path, help="Run workspace directory")
    run.add_argument("--repo-cache-dir", type=Path, default=None, help="Optional local cache directory for cloned paper repositories")
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
    run.add_argument("--max-coding-agent-steps", type=int, default=24, help="Maximum CodingAgent controller steps per patch attempt")
    run.add_argument("--codingagent-path", type=Path, default=None, help="Path to an external CodingAgent checkout; overrides CODINGAGENT_PATH and config")
    run.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config file; supports agents.codingagent_path")
    run.add_argument("--mirror-profile", default="none", choices=["none", "cn", "autodl"], help="Preferred dependency mirror profile for LLM environment planning")
    run.add_argument("--mirror-strict", action="store_true", help="Require environment plans to stay on the selected mirror profile instead of silently falling back to official indexes")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch the requested command."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        codingagent_path = None
        if args.enable_coding_agent or args.codingagent_path or args.config:
            codingagent_path = configured_codingagent_path(args.codingagent_path, args.config)
        task = ReproTask(
            paper_url=args.paper,
            repo_url=args.repo,
            workspace_dir=args.workspace,
            repo_cache_dir=args.repo_cache_dir,
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
            codingagent_path=codingagent_path,
            config_path=args.config,
            mirror_profile=args.mirror_profile,
            mirror_strict=args.mirror_strict,
        )
        state = run_task(task)
        print(f"status: {state.status}")
        print(f"result: {state.result_path}")


if __name__ == "__main__":
    main()
