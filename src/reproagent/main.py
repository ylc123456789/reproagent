"""CLI entry point for reproagent."""
from __future__ import annotations

import argparse
from pathlib import Path

from .controller import run_controller
from .integrations.codingagent import configured_codingagent_path
from .models import ReproTask
from .session import list_sessions, session_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-first ML paper repo reproduction runner")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run = sub.add_parser("run")
    run.add_argument("--paper", required=True, help="Paper URL")
    run.add_argument("--repo", required=True, help="Git repository URL")
    run.add_argument("--workspace", required=True, type=Path, help="Run workspace directory")
    run.add_argument("--repo-cache-dir", type=Path, default=None, help="Optional local cache directory for cloned repos")
    run.add_argument("--mock-llm", action="store_true", help="Use deterministic mock LLM for local tests")
    run.add_argument("--model", default=None, help="OpenAI-compatible model name")
    run.add_argument("--api-base", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var with the API key")
    run.add_argument("--backend", default="conda", choices=["conda"], help="Execution backend")
    run.add_argument("--python-version", default="3.10", help="Python version for conda env")
    run.add_argument("--timeout", type=int, default=1800, help="Timeout per command batch (seconds)")
    run.add_argument("--max-steps", type=int, default=30, help="Max agent steps before force-finish")
    run.add_argument("--experiment-goal", required=True, help="Concrete reproduction goal")
    run.add_argument("--confirm-before-experiment", action="store_true",
                     help="Ask for confirmation before running experiment commands")
    run.add_argument("--dataset-cache", default="",
                     help="Shared dataset cache directory (torchvision/HF/torch.hub will auto-cache here)")
    run.add_argument("--enable-coding-agent", action="store_true", help="Allow CodingAgent to modify repo code")
    run.add_argument("--max-coding-agent-steps", type=int, default=24, help="Max CodingAgent steps per patch")
    run.add_argument("--codingagent-path", type=Path, default=None, help="Path to CodingAgent checkout")
    run.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config file")
    run.add_argument("--mirror-profile", default="none", choices=["none", "cn", "autodl"], help="Dependency mirror profile")
    run.add_argument("--mirror-strict", action="store_true", help="Require mirror-only package sources")
    run.add_argument("--env-namespace", default="", help="Condense env name for project sharing (e.g. res-xxx)")
    run.add_argument("--isolate-env", action="store_true", help="Force per-task env even with env-namespace")

    # --- resume ---
    resume = sub.add_parser("resume")
    resume.add_argument("workspace", type=Path, help="Path to an existing workspace directory")
    resume.add_argument("--instruction", required=True, help="New instruction for this continuation")
    resume.add_argument("--max-steps", type=int, default=None, help="Override max steps (default: keep original)")
    resume.add_argument("--model", default=None, help="Override model")
    resume.add_argument("--api-base", default=None, help="Override API base")
    resume.add_argument("--api-key-env", default=None, help="Override API key env var")
    resume.add_argument("--mock-llm", action="store_true", help="Use mock LLM")
    resume.add_argument("--timeout", type=int, default=None, help="Override timeout")

    # --- list ---
    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--root", type=Path, required=True, help="Root directory to scan for session.yaml files")

    # --- status ---
    status_cmd = sub.add_parser("status")
    status_cmd.add_argument("workspace", type=Path, help="Path to a workspace directory")
    return parser


def _build_run_task(args) -> ReproTask:
    codingagent_path = None
    if args.enable_coding_agent or args.codingagent_path or args.config:
        codingagent_path = configured_codingagent_path(args.codingagent_path, args.config)
    return ReproTask(
        paper_url=args.paper,
        repo_url=args.repo,
        workspace_dir=args.workspace,
        repo_cache_dir=args.repo_cache_dir,
        timeout_seconds=args.timeout,
        max_steps=args.max_steps,
        mock_llm=args.mock_llm,
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        backend=args.backend,
        python_version=args.python_version,
        experiment_goal=args.experiment_goal,
        enable_coding_agent=args.enable_coding_agent,
        max_coding_agent_steps=args.max_coding_agent_steps,
        codingagent_path=codingagent_path,
        config_path=args.config,
        mirror_profile=args.mirror_profile,
        mirror_strict=args.mirror_strict,
        confirm_before_experiment=args.confirm_before_experiment,
        dataset_cache_dir=args.dataset_cache,
        env_namespace=getattr(args, "env_namespace", ""),
        isolate_env=getattr(args, "isolate_env", False),
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        task = _build_run_task(args)
        state = run_controller(task)
        print(f"status: {state.status}")
        print(f"result: {state.result_path}")

    elif args.command == "resume":
        _cmd_resume(args)

    elif args.command == "list":
        cards = list_sessions(args.root)
        if not cards:
            print("No sessions found.")
        else:
            print(f"{'SESSION ID':<40} {'STATUS':<16} {'SUMMARY'}")
            print("-" * 80)
            for c in cards:
                print(f"{c['session_id']:<40} {c['status']:<16} {c['summary'][:60]}")
    elif args.command == "status":
        info = session_status(args.workspace)
        import json
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))


def _cmd_resume(args) -> None:
    """Resume a previous task in the same workspace."""
    import json as _json

    ws = Path(args.workspace)
    state_path = ws / "state.json"
    if not state_path.exists():
        raise SystemExit(f"No state.json found in {ws} — cannot resume.")

    state_data = _json.loads(state_path.read_text(encoding="utf-8"))
    orig_task = state_data.get("task", {})
    prev_summary = state_data.get("final_summary", "")

    task = ReproTask(
        paper_url=orig_task.get("paper_url", ""),
        repo_url=orig_task.get("repo_url", ""),
        workspace_dir=ws,
        repo_cache_dir=Path(orig_task["repo_cache_dir"]) if orig_task.get("repo_cache_dir") else None,
        task_id=orig_task.get("task_id", ""),  # same task_id → env reused
        timeout_seconds=args.timeout or orig_task.get("timeout_seconds", 1800),
        max_steps=args.max_steps or orig_task.get("max_steps", 30),
        mock_llm=args.mock_llm or orig_task.get("mock_llm", False),
        model=args.model or orig_task.get("model"),
        api_base=args.api_base or orig_task.get("api_base", "https://api.openai.com/v1"),
        api_key_env=args.api_key_env or orig_task.get("api_key_env", "OPENAI_API_KEY"),
        experiment_goal=_build_resume_goal(orig_task.get("experiment_goal", ""), args.instruction, prev_summary),
        enable_coding_agent=orig_task.get("enable_coding_agent", False),
        max_coding_agent_steps=orig_task.get("max_coding_agent_steps", 24),
        codingagent_path=Path(orig_task["codingagent_path"]) if orig_task.get("codingagent_path") else None,
        mirror_profile=orig_task.get("mirror_profile", "none"),
        mirror_strict=orig_task.get("mirror_strict", False),
        dataset_cache_dir=orig_task.get("dataset_cache_dir", ""),
        env_namespace=orig_task.get("env_namespace", ""),
        isolate_env=orig_task.get("isolate_env", False),
        parent_run=orig_task.get("parent_run"),
    )
    from .models import AgentState
    old_state = AgentState.model_validate(state_data)
    state = run_controller(task, resume_state=old_state)
    print(f"status: {state.status}")
    print(f"result: {state.result_path}")


def _build_resume_goal(original_goal: str, instruction: str, prev_summary: str) -> str:
    """Build the experiment goal for a resume run, incorporating previous results."""
    parts = [f"## Continuation of previous task\n\nNew instruction: {instruction}"]
    if original_goal:
        parts.append(f"\nOriginal goal: {original_goal}")
    if prev_summary:
        parts.append(f"\nPrevious results (summary): {prev_summary[:2000]}")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
