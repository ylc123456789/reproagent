"""CLI entry point for reproagent."""
from __future__ import annotations

import argparse
from pathlib import Path

from .controller import run_controller
from .integrations.codingagent import configured_codingagent_path
from .models import ReproTask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-first ML paper repo reproduction runner")
    sub = parser.add_subparsers(dest="command", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> None:
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
        )
        state = run_controller(task)
        print(f"status: {state.status}")
        print(f"result: {state.result_path}")


if __name__ == "__main__":
    main()
