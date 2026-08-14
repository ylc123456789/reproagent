"""CodingAgent integration.

This module is the only reproagent code that knows how to locate and import
CodingAgent (the boundary below), and it orchestrates repo-local patch
requests through that boundary (run_coding_agent_for_patch below).
ReproAgent treats CodingAgent as an external dependency; callers can provide
a checkout path through CLI, environment, or config.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import shlex
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType

from ..models import CodingAgentResult, CommandPlan, ReproState
from ..runtime.environment import conda_run_flag, find_conda

ENV_VAR = "CODINGAGENT_PATH"


def configured_codingagent_path(
    cli_path: str | Path | None,
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve CodingAgent path from CLI, environment, or config.

    Priority is CLI > CODINGAGENT_PATH > config agents.codingagent_path.
    CLI and environment relative paths resolve against the current working
    directory. Config relative paths resolve against the config file directory.
    """
    source_env = env if env is not None else os.environ
    if cli_path:
        return validate_codingagent_path(cli_path, base_dir=Path.cwd())
    env_path = source_env.get(ENV_VAR)
    if env_path:
        return validate_codingagent_path(env_path, base_dir=Path.cwd())
    if config_path:
        loaded = _load_config_codingagent_path(Path(config_path))
        if loaded:
            return validate_codingagent_path(loaded, base_dir=Path(config_path).expanduser().resolve().parent)
    return None


def validate_codingagent_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Return an absolute CodingAgent checkout path or raise a clear error."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir or Path.cwd()) / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise ValueError(f"CodingAgent path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"CodingAgent path is not a directory: {candidate}")
    if _import_root_for_checkout(candidate) is None:
        raise ValueError(
            "CodingAgent path does not look like a CodingAgent checkout. "
            f"Expected {candidate / 'src' / 'coding_agent' / '__init__.py'} "
            f"or {candidate / 'coding_agent' / '__init__.py'}."
        )
    return candidate


def run_code_task(
    *,
    codingagent_path: Path | None,
    repo_path: Path,
    task_goal: str,
    constraints: list[str],
    verify_commands: list[str],
    max_steps: int,
    timeout_seconds: int,
    api_base: str,
    api_key_env: str,
    model: str,
    output_dir: Path,
    env_policy: str = "",
    env_name: str = "",
):
    """Run CodingAgent through its Python API.

    env_policy (auto/reuse_only/frozen) and env_name are execution-contract
    fields.  Older CodingAgent checkouts without them stay supported — the
    fields are simply not passed and the textual constraints remain the
    fallback (graceful degradation).
    """
    with _codingagent_api(codingagent_path) as api:
        supported = {
            field: value
            for field, value in (("env_policy", env_policy), ("env_name", env_name))
            if value and field in getattr(api.CodeTaskSpec, "model_fields", {})
        }
        spec = api.CodeTaskSpec(
            workspace_path=repo_path,
            task_goal=task_goal,
            constraints=constraints,
            verify_commands=verify_commands,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            api_base=api_base,
            api_key_env=api_key_env,
            model=model,
            output_dir=output_dir,
            **supported,
        )
        return api.run_code_task(spec)


def _load_config_codingagent_path(config_path: Path) -> str | None:
    """Load load config codingagent path."""
    path = config_path.expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        agents = data.get("agents", {}) if isinstance(data, dict) else {}
        value = agents.get("codingagent_path") if isinstance(agents, dict) else None
        return str(value) if value else None
    return _load_simple_yaml_codingagent_path(text)


def _load_simple_yaml_codingagent_path(text: str) -> str | None:
    """Load load simple yaml codingagent path."""
    in_agents = False
    agents_indent = 0
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "agents:":
            in_agents = True
            agents_indent = indent
            continue
        if in_agents and indent <= agents_indent:
            in_agents = False
        if in_agents and stripped.startswith("codingagent_path:"):
            value = stripped.split(":", 1)[1].strip().strip("\'").strip('"')
            return value or None
    return None


def _import_root_for_checkout(checkout_path: Path) -> Path | None:
    """Find the Python import root for a CodingAgent checkout."""
    src_layout = checkout_path / "src" / "coding_agent" / "__init__.py"
    flat_layout = checkout_path / "coding_agent" / "__init__.py"
    if src_layout.exists():
        return checkout_path / "src"
    if flat_layout.exists():
        return checkout_path
    return None


@contextlib.contextmanager
def _codingagent_api(codingagent_path: Path | None) -> Iterator[ModuleType]:
    """Load CodingAgent public APIs from a checkout."""
    if codingagent_path is None:
        yield importlib.import_module("coding_agent")
        return

    checkout = validate_codingagent_path(codingagent_path)
    import_root = _import_root_for_checkout(checkout)
    assert import_root is not None
    with _isolated_sys_path_import(import_root):
        yield importlib.import_module("coding_agent")


@contextlib.contextmanager
def _isolated_sys_path_import(import_root: Path) -> Iterator[None]:
    """Import a module with a temporary sys.path entry."""
    root = str(import_root)
    original_path = list(sys.path)
    removed_modules: dict[str, ModuleType] = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "coding_agent" or name.startswith("coding_agent.")
    }
    for name in removed_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, root)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "coding_agent" or name.startswith("coding_agent."):
                sys.modules.pop(name, None)
        sys.modules.update(removed_modules)
        sys.path[:] = original_path


# ── patch orchestration ──────────────────────────────────────────

def run_coding_agent_for_patch(state: ReproState, plan: CommandPlan) -> CodingAgentResult:
    """Run the external CodingAgent to produce and verify a repository patch."""
    if state.repo_context is None:
        raise RuntimeError("repo context is required before running CodingAgent")
    output_dir = state.task.workspace_dir / "patches" / f"coding_agent_{len(state.coding_agent_results) + 1:02d}"
    env_summary = _environment_summary(state)
    verify_commands = _verification_commands(state, plan)
    report = run_code_task(
        codingagent_path=state.task.codingagent_path,
        repo_path=state.repo_context.repo_path,
        task_goal=_task_goal(state, plan, env_summary),
        constraints=_constraints(env_summary),
        verify_commands=verify_commands,
        max_steps=state.task.max_coding_agent_steps,
        timeout_seconds=state.task.timeout_seconds,
        api_base=state.task.api_base,
        api_key_env=state.task.api_key_env,
        model=state.task.model or "gpt-4.1",
        output_dir=output_dir,
        # Delegated code work must never mutate the operator's environment;
        # old checkouts without the field keep the textual constraints.
        env_policy="frozen",
        env_name=state.environment.env_name if state.environment else "",
    )
    return CodingAgentResult(
        status=report.status,
        summary=report.summary,
        changed_files=report.changed_files,
        diff_path=report.diff_path,
        report_path=output_dir / "patch_report.md",
        output_dir=output_dir,
        environment_summary=env_summary,
        verification_commands=[result.command for result in report.verification_results] or verify_commands,
        residual_risks=report.residual_risks,
    )


def _task_goal(state: ReproState, plan: CommandPlan, env_summary: str) -> str:
    """Build the CodingAgent patch goal text."""
    issues = "\n".join(f"- {item}" for item in plan.needs_user_input)
    commands = "\n".join(f"- {command}" for command in plan.commands)
    return f"""Modify the repository minimally so the experiment goal can be attempted without changing research semantics.

Experiment goal:
{state.task.experiment_goal}

Current plan summary:
{plan.summary}

Validation issues to resolve:
{issues or '- none recorded'}

Planned experiment commands before patch:
{commands or '- none'}

Execution environment:
{env_summary}

Your verification commands are already wrapped by reproagent to run inside that environment. If an import, dependency, CUDA, or package-version error appears, report it as an environment issue for reproagent instead of installing packages.

Prefer the smallest code/config change that resolves the validation issues. If a requested metric is not logged, add logging rather than changing training behavior. If the goal is bounded and the repo lacks suitable controls, add a minimal CLI/config control instead of editing default full-experiment behavior.
"""


def _constraints(env_summary: str) -> list[str]:
    """Build patch constraints for CodingAgent."""
    return [
        "Do not change model architecture unless explicitly required by the task.",
        "Do not change optimizer, loss function, dataset split, or evaluation metric unless explicitly required by the task.",
        "Prefer logging/configuration changes over algorithmic changes.",
        "Keep patches minimal and easy to review.",
        "Do not write outside the repository worktree.",
        "Do not remove files, datasets, checkpoints, or caches.",
        "Do not install or upgrade dependencies; reproagent has already prepared the conda environment.",
        "Use the provided verification commands and existing environment instead of running pip/conda/apt installs.",
        "Environment context: " + env_summary.replace(chr(10), " / "),
    ]


def _environment_summary(state: ReproState) -> str:
    """Summarize the prepared runtime environment."""
    if state.environment is None:
        return "No reproagent-managed environment is available; use only repo-local inspection and report environment blockers."
    audit_details = ""
    if state.environment_audit and state.environment_audit.details:
        audit_details = chr(10) + "- Audit details: " + " / ".join(state.environment_audit.details)
    codingagent_path = str(state.task.codingagent_path) if state.task.codingagent_path else "importable default"
    return (
        f"- reproagent has already prepared the conda environment {state.environment.env_name!r} for this repository.\n"
        f"- Environment backend: {state.environment.backend}. Created this run: {state.environment.created}.\n"
        f"- CodingAgent source: {codingagent_path}.\n"
        f"- Verification commands are executed via conda run {conda_run_flag(state.environment.env_name)} {state.environment.env_name} bash -c <command>.\n"
        "- Do not install, upgrade, or remove dependencies from CodingAgent. Dependency/environment repair belongs to reproagent environment stage.\n"
        "- Your responsibility is repo-local code/config edits and verification inside the prepared environment."
        f"{audit_details}"
    )


def _verification_commands(state: ReproState, plan: CommandPlan) -> list[str]:
    """Choose verification commands for the patch attempt."""
    commands: list[str] = []
    for attempt in state.probe_attempts:
        for result in attempt.results:
            command = result.command
            if "--help" in command and command not in commands:
                commands.append(command)
    for command in plan.commands:
        lowered = command.lower()
        if ("--debug" in lowered or "--help" in lowered) and command not in commands:
            commands.append(command)
    commands = commands[:3]
    if state.environment is None:
        return commands
    return [_wrap_verify_command(command, state.environment.env_name) for command in commands]


def _wrap_verify_command(command: str, env_name: str) -> str:
    """Wrap a verification command in the prepared conda environment."""
    conda = str(find_conda())
    return " ".join([
        shlex.quote(conda),
        "run",
        conda_run_flag(env_name),
        shlex.quote(env_name),
        "bash",
        "-c",
        shlex.quote(command),
    ])
