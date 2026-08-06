"""CodingAgent integration boundary.

This module is the only reproagent code that knows how to locate and import
CodingAgent. ReproAgent treats CodingAgent as an external dependency; callers
can provide a checkout path through CLI, environment, or config.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType

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
):
    """Run CodingAgent through its Python API."""
    with _codingagent_api(codingagent_path) as api:
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
