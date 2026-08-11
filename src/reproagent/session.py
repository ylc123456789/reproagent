"""Session index card (session.yaml) — write, scan, list, status.

Implements the session model from ResAgent docs/SESSION_AND_PROJECT_MODEL.md
§3 and §4.  Each workspace writes a lightweight yaml card that is the sole
cross-module contract for discoverability and resume.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from .models import AgentState


# ── write ──────────────────────────────────────────────────────────

def write_session_card(state: AgentState, *, created_at: str | None = None, **extra_bindings) -> Path:
    """Write (or overwrite) the session index card in the workspace root.

    On first write, created_at is set to the current time. On resume,
    pass the original created_at to preserve it; only updated_at is
    refreshed.
    """
    ws = state.task.workspace_dir
    ws.mkdir(parents=True, exist_ok=True)
    path = ws / "session.yaml"

    now = _utcnow()
    if created_at is None:
        # Try to preserve original created_at from an existing card
        if path.exists():
            try:
                existing = _read_yaml(path)
                created_at = existing.get("created_at", now)
            except Exception:
                created_at = now
        else:
            created_at = now

    bindings = {"conda_env": state.environment.env_name if state.environment else ""}
    if state.task.dataset_cache_dir:
        bindings["dataset_cache"] = state.task.dataset_cache_dir
    bindings.update(extra_bindings)

    # pip cache: always record the actual resolved path
    pip_cache_dir = _resolve_pip_cache(state.task.workspace_dir, state)
    if pip_cache_dir:
        bindings.setdefault("pip_cache", pip_cache_dir)

    card: dict = {
        "schema_version": 1,
        "session_id": state.task.task_id,
        "module": "reproagent",
        "kind": "task_session",
        "status": state.status,
        "created_at": created_at,
        "updated_at": now,
        "summary": state.final_summary[:500] if state.final_summary else "",
        "bindings": bindings,
        "key_artifacts": [
            {"type": "repro_result", "path": "result.md",
             "summary": state.final_summary[:200] if state.final_summary else ""},
        ],
        "resume": {
            "cli": f"reproagent resume {ws} --instruction \"...\"",
            "note": "同一工作区开新一轮 loop，注入上次结果摘要与新指令",
        },
    }
    if state.task.parent_run:
        card["parent"] = state.task.parent_run
    try:
        card["project_path"] = str(ws.resolve())
    except Exception:
        card["project_path"] = str(ws)

    _write_yaml(path, card)
    return path


def update_session_card(state: AgentState, **extra_bindings) -> Path:
    """Update the session card status and updated_at (e.g. during resume)."""
    return write_session_card(state, **extra_bindings)


# ── list / status ──────────────────────────────────────────────────

def list_sessions(root_dir: str | Path) -> list[dict]:
    """Scan a directory tree for session.yaml files and return their key fields."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    cards: list[dict] = []
    for path in sorted(root.rglob("session.yaml")):
        try:
            card = _read_yaml(path)
        except Exception:
            continue
        if not isinstance(card, dict):
            continue
        cards.append({
            "session_id": card.get("session_id", ""),
            "module": card.get("module", ""),
            "status": card.get("status", ""),
            "summary": str(card.get("summary", ""))[:200],
            "path": str(path),
            "updated_at": card.get("updated_at", ""),
        })
    return cards


def session_status(workspace_dir: str | Path) -> dict:
    """Read session.yaml + state.json from a workspace and return a summary."""
    ws = Path(workspace_dir)
    result: dict = {"session": None, "state": None}
    card_path = ws / "session.yaml"
    if card_path.exists():
        try:
            result["session"] = _read_yaml(card_path)
        except Exception:
            pass
    state_path = ws / "state.json"
    if state_path.exists():
        try:
            import json
            result["state"] = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return result


# ── helpers ────────────────────────────────────────────────────────

def _resolve_pip_cache(workspace_dir: Path, state: AgentState) -> str:
    """Resolve the actual pip cache path, matching runner._pip_cache_dir logic."""
    explicit = os.environ.get("REPROAGENT_PIP_CACHE", "").strip()
    if explicit:
        return explicit
    if state.task.dataset_cache_dir:
        return str(Path(state.task.dataset_cache_dir).parent / "pip-cache")
    return str(workspace_dir / ".cache" / "pip")


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_yaml(path: Path, data: dict) -> None:
    """Minimal yaml writer — no pyyaml dependency."""
    lines: list[str] = []
    _dict_to_lines(data, lines, indent=0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dict_to_lines(data: dict, lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            _dict_to_lines(value, lines, indent + 1)
        elif isinstance(value, list):
            if not value:
                lines.append(f"{prefix}{key}: []")
                continue
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    item_prefix = "  " * (indent + 1)
                    first = True
                    for ik, iv in item.items():
                        if first:
                            lines.append(f"{item_prefix}- {ik}: {_yaml_value(iv)}")
                            first = False
                        else:
                            lines.append(f"{item_prefix}  {ik}: {_yaml_value(iv)}")
                else:
                    lines.append(f"{'  ' * (indent + 1)}- {_yaml_value(item)}")
        else:
            lines.append(f"{prefix}{key}: {_yaml_value(value)}")


def _yaml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    # Fold all whitespace (incl. newlines) into single spaces: card fields are
    # display text, and keeping one line per field keeps both this minimal
    # reader and strict YAML parsers happy (multi-line values previously got
    # truncated on read-back and could inject spurious keys).
    s = " ".join(str(value).split())
    if ":" in s or "#" in s or s.startswith(" ") or s.endswith(" "):
        s = s.replace('"', '\\"')
        return f'"{s}"'
    return s


def _read_yaml(path: Path) -> dict:
    """Minimal yaml reader — no pyyaml dependency."""
    import re
    text = path.read_text(encoding="utf-8")
    result: dict = {}
    stack: list[tuple[int, dict | list]] = [(0, result)]
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped == "" or stripped.lstrip().startswith("#"):
            continue
        m = re.match(r"^(\s*)(?:-\s)?([^:]+?):\s*(.*)", stripped)
        if not m:
            continue
        indent = len(m.group(1))
        key = m.group(2)
        value = m.group(3).strip()
        # pop to correct indent level
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict = {}
            if isinstance(parent, list):
                parent.append(child)
            else:
                parent[key] = child
            stack.append((indent, child))
        else:
            parsed = _yaml_scalar(value)
            if isinstance(parent, list):
                parent.append(parsed)
            else:
                parent[key] = parsed
    return result


def _yaml_scalar(value: str):
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    return value.strip('"').strip("'")
