"""Deterministic dataset-cache bridging.

Why this exists: scripts hardcode relative dataset roots (e.g.
``datasets.MNIST('../data')``). Relative paths resolve against the process
cwd (the repo root where commands execute), NOT the script's location —
multi-step path reasoning the LLM reliably gets wrong. And even when it gets
it right, the runner's parent-directory-traversal safety rule forbids the
``../`` symlink command that would fix it. So the only correct place to
bridge hardcoded roots to the shared cache is here, in framework code.

Design (see ResAgent docs/HANDOVER for the incident analysis):
    1. scan  — regex the cloned repo for hardcoded dataset roots
    2. resolve — turn them into absolute paths against the execution cwd
    3. link  — pre-create symlinks from the resolved locations to the cache
    4. report — render an absolute-path mapping for the LLM prompt

Graceful degradation: if anything is unclear (no cache, no matches, target
exists), we do nothing — behaviour falls back to "LLM downloads as before".
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ── detection rules ─────────────────────────────────────────────────────────
# Each rule: (name, regex with capture group for the path, optional dataset group).
# Regexes are applied to whole file text; [^)] spans newlines so multi-line
# calls are matched.

_TORCHVISION_POSITIONAL = re.compile(
    r"datasets\.([A-Za-z0-9_]+)\(\s*['\"]([^'\"]+)['\"]"
)
_TORCHVISION_ROOT_KW = re.compile(
    r"datasets\.([A-Za-z0-9_]+)\([^)]*?\broot\s*=\s*['\"]([^'\"]+)['\"]"
)
_GENERIC_KW = re.compile(
    r"\b(?:data_?dir|data_?root|datadir)\s*=\s*['\"]([^'\"]+)['\"]"
)
_ARGPARSE_DEFAULT = re.compile(
    r"add_argument\(\s*['\"](?:--data(?:[-_]?dir)?|--dataset(?:[-_]?dir)?)['\"]"
    r"[^)]*?default\s*=\s*['\"]([^'\"]+)['\"]"
)

_SKIP_DIR_PARTS = {".git", "__pycache__", ".venv", "venv", "env", "site-packages"}


def _is_usable_path(raw: str) -> bool:
    """Filter out values that are not plain relative local paths."""
    if not raw or raw in (".", "./"):
        return False
    if raw.startswith(("/", "~")):
        return False                      # absolute — symlinks can't help
    if "://" in raw or raw.startswith("http"):
        return False                      # URL
    if any(ch in raw for ch in ("{", "}", "$", "%")):
        return False                      # f-string / env / format interpolation
    return True


def scan_dataset_roots(repo_path: Path) -> list[dict]:
    """Scan repo .py files for hardcoded dataset roots.

    Returns refs: {file, line, declared, resolved, dataset} where `resolved`
    is the absolute path the declared root resolves to at execution time
    (cwd = repo root). Only paths inside the workspace are kept.
    """
    repo_path = Path(repo_path)
    workspace_dir = repo_path.parent
    refs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(py: Path, pos: int, declared: str, dataset: str | None, text: str):
        if not _is_usable_path(declared):
            return
        resolved = (repo_path / declared).resolve()
        try:
            resolved.relative_to(workspace_dir.resolve())
        except ValueError:
            return  # escapes the workspace — never link outside
        if resolved in (repo_path.resolve(), workspace_dir.resolve()):
            return  # never link the repo root or workspace root themselves
        key = (str(resolved), dataset or "")
        if key in seen:
            return
        seen.add(key)
        refs.append({
            "file": str(py.relative_to(repo_path)),
            "line": text.count("\n", 0, pos) + 1,
            "declared": declared,
            "resolved": str(resolved),
            "dataset": dataset,
        })

    for py in sorted(repo_path.rglob("*.py")):
        if any(part in _SKIP_DIR_PARTS for part in py.parts):
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _TORCHVISION_POSITIONAL.finditer(text):
            _add(py, m.start(), m.group(2), m.group(1), text)
        for m in _TORCHVISION_ROOT_KW.finditer(text):
            _add(py, m.start(), m.group(2), m.group(1), text)
        for m in _GENERIC_KW.finditer(text):
            _add(py, m.start(), m.group(1), None, text)
        for m in _ARGPARSE_DEFAULT.finditer(text):
            _add(py, m.start(), m.group(1), None, text)
    return refs


# ── link preparation ────────────────────────────────────────────────────────

def _cache_hit(cache: Path, dataset: str | None) -> bool | None:
    """Whether the cache appears to contain this dataset (None = unknown)."""
    if dataset:
        for candidate in (cache / dataset / "raw", cache / dataset):
            if candidate.is_dir() and any(candidate.iterdir()):
                return True
        return False
    return None


def prepare_dataset_links(
    *,
    repo_path: Path,
    workspace_dir: Path,
    cache_root: str,
    allowed_write_root: Path | None = None,
) -> list[dict]:
    """Scan, resolve, and pre-create symlinks into the dataset cache.

    allowed_write_root restricts where symlinks may be created: resolved
    paths outside it are skipped and reported instead of linked (shared
    mode operates on an external repo — never create links beside it).
    When None, the historical behaviour (any path inside repo_path.parent)
    is kept. Returns the annotated ref list (also persisted on
    AgentState.dataset_links and rendered into prompts). Never raises —
    cache bridging is best-effort.
    """
    refs = scan_dataset_roots(repo_path)
    if not cache_root:
        for r in refs:
            r.update(link="no_cache", cache_hit=None)
        return refs

    cache = Path(cache_root)
    if not cache.is_dir():
        for r in refs:
            r.update(link="no_cache", cache_hit=None)
        return refs

    write_root = Path(allowed_write_root).resolve() if allowed_write_root is not None else None

    linked_paths: set[str] = set()
    for r in refs:
        resolved = Path(r["resolved"])
        r["cache_hit"] = _cache_hit(cache, r["dataset"])

        if write_root is not None:
            try:
                resolved.relative_to(write_root)
            except ValueError:
                r["link"] = "outside_write_root"  # escapes the allowed root — never link
                continue

        if str(resolved) in linked_paths:
            r["link"] = "created"      # another ref already linked this path
            continue
        if os.path.lexists(resolved):
            r["link"] = "exists"       # never clobber real dirs / existing links
            continue

        # Granularity: if the resolved path's basename names a dataset dir in
        # the cache (e.g. ./data/wikitext-2 vs cache/wikitext-2), link to that
        # subdir; otherwise link to the cache root (torchvision layout:
        # root/<DatasetName>/raw must resolve into the cache).
        target = cache / resolved.name
        if not target.is_dir():
            target = cache
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(target), str(resolved))
            linked_paths.add(str(resolved))
            r["link"] = "created"
            r["link_target"] = str(target)
        except OSError:
            r["link"] = "failed"
    return refs


# ── prompt rendering ────────────────────────────────────────────────────────

def render_dataset_block(task, repo_path: Path | None, links: list[dict]) -> str:
    """Render the dataset-cache block for prompts. Empty string if N/A."""
    cache = getattr(task, "dataset_cache_dir", "") or ""
    if not cache:
        return ""

    lines = [
        "\n## Dataset cache (resolved by the system — do NOT re-derive paths)",
        f"Commands run with cwd = {repo_path} (the repository root). Relative dataset "
        "paths in scripts resolve against THIS directory, not the script's location.",
        f"Cache root: {cache} (TORCH_HOME / HF_HOME etc. already point here).",
    ]
    if not links:
        lines.append(
            "No hardcoded dataset paths were detected in the repo. If a script "
            "downloads data, prefer directing it into the cache root."
        )
        return "\n".join(lines)

    for r in links:
        lines.append(f"- {r['file']}:{r['line']}  root='{r['declared']}'")
        lines.append(f"    resolves to: {r['resolved']}")
        if r.get("link") == "created":
            lines.append(f"    symlink: {r['resolved']} -> {r.get('link_target', cache)} [pre-created]")
        elif r.get("link") == "exists":
            lines.append("    symlink: path already exists (left as-is)")
        elif r.get("link") == "no_cache":
            lines.append("    symlink: none (cache unavailable)")
        elif r.get("link") == "outside_write_root":
            lines.append("    symlink: skipped (resolves outside the allowed write root — no link created)")
        if r.get("cache_hit") is True:
            lines.append("    cache: HIT — do NOT re-download; `download=True` will no-op.")
        elif r.get("cache_hit") is False:
            lines.append("    cache: MISS — first run downloads into the cache for future reuse.")
    lines.append("Never re-download a dataset whose cache is a HIT.")
    return "\n".join(lines)
