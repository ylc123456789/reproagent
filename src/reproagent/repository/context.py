"""Build repository and paper context for the LLM."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..models import RepoContext, ReproTask
from ..runtime.hardware import collect_hardware_text

CLONE_ATTEMPTS = 3
CLONE_TIMEOUT_SECONDS = 300


def clone_repo(task: ReproTask) -> Path:
    """Clone or reuse the target repository checkout for a run."""
    repo_path = task.workspace_dir / "repo"
    log_dir = task.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if repo_path.exists():
        if _is_usable_repo(repo_path) and _repo_matches(repo_path, task.repo_url):
            return repo_path
        _remove_path(repo_path)  # stale or a different repository — re-clone

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = _repo_cache_dir(task)
    if cache_dir is not None:
        cached = _clone_from_cache(task.repo_url, repo_path, cache_dir, log_dir)
        if cached is not None:
            return cached

    latest_error = ""
    for attempt in range(1, CLONE_ATTEMPTS + 1):
        result = _run_clone(task.repo_url, repo_path)
        _write_clone_logs(log_dir, result, attempt)
        latest_error = (result.stderr or "").strip() or (result.stdout or "").strip()
        if result.returncode == 0 and _is_usable_repo(repo_path):
            _refresh_cache_from_repo(repo_path, task.repo_url, cache_dir, log_dir)
            return repo_path
        _remove_path(repo_path)
        if attempt < CLONE_ATTEMPTS:
            time.sleep(min(5 * attempt, 30))

    raise RuntimeError(f"git clone failed after {CLONE_ATTEMPTS} attempts: {latest_error}")


def setup_workspace(task: ReproTask) -> Path:
    """Prepare the repository worktree for a task.

    Exactly one repository source must be declared for a new task:
    repo_url (isolated clone), copy_from (local worktree copy preserving
    uncommitted changes), or external_repo_path (operate on an existing
    repository in place).  All three empty is only valid when the
    workspace already contains a usable repo (resume semantics).
    Conflicts raise ValueError — never guess a precedence.
    """
    mode, source = _resolve_source_mode(task)
    if mode == "isolated":
        return clone_repo(task)
    if mode == "copy":
        return _copy_worktree(source, task)
    if mode == "shared":
        return _bind_external_repo(source)
    return Path(source)  # resume: existing workspace/repo (validated above)


def workspace_mode(task: ReproTask) -> str:
    """Derive the workspace mode from the task's source fields (no validation)."""
    if task.external_repo_path:
        return "shared"
    if task.copy_from:
        return "copy"
    if task.repo_url:
        return "isolated"
    return "resume"


def _resolve_source_mode(task: ReproTask) -> tuple[str, str]:
    """Return (mode, source) for the task's single repository source."""
    declared = [
        ("isolated", task.repo_url, "repo_url"),
        ("copy", task.copy_from, "copy_from"),
        ("shared", task.external_repo_path, "external_repo_path"),
    ]
    given = [(mode, value, name) for mode, value, name in declared if value]
    if len(given) > 1:
        detail = ", ".join(f"{name}={value!r}" for _, value, name in given)
        raise ValueError(
            f"Exactly one repository source must be given, got {len(given)}: {detail}. "
            "repo_url, copy_from, and external_repo_path are mutually exclusive."
        )
    if len(given) == 1:
        mode, value, _ = given[0]
        return mode, value
    # zero sources — only resume semantics may reuse the existing workspace repo
    repo_path = task.workspace_dir / "repo"
    if _is_usable_repo(repo_path):
        return "resume", str(repo_path)
    raise ValueError(
        "No repository source given: provide one of repo_url, copy_from, or "
        "external_repo_path (all empty is only allowed when the workspace "
        f"already contains a usable repo — {repo_path} does not)."
    )


def _copy_worktree(source: str, task: ReproTask) -> Path:
    """Copy a local worktree into the task workspace, preserving uncommitted changes."""
    src = Path(source).expanduser()
    if not src.is_dir() or not _is_usable_repo(src):
        raise ValueError(f"copy_from is not a usable repository worktree: {src}")
    dst = task.workspace_dir / "repo"
    if dst.exists():
        _remove_path(dst)  # explicit source is authoritative for a fresh task
    shutil.copytree(str(src), str(dst), symlinks=True)
    return dst


def _bind_external_repo(source: str) -> Path:
    """Validate and return an existing repository to operate on in place."""
    repo = Path(source).expanduser().resolve()
    if not repo.is_dir() or not _is_usable_repo(repo):
        raise ValueError(f"external_repo_path is not a usable repository: {repo}")
    return repo


def collect_context(task: ReproTask) -> RepoContext:
    """Collect repository, hardware, paper, and README context for planning."""
    repo_path = setup_workspace(task)
    commit = _git_commit(repo_path)
    file_tree = _file_tree(repo_path, limit=500)
    readme_text = _read_readmes(repo_path)
    hardware_text = collect_hardware_text()

    ctx_dir = task.workspace_dir / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ctx_dir / "context_summary.md"
    summary = f"""# Reproduction Context

## Paper

{task.paper_url}

## Repository

{task.repo_url}

Commit: `{commit or 'unknown'}`

## Hardware

```text
{hardware_text}
```

## File Tree

```text
{file_tree}
```

## README Excerpts

{readme_text[:20000]}
"""
    summary_path.write_text(summary, encoding="utf-8")

    return RepoContext(
        repo_path=repo_path,
        commit_hash=commit,
        file_tree=file_tree,
        readme_text=readme_text[:30000],
        hardware_text=hardware_text,
        paper_url=task.paper_url,
        summary_path=summary_path,
    )


def _repo_cache_dir(task: ReproTask) -> Path | None:
    """Return the optional repository cache directory."""
    value = task.repo_cache_dir or os.environ.get("REPROAGENT_REPO_CACHE_DIR")
    return Path(value).expanduser() if value else None


def _clone_from_cache(repo_url: str, repo_path: Path, cache_dir: Path, log_dir: Path) -> Path | None:
    """Clone the target repo from a local cache when available."""
    cached_repo = _cached_repo_path(cache_dir, repo_url)
    cached_repo.parent.mkdir(parents=True, exist_ok=True)
    if not _is_usable_repo(cached_repo):
        _remove_path(cached_repo)
        result = _run_clone(repo_url, cached_repo)
        _write_named_logs(log_dir, "clone_cache_seed", result)
        if result.returncode != 0 or not _is_usable_repo(cached_repo):
            _remove_path(cached_repo)
            return None
    else:
        _refresh_cached_repo(cached_repo, log_dir)

    result = _run_clone(str(cached_repo), repo_path)
    _write_named_logs(log_dir, "clone_cache", result)
    if result.returncode == 0 and _is_usable_repo(repo_path):
        return repo_path
    _remove_path(repo_path)
    return None


def _refresh_cache_from_repo(repo_path: Path, repo_url: str, cache_dir: Path | None, log_dir: Path) -> None:
    """Best-effort copy of a successful network clone into the local cache."""
    if cache_dir is None:
        return
    cached_repo = _cached_repo_path(cache_dir, repo_url)
    if _is_usable_repo(cached_repo):
        return
    cached_repo.parent.mkdir(parents=True, exist_ok=True)
    result = _run_clone(str(repo_path), cached_repo)
    _write_named_logs(log_dir, "clone_cache_store", result)
    if result.returncode != 0 or not _is_usable_repo(cached_repo):
        _remove_path(cached_repo)


def _refresh_cached_repo(cached_repo: Path, log_dir: Path) -> None:
    """Best-effort refresh of an existing cached checkout."""
    result = subprocess.run(
        ["git", "-C", str(cached_repo), "pull", "--ff-only"],
        text=True,
        capture_output=True,
        timeout=CLONE_TIMEOUT_SECONDS,
    )
    _write_named_logs(log_dir, "clone_cache_refresh", result)


def _cached_repo_path(cache_dir: Path, repo_url: str) -> Path:
    """Return a stable cache path for a repository URL."""
    name = repo_url.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "repo"
    return cache_dir / slug


def _write_named_logs(log_dir: Path, name: str, result: subprocess.CompletedProcess[str]) -> None:
    """Write stdout/stderr logs for non-attempt clone helpers."""
    (log_dir / f"{name}.stdout").write_text(result.stdout or "", encoding="utf-8", errors="replace")
    (log_dir / f"{name}.stderr").write_text(result.stderr or "", encoding="utf-8", errors="replace")


def _run_clone(repo_url: str, repo_path: Path) -> subprocess.CompletedProcess[str]:
    """Run git clone and capture its output."""
    try:
        return subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", repo_url, str(repo_path)],
            text=True,
            capture_output=True,
            timeout=CLONE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        stderr = (stderr or "") + f"\nCommand timed out after {CLONE_TIMEOUT_SECONDS}s"
        return subprocess.CompletedProcess(exc.cmd, -1, stdout, stderr)


def _write_clone_logs(log_dir: Path, result: subprocess.CompletedProcess[str], attempt: int) -> None:
    """Write write clone logs."""
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    (log_dir / f"clone_{attempt:02d}.stdout").write_text(stdout, encoding="utf-8", errors="replace")
    (log_dir / f"clone_{attempt:02d}.stderr").write_text(stderr, encoding="utf-8", errors="replace")
    (log_dir / "clone.stdout").write_text(stdout, encoding="utf-8", errors="replace")
    (log_dir / "clone.stderr").write_text(stderr, encoding="utf-8", errors="replace")


def _repo_matches(repo_path: Path, repo_url: str) -> bool:
    """Return whether the existing checkout's origin is the requested URL.

    A usable repo without a verifiable origin (no .git, no remote) is
    treated as non-matching — never silently reuse an unverified checkout.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            text=True, capture_output=True, timeout=10,
        )
    except Exception:
        return False
    return result.returncode == 0 and _normalize_repo_url(result.stdout) == _normalize_repo_url(repo_url)


def _normalize_repo_url(url: str) -> str:
    """Compare-friendly URL form: drop trailing slash and .git suffix."""
    normalized = url.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def _is_usable_repo(repo_path: Path) -> bool:
    """Return whether a path is a usable git worktree."""
    if not repo_path.exists() or not repo_path.is_dir():
        return False
    if not _has_worktree_files(repo_path):
        return False
    git_dir = repo_path / ".git"
    if git_dir.exists():
        return _git_commit(repo_path) is not None
    return True


def _has_worktree_files(repo_path: Path) -> bool:
    """Return whether has worktree files."""
    try:
        children = [path for path in repo_path.iterdir() if path.name != ".git"]
    except OSError:
        return False
    if not children:
        return False
    markers = {"README.md", "README.rst", "README.txt", "readme.md", "setup.py", "pyproject.toml", "requirements.txt", "environment.yml", "environment.yaml"}
    if any(path.name in markers for path in children):
        return True
    return any(path.is_file() for path in children) or any(path.is_dir() for path in children)


def _remove_path(path: Path) -> None:
    """Remove a stale clone target."""
    if path.exists():
        shutil.rmtree(path)


def _git_commit(repo_path: Path) -> str | None:
    """Return the current repository commit hash."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _file_tree(repo_path: Path, limit: int = 250) -> str:
    """Build a compact repository file tree excerpt."""
    ignored = {".git", "__pycache__", ".venv", "venv", "env", "site-packages"}
    lines: list[str] = []
    for path in sorted(repo_path.rglob("*")):
        rel = path.relative_to(repo_path)
        if any(part in ignored for part in rel.parts):
            continue
        if len(lines) >= limit:
            lines.append("... truncated ...")
            break
        suffix = "/" if path.is_dir() else ""
        lines.append(str(rel) + suffix)
    return "\n".join(lines)


def _read_readmes(repo_path: Path) -> str:
    """Read read readmes."""
    names = ["README.md", "README.rst", "README.txt", "readme.md"]
    chunks: list[str] = []
    for name in names:
        path = repo_path / name
        if path.exists():
            chunks.append(f"\n# {name}\n" + path.read_text(encoding="utf-8", errors="replace"))
    docs_dir = repo_path / "docs"
    if docs_dir.exists():
        for path in sorted(docs_dir.rglob("*.md"))[:5]:
            chunks.append(f"\n# {path.relative_to(repo_path)}\n" + path.read_text(encoding="utf-8", errors="replace")[:5000])
    return "\n".join(chunks)
