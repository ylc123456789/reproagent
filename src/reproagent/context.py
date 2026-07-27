"""Build repository and paper context for the LLM."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .hardware import collect_hardware_text
from .models import RepoContext, ReproTask

CLONE_ATTEMPTS = 3
CLONE_TIMEOUT_SECONDS = 300


def clone_repo(task: ReproTask) -> Path:
    repo_path = task.workspace_dir / "repo"
    log_dir = task.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if repo_path.exists():
        if _is_usable_repo(repo_path):
            return repo_path
        _remove_path(repo_path)

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    latest_error = ""
    for attempt in range(1, CLONE_ATTEMPTS + 1):
        result = _run_clone(task.repo_url, repo_path)
        _write_clone_logs(log_dir, result, attempt)
        latest_error = (result.stderr or "").strip() or (result.stdout or "").strip()
        if result.returncode == 0 and _is_usable_repo(repo_path):
            return repo_path
        _remove_path(repo_path)
        if attempt < CLONE_ATTEMPTS:
            time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"git clone failed after {CLONE_ATTEMPTS} attempts: {latest_error}")


def collect_context(task: ReproTask) -> RepoContext:
    repo_path = clone_repo(task)
    commit = _git_commit(repo_path)
    file_tree = _file_tree(repo_path)
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


def _run_clone(repo_url: str, repo_path: Path) -> subprocess.CompletedProcess[str]:
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
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    (log_dir / f"clone_{attempt:02d}.stdout").write_text(stdout, encoding="utf-8", errors="replace")
    (log_dir / f"clone_{attempt:02d}.stderr").write_text(stderr, encoding="utf-8", errors="replace")
    (log_dir / "clone.stdout").write_text(stdout, encoding="utf-8", errors="replace")
    (log_dir / "clone.stderr").write_text(stderr, encoding="utf-8", errors="replace")


def _is_usable_repo(repo_path: Path) -> bool:
    if not repo_path.exists() or not repo_path.is_dir():
        return False
    if not _has_worktree_files(repo_path):
        return False
    git_dir = repo_path / ".git"
    if git_dir.exists():
        return _git_commit(repo_path) is not None
    return True


def _has_worktree_files(repo_path: Path) -> bool:
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
    if path.exists():
        shutil.rmtree(path)


def _git_commit(repo_path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _file_tree(repo_path: Path, limit: int = 250) -> str:
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
