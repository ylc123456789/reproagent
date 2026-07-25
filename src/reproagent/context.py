"""Build repository and paper context for the LLM."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .hardware import collect_hardware_text
from .models import RepoContext, ReproTask


def clone_repo(task: ReproTask) -> Path:
    repo_path = task.workspace_dir / "repo"
    if repo_path.exists() and any(repo_path.iterdir()):
        return repo_path
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", task.repo_url, str(repo_path)],
        text=True,
        capture_output=True,
        timeout=300,
    )
    log_dir = task.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "clone.stdout").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / "clone.stderr").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return repo_path


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
