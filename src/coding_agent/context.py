from __future__ import annotations

import subprocess
from pathlib import Path

from .models import CodeTaskSpec, FileSnippet, RepoContext
from .safety import BLOCKED_PATH_PARTS, BLOCKED_SUFFIXES

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def build_repo_context(spec: CodeTaskSpec, max_files: int = 24, max_bytes: int = 12_000) -> RepoContext:
    repo = spec.repo_path
    tree = _list_tree(repo)
    candidates = _rank_candidate_files(repo, tree, spec.task_goal)
    snippets = [_read_snippet(repo / path, path, max_bytes) for path in candidates[:max_files]]
    initial_diff = _git_diff(repo)
    return RepoContext(repo_path=repo, tree=tree, snippets=snippets, initial_diff=initial_diff)


def _list_tree(repo: Path, limit: int = 500) -> list[str]:
    paths: list[str] = []
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo).as_posix()
        if _is_blocked(rel):
            continue
        if path.is_file():
            paths.append(rel)
        if len(paths) >= limit:
            break
    return paths


def _rank_candidate_files(repo: Path, tree: list[str], task_goal: str) -> list[str]:
    keywords = {word.lower() for word in task_goal.replace("_", " ").replace("-", " ").split() if len(word) >= 4}
    scored: list[tuple[int, str]] = []
    for rel in tree:
        suffix = Path(rel).suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            continue
        score = 0
        lower_rel = rel.lower()
        if Path(rel).name.lower() in {"readme.md", "pyproject.toml", "setup.py", "requirements.txt"}:
            score += 5
        score += sum(2 for keyword in keywords if keyword in lower_rel)
        try:
            text = (repo / rel).read_text(encoding="utf-8", errors="ignore")[:20_000].lower()
        except OSError:
            text = ""
        score += sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scored.append((score, rel))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if scored:
        return [rel for _, rel in scored]
    return [rel for rel in tree if Path(rel).suffix.lower() in TEXT_SUFFIXES]


def _read_snippet(path: Path, rel: str, max_bytes: int) -> FileSnippet:
    data = path.read_bytes()
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="ignore")
    return FileSnippet(path=rel, text=text, truncated=truncated)


def _is_blocked(rel: str) -> bool:
    path = Path(rel)
    if set(path.parts) & BLOCKED_PATH_PARTS:
        return True
    return path.suffix.lower() in BLOCKED_SUFFIXES


def _git_diff(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--no-ext-diff"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout
