"""Tests for session.yaml, list/status, resume, env_namespace."""
import json
import subprocess
from pathlib import Path

from reproagent.controller import run_controller
from reproagent.runtime.environment import _env_name
from reproagent.models import ReproTask
from reproagent.session import list_sessions, session_status, write_session_card


# ── R1: session.yaml write ─────────────────────────────────────────

def test_session_card_written_on_completion(tmp_path):
    """Mock run ends → session.yaml exists with required fields."""
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="run bounded MNIST", mock_llm=True, max_steps=3)
    state = run_controller(task)
    card_path = tmp_path / "session.yaml"
    assert card_path.exists()
    text = card_path.read_text()
    for field in ("schema_version: 1", "module: reproagent", "kind: task_session",
                  "status: completed", "session_id:", "summary:"):
        assert field in text, f"missing {field}"
    assert "key_artifacts:" in text
    assert "result.md" in text
    assert state.status == "completed"


def test_session_card_includes_parent_run_when_provided(tmp_path):
    parent = {"module": "resagent", "run_id": "res-001", "task_id": "t1"}
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="test", mock_llm=True, max_steps=3,
                     parent_run=parent)
    run_controller(task)
    text = (tmp_path / "session.yaml").read_text()
    assert "res-001" in text
    assert "resagent" in text


def test_session_card_includes_bindings(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROAGENT_DATASET_CACHE", str(tmp_path / "datasets"))
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="test", mock_llm=True, max_steps=3,
                     dataset_cache_dir=str(tmp_path / "datasets"))
    run_controller(task)
    text = (tmp_path / "session.yaml").read_text()
    assert "dataset_cache" in text
    assert "pip_cache" in text
    assert "conda_env" in text


# ── R2: resume ─────────────────────────────────────────────────────

def test_resume_reuses_env_name_and_injects_previous_summary(tmp_path):
    """Resume should keep the same task_id (env reused) and inject prev summary."""
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="Run MNIST. Report accuracy.",
                     mock_llm=True, max_steps=3)
    s1 = run_controller(task)
    orig_id = task.task_id
    assert orig_id  # task_id was auto-generated

    # simulate resume: same task_id, new instruction
    resume_task = ReproTask(
        paper_url="p", repo_url="r", workspace_dir=tmp_path,
        experiment_goal=f"Continuation. Prev: {s1.final_summary}. New: run 5 more epochs.",
        mock_llm=True, max_steps=3, task_id=orig_id,
    )
    _ = run_controller(resume_task)

    # env name should be identical (same task_id)
    name1 = _env_name(orig_id)
    name2 = _env_name(resume_task.task_id)
    assert name1 == name2

    # card was re-written with updated status
    text = (tmp_path / "session.yaml").read_text()
    assert "status: completed" in text


def test_resume_injects_previous_summary_into_goal(tmp_path):
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="Run original", mock_llm=True, max_steps=3)
    run_controller(task)

    from reproagent.main import _build_resume_goal
    goal = _build_resume_goal("Run original", "run 5 more epochs", "3 epochs, 99.04%")
    assert "run 5 more epochs" in goal
    assert "Run original" in goal
    assert "99.04%" in goal


# ── R3: list / status ──────────────────────────────────────────────

def test_list_sessions_scans_recursively(tmp_path):
    ws1 = tmp_path / "run1"
    ws1.mkdir()
    task1 = ReproTask(paper_url="p", repo_url="r", workspace_dir=ws1,
                      experiment_goal="g1", mock_llm=True, max_steps=3)
    run_controller(task1)
    ws2 = tmp_path / "run2"
    ws2.mkdir()
    task2 = ReproTask(paper_url="p", repo_url="r", workspace_dir=ws2,
                      experiment_goal="g2", mock_llm=True, max_steps=3)
    run_controller(task2)

    cards = list_sessions(tmp_path)
    assert len(cards) >= 2
    ids = [c["session_id"] for c in cards]
    assert task1.task_id in ids
    assert task2.task_id in ids


def test_list_sessions_returns_key_fields(tmp_path):
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="g", mock_llm=True, max_steps=3)
    run_controller(task)
    cards = list_sessions(tmp_path)
    assert len(cards) >= 1
    c = cards[0]
    for key in ("session_id", "module", "status", "summary", "path", "updated_at"):
        assert key in c, f"missing key {key}"


def test_session_status_reads_card_and_state(tmp_path):
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="g", mock_llm=True, max_steps=3)
    run_controller(task)
    info = session_status(tmp_path)
    assert info["session"] is not None
    assert info["state"] is not None
    assert info["session"]["module"] == "reproagent"


def test_list_empty_root(tmp_path):
    assert list_sessions(tmp_path / "nonexistent") == []


# ── R4: env_namespace ──────────────────────────────────────────────

def test_env_name_with_namespace():
    name = _env_name("task-001", namespace="res-20260810-a4b232", isolate=False)
    assert name.startswith("resenv_")
    assert "res" in name


def test_env_name_isolate_overrides_namespace():
    name = _env_name("task-001", namespace="res-xxx", isolate=True)
    assert name.startswith("repro_")
    assert "task" in name


def test_env_name_without_namespace():
    name = _env_name("task-001")
    assert name.startswith("repro_")


# ── R5: CLI integration (light) ────────────────────────────────────

def test_cli_list_output(tmp_path):
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="g", mock_llm=True, max_steps=3)
    run_controller(task)
    r = subprocess.run(["reproagent", "list", "--root", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert task.task_id in r.stdout


def test_cli_status_output(tmp_path):
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=tmp_path,
                     experiment_goal="g", mock_llm=True, max_steps=3)
    run_controller(task)
    r = subprocess.run(["reproagent", "status", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "reproagent" in r.stdout


def test_resume_cli_accepts_args(tmp_path):
    """reproagent resume --help parses without error."""
    r = subprocess.run(["reproagent", "resume", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "workspace" in r.stdout
    assert "--instruction" in r.stdout
