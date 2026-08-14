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

    from reproagent.agent import _build_resume_goal
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


# ── R6: execution contract v1 bindings sub-schema ─────────────────

def test_bindings_repo_and_environment_sections(tmp_path):
    """Card bindings follow the contract: repo{path,origin,commit,mode}
    and environment{name,policy,certification,certified_at,audit_artifact}."""
    from reproagent.models import AgentState, EnvironmentAudit, EnvironmentInfo, RepoContext
    from reproagent.session import _read_yaml

    task = ReproTask(repo_url="https://example.invalid/org/repo.git", workspace_dir=tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    audit_out = tmp_path / "logs" / "environment_audit.stdout"
    audit_out.parent.mkdir()
    audit_out.write_text("ok", encoding="utf-8")
    state = AgentState(
        task=task,
        repo_context=RepoContext(repo_path=repo, commit_hash="abc1234"),
        environment=EnvironmentInfo(env_name="resenv_x", created=True),
        last_audit=EnvironmentAudit(success=True, summary="passed", stdout_path=audit_out),
        status="completed",
        final_summary="done",
    )
    write_session_card(state)

    card = _read_yaml(tmp_path / "session.yaml")
    repo_binding = card["bindings"]["repo"]
    assert repo_binding["path"] == str(repo)
    assert repo_binding["origin"] == "https://example.invalid/org/repo.git"
    assert repo_binding["commit"] == "abc1234"
    assert repo_binding["mode"] == "isolated"
    env_binding = card["bindings"]["environment"]
    assert env_binding["name"] == "resenv_x"
    assert env_binding["policy"] == "auto"
    assert env_binding["certification"] == "experiment"
    assert env_binding["certified_at"]
    assert env_binding["audit_artifact"] == "logs/environment_audit.stdout"
    # legacy flat keys stay for backward compatibility
    assert card["bindings"]["conda_env"] == "resenv_x"


def test_bindings_certification_none_without_audit(tmp_path):
    from reproagent.models import AgentState, EnvironmentInfo, RepoContext
    from reproagent.session import _read_yaml

    task = ReproTask(repo_url="r", workspace_dir=tmp_path)
    state = AgentState(
        task=task,
        repo_context=RepoContext(repo_path=tmp_path / "repo"),
        environment=EnvironmentInfo(env_name="resenv_y", created=True),
        status="completed",
    )
    write_session_card(state)

    env_binding = _read_yaml(tmp_path / "session.yaml")["bindings"]["environment"]
    assert env_binding["certification"] == "none"
    assert "certified_at" not in env_binding
    assert "audit_artifact" not in env_binding


def test_bindings_shared_mode_reports_shared_and_local_origin(tmp_path):
    from reproagent.models import AgentState, RepoContext
    from reproagent.session import _read_yaml

    external = tmp_path / "project" / "repos" / "foo"
    external.mkdir(parents=True)
    task = ReproTask(external_repo_path=str(external), workspace_dir=tmp_path / "ws")
    state = AgentState(
        task=task,
        repo_context=RepoContext(repo_path=external),
        status="completed",
    )
    write_session_card(state)

    repo_binding = _read_yaml(tmp_path / "ws" / "session.yaml")["bindings"]["repo"]
    assert repo_binding["mode"] == "shared"
    assert repo_binding["origin"] == "local"
    assert repo_binding["path"] == str(external)


def test_session_status_tolerates_legacy_card_without_bindings(tmp_path):
    (tmp_path / "session.yaml").write_text(
        "schema_version: 1\nsession_id: old-1\nmodule: reproagent\n"
        "kind: task_session\nstatus: completed\n",
        encoding="utf-8",
    )
    info = session_status(tmp_path)
    assert info["session"]["session_id"] == "old-1"


def test_bindings_legacy_zero_source_resume_writes_valid_mode(tmp_path):
    """Resume of a legacy zero-source task still writes the repo binding,
    and the mode is never the non-contract value 'resume'."""
    from reproagent.models import AgentState, RepoContext
    from reproagent.session import _read_yaml

    repo = tmp_path / "repo"
    repo.mkdir()
    task = ReproTask(workspace_dir=tmp_path)  # all sources empty (legacy resume)
    state = AgentState(
        task=task,
        repo_context=RepoContext(repo_path=repo),
        status="completed",
    )
    write_session_card(state)

    repo_binding = _read_yaml(tmp_path / "session.yaml")["bindings"]["repo"]
    assert repo_binding["mode"] == "isolated"  # never "resume"
    assert repo_binding["path"] == str(repo)


def test_key_artifacts_register_existing_evidence(tmp_path):
    """key_artifacts list real existing paths: result, audit, experiment logs."""
    from reproagent.models import (
        AgentObservation,
        AgentState,
        CommandResult,
        EnvironmentAudit,
        RepoContext,
    )
    from reproagent.session import _read_yaml

    logs = tmp_path / "logs"
    logs.mkdir()
    train_stdout = logs / "experiment_01_01.stdout"
    train_stdout.write_text("Epoch 1 | loss 0.5", encoding="utf-8")
    audit_out = logs / "environment_audit.stdout"
    audit_out.write_text("audit ok", encoding="utf-8")
    (tmp_path / "result.md").write_text("# result", encoding="utf-8")

    state = AgentState(
        task=ReproTask(repo_url="r", workspace_dir=tmp_path),
        repo_context=RepoContext(repo_path=tmp_path / "repo"),
        last_audit=EnvironmentAudit(success=True, summary="passed", stdout_path=audit_out),
        status="completed",
        steps=[AgentObservation(
            step=1, action="run_commands", stage_hint="experiment",
            command_results=[CommandResult(
                command="python train.py", exit_code=0,
                stdout_path=train_stdout,
                stderr_path=logs / "experiment_01_01.stderr",
                duration_seconds=1.0,
            )],
        )],
    )
    write_session_card(state)

    card = _read_yaml(tmp_path / "session.yaml")
    artifacts = card["key_artifacts"]
    by_type = {a["type"]: a for a in artifacts}
    assert set(by_type) == {"experiment_result", "environment_audit", "experiment_log"}
    assert by_type["experiment_result"]["path"] == "result.md"
    assert by_type["environment_audit"]["path"] == "logs/environment_audit.stdout"
    assert by_type["experiment_log"]["path"] == "logs/experiment_01_01.stdout"
    # every registered path exists on disk
    for artifact in artifacts:
        assert (tmp_path / artifact["path"]).exists(), artifact["path"]


def test_key_artifacts_prioritize_experiment_evidence(tmp_path):
    """A long setup/probe phase must not crowd the primary experiment log
    out of the capped key_artifacts list."""
    from reproagent.models import AgentObservation, AgentState, CommandResult, RepoContext
    from reproagent.session import _read_yaml

    logs = tmp_path / "logs"
    logs.mkdir()
    (tmp_path / "result.md").write_text("# result", encoding="utf-8")

    def make_step(step: int, stage: str, count: int) -> AgentObservation:
        results = []
        for i in range(count):
            out = logs / f"{stage}_{step:02d}_{i:02d}.stdout"
            out.write_text(f"{stage} log", encoding="utf-8")
            results.append(CommandResult(
                command=f"{stage} cmd {i}", exit_code=0,
                stdout_path=out, stderr_path=logs / f"{stage}_{step:02d}_{i:02d}.stderr",
                duration_seconds=0.1,
            ))
        return AgentObservation(step=step, action="run_commands", stage_hint=stage,
                                command_results=results)

    steps = []
    for i in range(7):
        steps.append(make_step(i + 1, "environment", 2))  # 14 probe logs fill the cap
    steps.append(make_step(8, "experiment", 1))           # the primary evidence

    state = AgentState(
        task=ReproTask(repo_url="r", workspace_dir=tmp_path),
        repo_context=RepoContext(repo_path=tmp_path / "repo"),
        status="completed",
        steps=steps,
    )
    write_session_card(state)

    artifacts = _read_yaml(tmp_path / "session.yaml")["key_artifacts"]
    experiment_paths = [a["path"] for a in artifacts
                        if "experiment_08" in a["path"]]
    assert experiment_paths, "primary experiment log was dropped by the cap"
