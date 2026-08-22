import json

from reproagent.report import _clean_text


def test_clean_text_removes_common_mojibake():
    text = "Smoke Test 鈥?Spiral ODE and paper鈥檚 metrics"

    cleaned = _clean_text(text)

    assert "鈥" not in cleaned
    assert "paper's" in cleaned


def test_clean_text_repairs_medium_run_mojibake_samples():
    text = "GPU鈥慳ccelerated ODE鈥慛et time鈥憇eries full鈥憇cale torch鈮?.3.0"

    cleaned = _clean_text(text)

    assert "鈥" not in cleaned
    assert "GPU-accelerated" in cleaned
    assert "ODE-Net" in cleaned
    assert "time-series" in cleaned
    assert "full-scale" in cleaned
    assert "torch>=.3.0" in cleaned


def test_clean_text_repairs_latin1_utf8_mojibake():
    text = "paperâ€™s GPUâ€“accelerated run"

    cleaned = _clean_text(text)

    assert cleaned == "paper's GPU-accelerated run"


def test_agent_result_writes_structured_result_and_freezes_evidence(tmp_path):
    from reproagent.models import AgentState, RepoContext, ReproTask
    from reproagent.report import write_agent_result

    workspace = tmp_path / "run"
    repo = workspace / "repo"
    output = repo / "outputs" / "metrics.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"accuracy": 0.91}', encoding="utf-8")
    state = AgentState(
        task=ReproTask(workspace_dir=workspace),
        repo_context=RepoContext(repo_path=repo),
        status="completed",
        final_summary="bounded run completed",
        structured_result={
            "metrics": {"accuracy": 0.91},
            "parameters": {"epochs": 3},
            "deviations": ["short run"],
            "evidence_files": ["outputs/metrics.json"],
        },
    )

    report = write_agent_result(state)
    result = json.loads((workspace / "result.json").read_text(encoding="utf-8"))

    assert result["schema"] == "repro_result_v1"
    assert result["metrics"] == {"accuracy": 0.91}
    assert result["parameters"] == {"epochs": 3}
    assert result["evidence"][0]["path"] == "evidence/repo/outputs/metrics.json"
    assert (workspace / result["evidence"][0]["path"]).read_text() == '{"accuracy": 0.91}'
    assert "Frozen Evidence" in report.read_text(encoding="utf-8")
    persisted = json.loads((workspace / "state.json").read_text(encoding="utf-8"))
    assert persisted["produced_files"]["result_json"].endswith("result.json")


def test_agent_result_rejects_evidence_outside_repo_and_workspace(tmp_path):
    from reproagent.models import AgentState, RepoContext, ReproTask
    from reproagent.report import write_agent_result

    workspace = tmp_path / "run"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("not experiment evidence", encoding="utf-8")
    state = AgentState(
        task=ReproTask(workspace_dir=workspace),
        repo_context=RepoContext(repo_path=repo),
        status="completed",
        structured_result={"evidence_files": [str(outside)]},
    )

    write_agent_result(state)
    result = json.loads((workspace / "result.json").read_text(encoding="utf-8"))

    assert result["evidence"] == []
    assert "outside the repository/workspace" in result["warnings"][0]


def test_separate_attempt_workspaces_preserve_same_named_evidence(tmp_path):
    from reproagent.models import AgentState, RepoContext, ReproTask
    from reproagent.report import write_agent_result

    repo = tmp_path / "shared-repo"
    metric_file = repo / "final_metrics.json"
    repo.mkdir()

    def finish_attempt(name: str, value: float) -> str:
        metric_file.write_text(json.dumps({"accuracy": value}), encoding="utf-8")
        workspace = tmp_path / name
        state = AgentState(
            task=ReproTask(workspace_dir=workspace),
            repo_context=RepoContext(repo_path=repo),
            status="completed",
            structured_result={
                "metrics": {"accuracy": value},
                "evidence_files": ["final_metrics.json"],
            },
        )
        write_agent_result(state)
        result = json.loads((workspace / "result.json").read_text(encoding="utf-8"))
        return (workspace / result["evidence"][0]["path"]).read_text(encoding="utf-8")

    first = finish_attempt("attempt-1", 0.8)
    second = finish_attempt("attempt-2", 0.7)

    assert json.loads(first)["accuracy"] == 0.8
    assert json.loads(second)["accuracy"] == 0.7
