from pathlib import Path

from reproagent.llm import _chat_completions_url
from reproagent.models import ReproTask


def test_chat_completions_url_from_base():
    assert _chat_completions_url("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1/chat/completions"


def test_chat_completions_url_accepts_full_endpoint():
    url = "https://example.com/v1/chat/completions"
    assert _chat_completions_url(url) == url


def test_task_api_defaults():
    task = ReproTask(paper_url="p", repo_url="r", workspace_dir=Path("runs/x"))
    assert task.api_base == "https://api.openai.com/v1"
    assert task.api_key_env == "OPENAI_API_KEY"
    assert task.backend == "conda"


def test_recent_logs_includes_environment_audit(tmp_path):
    from reproagent.llm import _recent_logs
    from reproagent.models import EnvironmentAudit, ReproState, ReproTask

    audit_stdout = tmp_path / "audit.stdout"
    audit_stdout.write_text("audit raw output", encoding="utf-8")
    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path),
        environment_audit=EnvironmentAudit(
            success=True,
            summary="Environment audit requires repair.",
            details=["GPU repair required"],
            has_warnings=True,
            requires_repair=True,
            stdout_path=audit_stdout,
        ),
    )

    logs = _recent_logs(state)

    assert "latest environment audit" in logs
    assert "requires_repair=True" in logs
    assert "GPU repair required" in logs
    assert "audit raw output" in logs


def test_recent_logs_include_validation_issues_and_coding_agent_result(tmp_path):
    from reproagent.llm import _recent_logs
    from reproagent.models import CodingAgentResult, CommandPlan, ReproState, ReproTask, StageResult

    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path),
    )
    state.coding_agent_results.append(CodingAgentResult(
        status="completed",
        summary="Added loss logging.",
        changed_files=["train.py"],
    ))
    state.experiment_attempts.append(StageResult(
        stage="experiment",
        attempt=1,
        plan=CommandPlan(
            stage="experiment",
            summary="bad bounded plan",
            commands=["python train.py"],
            feasibility="needs_config",
            needs_user_input=["Goal asks for a bounded run, but commands do not set epochs."],
            stop_reason="Plan validation issues: missing explicit budget.",
        ),
        results=[],
    ))

    logs = _recent_logs(state)

    assert "latest CodingAgent result" in logs
    assert "Added loss logging" in logs
    assert "validation/user issues" in logs
    assert "missing explicit budget" in logs



def test_plan_experiment_includes_goal_in_prompt(tmp_path, monkeypatch):
    from reproagent import llm
    from reproagent.models import RepoContext, ReproState, ReproTask

    captured = {}

    def fake_complete(state, prompt):
        captured["prompt"] = prompt
        return '{"stage":"experiment","summary":"s","commands":[],"assumptions":[]}'

    goal = "Run a bounded GPU MNIST ODE-Net experiment and report test accuracy."
    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal=goal)
    state = ReproState(task=task, repo_context=RepoContext(repo_path=tmp_path))
    monkeypatch.setattr(llm, "_openai_compatible_text", fake_complete)

    llm.plan_experiment(state)

    assert goal in captured["prompt"]
    assert "experiment goal" in captured["prompt"].lower()
    assert "Experiment profile" not in captured["prompt"]


def test_plan_environment_prompt_quotes_shell_sensitive_pip_specifiers(tmp_path, monkeypatch):
    from reproagent import llm
    from reproagent.models import RepoContext, ReproState, ReproTask

    captured = {}

    def fake_complete(state, prompt):
        captured["prompt"] = prompt
        return '{"stage":"environment","summary":"s","commands":[],"assumptions":[]}'

    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path),
        repo_context=RepoContext(repo_path=tmp_path),
    )
    monkeypatch.setattr(llm, "_openai_compatible_text", fake_complete)

    llm.plan_environment(state)

    assert 'pip install "numpy<2" scipy' in captured["prompt"]



def test_plan_probe_asks_for_safe_interface_discovery(tmp_path, monkeypatch):
    from reproagent import llm
    from reproagent.models import RepoContext, ReproState, ReproTask

    captured = {}

    def fake_complete(state, prompt):
        captured["prompt"] = prompt
        return '{"stage":"probe","summary":"inspect","commands":["python train.py --help"],"assumptions":[],"feasibility":"ready_to_run"}'

    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal="Run bounded training.")
    state = ReproState(task=task, repo_context=RepoContext(repo_path=tmp_path))
    monkeypatch.setattr(llm, "_openai_compatible_text", fake_complete)

    plan = llm.plan_probe(state)

    assert plan.stage == "probe"
    assert "before any real training" in captured["prompt"]
    assert "--help" in captured["prompt"]


def test_plan_experiment_requires_goal_constraints_from_probe(tmp_path, monkeypatch):
    from reproagent import llm
    from reproagent.models import CommandPlan, CommandResult, RepoContext, ReproState, ReproTask, StageResult

    captured = {}
    probe_stdout = tmp_path / "probe.stdout"
    probe_stdout.write_text("--nepochs NEPOCHS\n--gpu GPU\n", encoding="utf-8")
    probe_stderr = tmp_path / "probe.stderr"
    probe_stderr.write_text("", encoding="utf-8")

    def fake_complete(state, prompt):
        captured["prompt"] = prompt
        return '{"stage":"experiment","summary":"run bounded","commands":["python examples/odenet_mnist.py --nepochs 1 --gpu 0"],"assumptions":[],"feasibility":"ready_to_run","expected_runtime":"minutes"}'

    task = ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path, experiment_goal="Run bounded GPU MNIST.")
    state = ReproState(task=task, repo_context=RepoContext(repo_path=tmp_path))
    state.probe_attempts.append(StageResult(
        stage="probe",
        attempt=1,
        plan=CommandPlan(stage="probe", summary="help", commands=["python examples/odenet_mnist.py --help"]),
        results=[CommandResult(command="python examples/odenet_mnist.py --help", exit_code=0, stdout_path=probe_stdout, stderr_path=probe_stderr, duration_seconds=0.1)],
    ))
    monkeypatch.setattr(llm, "_openai_compatible_text", fake_complete)

    plan = llm.plan_experiment(state)

    assert "Do not assume default script parameters" in captured["prompt"]
    assert "--nepochs NEPOCHS" in captured["prompt"]
    assert "attention heads" in captured["prompt"]
    assert "Do not use cd, tee" in captured["prompt"]
    assert "does not print" in captured["prompt"]
    assert plan.commands == ["python examples/odenet_mnist.py --nepochs 1 --gpu 0"]



def test_complete_plan_repairs_invalid_regex_backslash(tmp_path, monkeypatch):
    from reproagent import llm
    from reproagent.models import RepoContext, ReproState, ReproTask

    def fake_complete(state, prompt):
        return r'''{
          "stage": "probe",
          "summary": "inspect",
          "commands": [
            "grep -nE 'argparse|parser\.add_argument|epoch' examples/odenet_mnist.py"
          ],
          "assumptions": [],
          "feasibility": "ready_to_run"
        }'''

    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path),
        repo_context=RepoContext(repo_path=tmp_path),
    )
    monkeypatch.setattr(llm, "_openai_compatible_text", fake_complete)

    plan = llm.plan_probe(state)

    assert plan.commands == ["grep -nE 'argparse|parser\\.add_argument|epoch' examples/odenet_mnist.py"]


def test_complete_plan_drops_false_needs_user_input(tmp_path, monkeypatch):
    from reproagent import llm
    from reproagent.models import RepoContext, ReproState, ReproTask

    def fake_complete(state, prompt):
        return '{"stage":"experiment","summary":"s","commands":[],"assumptions":[],"needs_user_input":false}'

    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path),
        repo_context=RepoContext(repo_path=tmp_path),
    )
    monkeypatch.setattr(llm, "_openai_compatible_text", fake_complete)

    plan = llm.plan_experiment(state)

    assert plan.needs_user_input == []


def test_final_review_normalizes_llm_text(tmp_path, monkeypatch):
    from reproagent import llm
    from reproagent.models import RepoContext, ReproState, ReproTask

    def fake_complete(state, prompt):
        return "paperâ€™s GPU鈥慳ccelerated result"

    state = ReproState(
        task=ReproTask(paper_url="paper", repo_url="repo", workspace_dir=tmp_path),
        repo_context=RepoContext(repo_path=tmp_path),
    )
    monkeypatch.setattr(llm, "_openai_compatible_text", fake_complete)

    summary = llm.final_review(state)

    assert summary == "paper's GPU-accelerated result"


def test_base_context_includes_autodl_mirror_policy(tmp_path):
    from reproagent.llm import _base_context
    from reproagent.models import RepoContext, ReproState, ReproTask

    state = ReproState(
        task=ReproTask(
            paper_url="paper",
            repo_url="repo",
            workspace_dir=tmp_path,
            experiment_goal="Run GPU MNIST.",
            mirror_profile="autodl",
            mirror_strict=True,
        ),
        repo_context=RepoContext(repo_path=tmp_path, readme_text="", file_tree=""),
    )

    context = _base_context(state)

    assert "Mirror policy: autodl (strict)" in context
    assert "AutoDL official guidance" in context
    assert "remove PyTorch official" in context
    assert "download.pytorch.org" in context
    assert "Strict mirror mode" in context
