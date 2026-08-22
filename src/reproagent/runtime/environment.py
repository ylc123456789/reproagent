"""Conda execution backend.

The MVP is conda-first: every task gets one isolated conda environment. LLM
commands stay plain shell commands; this module wraps them with `conda run`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..models import AgentState, EnvironmentInfo


def ensure_environment(state: AgentState) -> EnvironmentInfo:
    """Create or reuse the task conda environment."""
    assert state.repo_context is not None
    conda = find_conda()
    if conda is None:
        raise RuntimeError("conda was not found. Set REPROAGENT_CONDA_EXE or install Miniconda/Anaconda so conda is on PATH.")

    logs = state.task.workspace_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "conda_setup.stdout"
    stderr_path = logs / "conda_setup.stderr"

    if state.task.reuse_mode == "content_addressed":
        # Content identity takes precedence over any env_name hint: in this
        # mode env_name is only a ResAgent-injected CANDIDATE, never a
        # direct binding. The fingerprint always decides.
        if not state.task.resource_root:
            raise RuntimeError("reuse_mode=content_addressed requires resource_root")
        return _ensure_content_addressed(state, conda, stdout_path, stderr_path)

    if state.task.env_name:
        # Legacy explicit binding: use the referenced environment unchanged.
        # Never create a substitute; unresolvable references fail loudly.
        if _find_conda_env(conda, state.task.env_name) is None:
            raise RuntimeError(
                f"Explicit environment {state.task.env_name!r} was not found. "
                "Provide an existing conda env name or absolute prefix, or "
                "leave env_name empty to let reproagent create one."
            )
        stdout_path.write_text(f"explicit environment bound: {state.task.env_name}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return EnvironmentInfo(
            env_name=state.task.env_name,
            created=False,
            setup_stdout_path=stdout_path,
            setup_stderr_path=stderr_path,
        )

    env_name = _env_name(state.task.task_id, namespace=state.task.env_namespace, isolate=state.task.isolate_env)

    if _conda_env_exists(conda, env_name):
        info = EnvironmentInfo(env_name=env_name, created=False, setup_stdout_path=stdout_path, setup_stderr_path=stderr_path)
        stdout_path.write_text(f"conda env already exists: {env_name}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return info

    env_file = _find_environment_yml(state.repo_context.repo_path)
    if env_file is not None:
        cmd = [conda, "env", "create", "-n", env_name, "-f", str(env_file)]
        used_yml = True
    else:
        cmd = [conda, "create", "-n", env_name, f"python={state.task.python_version}", "-y"]
        used_yml = False

    result = _run_conda_setup_with_retries(
        cmd,
        cwd=state.repo_context.repo_path,
        timeout=state.task.timeout_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if result.returncode != 0:
        raise RuntimeError(f"conda environment setup failed for {env_name}; see {stderr_path}")

    return EnvironmentInfo(
        env_name=env_name,
        created=True,
        used_environment_yml=used_yml,
        setup_command=" ".join(cmd),
        setup_stdout_path=stdout_path,
        setup_stderr_path=stderr_path,
    )



def _run_conda_setup_with_retries(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
    attempts: int = 3,
    delay_seconds: float = 3.0,
) -> subprocess.CompletedProcess[str]:
    """Run conda setup commands with transient-error retries."""
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
        last_result = result
        stdout_chunks.append(f"===== conda setup attempt {attempt}/{attempts} =====\n")
        stdout_chunks.append(result.stdout or "")
        stderr_chunks.append(f"===== conda setup attempt {attempt}/{attempts} =====\n")
        stderr_chunks.append(result.stderr or "")
        stdout_path.write_text("".join(stdout_chunks), encoding="utf-8", errors="replace")
        stderr_path.write_text("".join(stderr_chunks), encoding="utf-8", errors="replace")
        if result.returncode == 0:
            return result
        if attempt < attempts and _is_transient_conda_setup_error(result.stderr or ""):
            time.sleep(delay_seconds)
            continue
        return result
    assert last_result is not None
    return last_result


def _is_transient_conda_setup_error(stderr: str) -> bool:
    """Return whether conda setup failed for a transient reason."""
    lowered = stderr.lower()
    markers = (
        "condahttperror",
        "http 502",
        "http 503",
        "http 504",
        "bad gateway",
        "gateway timeout",
        "remote server error",
        "connection aborted",
        "connection reset",
        "read timed out",
    )
    return any(marker in lowered for marker in markers)



def conda_run_flag(env_ref: str) -> str:
    """Return the conda run flag for an environment reference.

    One documented rule: absolute paths are conda PREFIXES (`-p`);
    everything else is a NAME (`-n`). Shared by command execution, the
    environment audit, and CodingAgent verification wrapping.
    """
    return "-p" if Path(env_ref).expanduser().is_absolute() else "-n"


def build_backend_command(env_name: str, command: str, conda: str | None = None) -> list[str]:
    """Wrap a plain shell command so it runs inside the task conda env."""
    conda_exe = conda or find_conda() or "conda"
    return [conda_exe, "run", "--no-capture-output", conda_run_flag(env_name), env_name,
            "bash", "-o", "pipefail", "-c", command]


def find_conda() -> str | None:
    """Find conda from explicit config, PATH, or common install locations."""
    configured = os.environ.get("REPROAGENT_CONDA_EXE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    found = shutil.which("conda")
    if found:
        return found
    for candidate in (
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
        Path("/opt/conda/bin/conda"),
    ):
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def _env_name(task_id: str, namespace: str = "", isolate: bool = False) -> str:
    """Create a unique conda environment name.

    When namespace is provided (e.g. a ResAgent run_id), the env is shared
    across tasks within that project: ``resenv_<sanitized_namespace>``.
    When isolate_env is True, falls back to per-task naming even with a
    namespace.
    """
    if not isolate and namespace:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", namespace)
        return f"resenv_{safe}"[:80]
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", task_id)
    return f"repro_{safe}"[:80]


def _find_environment_yml(repo_path: Path) -> Path | None:
    """Find a repository-provided conda environment file."""
    for name in ("environment.yml", "environment.yaml", "conda.yml", "conda.yaml"):
        path = repo_path / name
        if path.exists():
            return path
    return None


def _conda_env_exists(conda: str, env_name: str) -> bool:
    """Return whether a conda environment already exists."""
    return _find_conda_env(conda, env_name) is not None


def _find_conda_env(conda: str, env_ref: str) -> str | None:
    """Resolve an environment reference to a conda env path, or None.

    One documented rule: an absolute path is matched as a conda PREFIX;
    anything else is matched as an environment NAME.
    """
    result = subprocess.run([conda, "env", "list", "--json"], text=True, capture_output=True, timeout=60)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    ref = Path(env_ref).expanduser()
    for env_path in data.get("envs", []):
        candidate = Path(env_path)
        if ref.is_absolute():
            if candidate == ref or candidate.resolve() == ref.resolve():
                return env_path
        elif candidate.name == env_ref:
            return env_path
    return None


# ── content-addressed reuse or create (milestone-2) ───────────────

def task_run_id(task) -> str:
    """ResAgent run_id from the task's parent_run, '' when absent.

    Shared by this module and env_manager (manifest created_by / usage /
    audit audited_by) so the orchestrator run is recorded consistently.
    """
    parent = task.parent_run or {}
    return str(parent.get("run_id", ""))


def _git_head(repo_path: Path) -> str:
    """git HEAD of the repository, defensively ('' when unavailable)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True, capture_output=True, timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _drift_blocker(identifier: str, manifest_file: Path, expected, actual) -> RuntimeError:
    """Structured blocker per the §6.4 operator ruling: refuse, never auto-
    recreate (env_id collides) and never repair in place (global state)."""
    return RuntimeError(
        f"environment drift detected for {identifier} — refusing blind reuse.\n"
        f"manifest: {manifest_file}\n"
        f"expected resolved_fingerprint: {expected}\n"
        f"actual   resolved_fingerprint: {actual}\n"
        "The manifest has been marked drifted. Repair or recreate decisions "
        "belong to M2-P3 (ResAgent resource selection)."
    )


def _ensure_content_addressed(state, conda: str, stdout_path, stderr_path) -> EnvironmentInfo:
    """Exact reuse or create by content identity.

    Deterministic: spec -> spec_fingerprint -> env_id -> manifest lookup ->
    locked creation / verified reuse / structured refusal.  The LLM never
    participates in any of these decisions.
    """
    from . import env_manager
    from .audit import audit_environment
    from .env_identity import (
        collect_environment_spec,
        env_id,
        project_slug_for,
        resolved_fingerprint,
        spec_fingerprint,
    )

    task = state.task
    repo_path = state.repo_context.repo_path if state.repo_context else task.workspace_dir / "repo"
    root = Path(task.resource_root).expanduser()
    spec = collect_environment_spec(task, repo_path, probe_log=stderr_path)
    fingerprint = spec_fingerprint(spec)
    identifier = env_id(project_slug_for(task), fingerprint)
    prefix = str(env_manager.conda_envs_dir(root) / identifier)
    manifest_file = env_manager.manifest_path(root, identifier)

    # env_name is only a CANDIDATE hint in this mode (ResAgent injects the
    # manifest-candidate prefix).  The fingerprint is the identity
    # authority: a candidate that equals the identifier or its prefix is
    # simply what the lookup below would find anyway; a mismatching
    # candidate is ignored and the fingerprint lookup/create proceeds.
    candidate = (task.env_name or "").strip()
    if candidate:
        candidate_matches = (
            candidate == identifier
            or str(Path(candidate).expanduser()) == prefix
        )
        stdout_path.write_text(
            f"env_name candidate {'matches' if candidate_matches else 'ignored (mismatch with fingerprint identity)'} "
            f"{identifier}\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")

    manifest = env_manager.read_manifest(root, identifier)
    if manifest is None:
        return _create_content_addressed(state, conda, root, identifier, prefix, spec,
                                         fingerprint, stdout_path, stderr_path)

    manifest_state = manifest.get("state")
    if manifest_state == "creating":
        lock = env_manager.read_lock(root, fingerprint)
        holder_alive = lock is not None and env_manager.holder_alive(lock)
        if holder_alive:
            # An in-flight creator: wait for ready (reuse) or failed (retry).
            return _wait_then_retry(state, conda, root, identifier, fingerprint,
                                    stdout_path, stderr_path)
        env_manager.mark_failed(root, identifier)  # creator died mid-creation
        return _create_content_addressed(state, conda, root, identifier, prefix, spec,
                                         fingerprint, stdout_path, stderr_path)

    if manifest_state in ("drifted", "failed"):
        raise RuntimeError(
            f"environment {identifier} is {manifest_state} — refusing reuse. "
            f"manifest: {manifest_file}. Remediation belongs to M2-P3."
        )

    # ready: verify the prefix, recompute the inventory, re-audit
    if not Path(prefix).exists():
        raise RuntimeError(
            f"manifest for {identifier} is ready but its prefix is missing: {prefix}. "
            f"Refusing reuse; manifest: {manifest_file}"
        )
    inventory = env_manager.collect_resolved_inventory(conda, prefix)
    if not (inventory.get("python") and inventory.get("pip_inventory_sha256")
            and inventory.get("conda_inventory_sha256")):
        raise RuntimeError(
            f"environment {identifier} inventory could not be verified — "
            f"refusing reuse. manifest: {manifest_file}"
        )
    actual = resolved_fingerprint(inventory)
    expected = manifest.get("resolved_fingerprint")
    if expected is None:
        # Post-creation package changes were never finalized by a passing
        # audit — the recorded inventory is unverifiable, refuse reuse.
        raise RuntimeError(
            f"environment {identifier} has no verified inventory — the creating "
            f"run's installations were never finalized by a passing audit. "
            f"Refusing reuse. manifest: {manifest_file}"
        )
    if actual != expected:
        env_manager.mark_drifted(root, identifier, expected=expected, actual=actual,
                                 details="resolved inventory mismatch at reuse time")
        raise _drift_blocker(identifier, manifest_file, expected, actual)

    info = EnvironmentInfo(
        env_name=prefix,
        created=False,
        setup_stdout_path=stdout_path,
        setup_stderr_path=stderr_path,
    )
    state.environment = info  # shim state: the caller replaces it anyway
    audit = audit_environment(state)
    if not audit.success:
        raise RuntimeError(
            f"environment {identifier} failed its pre-reuse audit: {audit.summary}. "
            "The environment is NOT certified for experiments."
        )
    # Second drift line: spec-declared distributions must actually exist.
    violations = env_manager.check_spec_compliance(
        conda, prefix, env_manager.parse_requirements(repo_path))
    if violations:
        raise RuntimeError(
            f"environment {identifier} failed spec compliance: "
            + "; ".join(violations)
        )
    checks = [{
        "name": "policy",
        "outcome": "pass",
        "detail": audit.summary,
        "evidence_path": str(audit.stdout_path) if audit.stdout_path else "",
    }]
    artifact = env_manager.audit_artifact_v1(
        env_id=identifier, level="experiment", outcome="pass",
        resolved_fingerprint=actual,
        audited_by={"module": "reproagent", "run_id": task_run_id(task), "task_id": task.task_id},
        checks=checks,
    )
    env_manager.record_audit(
        root, identifier, artifact=artifact, resolved=inventory,
        resolved_fingerprint=actual, certification="experiment",
        usage={"run_id": task_run_id(task), "task_id": task.task_id, "at": env_manager.utcnow()},
    )
    stdout_path.write_text(f"reused environment: {identifier} (fingerprint verified, audit passed)\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return info


def _wait_then_retry(state, conda: str, root, identifier: str, fingerprint: str,
                     stdout_path, stderr_path) -> EnvironmentInfo:
    """Bounded wait for an in-flight creation, then re-dispatch on its outcome:
    ready → reuse; failed / dead holder → re-create; timeout → structured error."""
    from . import env_manager

    deadline = time.monotonic() + 120
    while True:
        manifest = env_manager.read_manifest(root, identifier)
        if manifest is None or manifest.get("state") in ("ready", "failed"):
            return _ensure_content_addressed(state, conda, stdout_path, stderr_path)
        lock = env_manager.read_lock(root, fingerprint)
        if lock is not None and not env_manager.holder_alive(lock):
            # Local dead holder — take over the lock instead of waiting it out.
            env_manager.recover_stale_lock(root, fingerprint)
            return _ensure_content_addressed(state, conda, stdout_path, stderr_path)
        if lock is None:
            return _ensure_content_addressed(state, conda, stdout_path, stderr_path)
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"timed out waiting for the in-flight creation of {identifier}; "
                "retry after it finishes."
            )
        time.sleep(0.2)


def _create_content_addressed(state, conda: str, root, identifier: str, prefix: str,
                              spec: dict, fingerprint: str, stdout_path, stderr_path) -> EnvironmentInfo:
    """Locked creation per §6.2: acquire, re-check, creating manifest, create,
    inventory, ready.  A waiting caller that sees `ready` re-enters reuse."""
    from . import env_manager
    from .env_identity import resolved_fingerprint

    task = state.task
    repo_path = state.repo_context.repo_path if state.repo_context else task.workspace_dir / "repo"
    deadline = time.monotonic() + 120
    while True:
        lock = env_manager.acquire_creation_lock(root, fingerprint)
        if lock is not None:
            break
        if env_manager.recover_stale_lock(root, fingerprint) is not None:
            # a local dead creator held the lock — recovered, retry acquisition
            continue
        manifest = env_manager.read_manifest(root, identifier)
        if manifest is not None and manifest.get("state") == "ready":
            # another creator finished while we waited — reuse it
            return _ensure_content_addressed(state, conda, stdout_path, stderr_path)
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"timed out waiting for the concurrent creator of {identifier}; "
                "retry after the in-flight creation finishes."
            )
        time.sleep(0.2)

    try:
        # re-check under the lock — the loser may have completed the env
        manifest = env_manager.read_manifest(root, identifier)
        if manifest is not None and manifest.get("state") == "ready":
            env_manager.release_creation_lock(lock)
            return _ensure_content_addressed(state, conda, stdout_path, stderr_path)

        manifest = env_manager.new_manifest(
            env_id=identifier,
            prefix=prefix,
            spec=spec,
            spec_fingerprint=fingerprint,
            created_by={"module": "reproagent", "run_id": task_run_id(task), "task_id": task.task_id},
            provenance={
                # collection semantics: repo_path (absolute) and repo_commit
                # (git HEAD) are required provenance for stale-candidate culling
                "repo_path": str(repo_path),
                "repo_origin": task.repo_url or "local",
                "repo_commit": (state.repo_context.commit_hash
                                if state.repo_context and state.repo_context.commit_hash
                                else _git_head(repo_path)),
            },
        )
        env_manager.write_manifest_atomic(root, identifier, manifest)

        create_cmd = [conda, "create", "-p", prefix, f"python={task.python_version}", "-y"]
        result = _run_conda_setup_with_retries(
            create_cmd,
            cwd=repo_path,
            timeout=task.timeout_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        if result.returncode != 0:
            env_manager.mark_failed(root, identifier)
            raise RuntimeError(
                f"conda environment creation failed for {identifier}; see {stderr_path}"
            )

        inventory = env_manager.collect_resolved_inventory(conda, prefix)
        resolved = resolved_fingerprint(inventory)
        env_manager.mark_ready(root, identifier, resolved, inventory)
        env_manager.record_usage(root, identifier, run_id=task_run_id(task), task_id=task.task_id)
        return EnvironmentInfo(
            env_name=prefix,
            created=True,
            setup_command=" ".join(create_cmd),
            setup_stdout_path=stdout_path,
            setup_stderr_path=stderr_path,
        )
    finally:
        env_manager.release_creation_lock(lock)
