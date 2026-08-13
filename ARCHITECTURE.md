# reproagent Architecture

## 1. Purpose

`reproagent` is the experiment operator: an LLM-driven runner for machine-learning
research repositories. Reproduction is one of its modes — when a paper is given,
the goal is reproducing that paper; otherwise the goal is the experiment goal
exactly as specified.

Input:

```text
experiment_goal                 (always)
workspace_dir                   (always — session workspace, always private)
one of: repo_url                → isolated mode (clone into the workspace)
        copy_from               → copy mode (local worktree copy, uncommitted changes kept)
        external_repo_path      → shared mode (operate on an existing repo in place)
setup_only (optional)           → provision the environment, no experiments
```

Output:

```text
result.md
state.json
session.yaml        (with execution-contract bindings: repo/environment/caches)
logs/
patches/
```

The LLM drives the entire process in an agent loop: it observes state, chooses an
action (install dependencies, probe the repository, run experiments, delegate to
CodingAgent), executes it, and repeats until the experiment goal is satisfied or
the step budget is exhausted.

## 2. File Structure

```text
src/reproagent/
  __init__.py                  # stable public API: ReproTask, ReproState, CommandPlan, CommandResult
  agent.py                     # top-level public run/resume API (run_task, resume_task)
  models.py                    # shared Pydantic data structures (public + persisted state)
  main.py                      # CLI only: argparse, task assembly, calls into agent.py
  controller/                  # agentic loop, action dispatch, prompts
    __init__.py                # re-exports run_controller (ResAgent imports this)
    loop.py                    # state loop: init/resume, step budget, finish, report + session card
    actions.py                 # action parsing + execution: run_commands, audit_env, call_coding_agent
    prompts.py                 # system prompt + turn prompt builders + formatting helpers
  runtime/                     # side-effect execution
    runner.py                  # command safety checks, execution, live log streaming,
                               # pip cache layering: REPROAGENT_PIP_CACHE >
                               # sibling of dataset cache > per-workspace
    environment.py             # conda environment creation and command wrapping (bash -o pipefail)
    audit.py                   # post-setup environment audit (torch/tf/jax)
    hardware.py                # lightweight hardware context collection
    dataset_cache.py           # scan hardcoded dataset roots, resolve to absolute
                               # paths, pre-create symlinks into the shared cache
  repository/
    context.py                 # workspace setup (isolated/copy/shared/resume),
                               # clone, and README/file-tree/hardware collection
  context/
    policy.py                  # context budget policy scaled to model window
  integrations/
    codingagent.py             # CodingAgent location/import boundary + patch orchestration
  llm.py                       # pure API transport layer (serialise, HTTP, retry) + mock responses
  session.py                   # session.yaml cards, list/status, resume helpers
  report.py                    # write result.md and state.json
  text.py                      # text normalization for LLM output and reports
```

Top-level `runner.py`, `env.py`, `audit.py`, `hardware.py`, `dataset_cache.py`,
`context_policy.py`, `prompts.py`, and `coding.py` remain as thin compatibility
shims that forward public symbols to the locations above. They hold no second
implementation. `context` is a package (not a shim module): its `__init__.py`
re-exports `clone_repo`/`collect_context` from `repository.context`, because a
same-named package shadows a same-named module in Python's import system.
`tests/test_compat_shims.py` pins all of these legacy paths by identity.

### Dependency direction

```text
main
  -> agent
      -> controller (loop -> actions -> prompts)
          -> runtime / repository / context / integrations
      -> session / report / llm

models
  <- importable by every layer, imports nothing internal
```

`models.py` imports no controller, runtime, or integration code. Prompt files
do no I/O. Report/session modules never trigger agent decisions.

## 3. Environment Strategy

MVP uses conda only.

```text
environment.yml present → conda env create -n <env> -f environment.yml
otherwise               → conda create -n <env> python=<version> -y
```

All LLM-proposed commands run through:

```bash
conda run -n <env> bash -o pipefail -c "<command>"
```

`pipefail` makes pipeline failures propagate: `python train.py | tail` reports
the failure of `python`, not the success of `tail`.

## 4. Workflow (Agent Loop)

The controller runs a single loop. There are no fixed stages — the LLM decides
what to do next based on the current state.

```
initialize:
  → setup the workspace (isolated clone / copy worktree / bind external repo)
  → create conda env
  → collect hardware / file-tree / README context
  → bridge hardcoded dataset roots into the shared cache (best-effort,
    bounded by the allowed write root: task workspace, or the external
    repo itself in shared mode)

loop (max N steps):
  → build prompt from current state (compacted history + file cache)
  → LLM returns JSON action
  → execute action via one of four tools:
      run_commands      — execute shell commands
      audit_env          — check installed packages and GPU
      call_coding_agent  — delegate code modifications to CodingAgent
      finish             — write final report and end the run
  → observe result, update state, repeat

finalize:
  → write result.md + state.json
  → write session.yaml card
```

Tools are described in the system prompt. The LLM can freely switch between
"stages" — it can install a missing dependency at any time, re-audit after
changes, or call CodingAgent whenever a code change is needed.

## 5. Context Management

Context is rebuilt from structured state each turn, not appended as raw chat
history. This keeps prompts focused on current decision-making.

- System prompt: static rules, tool schemas, safety constraints.
- Turn prompt: goal, task params, environment state, mirror policy, latest
  audit, cached file reads, compacted step history, full last-step result.
- Old steps are compacted to single-line summaries with output tails.
- File reads are cached and de-duplicated.
- Context policy scales with the model window (DeepSeek 1M → loose limits,
  GPT-4o 128K → tighter limits).

## 6. LLM API Strategy

OpenAI-compatible chat completions endpoint. Each turn sends system + user
as a fresh pair of messages — no accumulated chat history.

```text
api_base + /chat/completions
api_key_env
model
temperature 0.2
```

All prompt/response pairs are saved to `logs/` for debugging. `--mock-llm`
uses the deterministic mock responses in `llm.py` and never hits the network.

## 7. CodingAgent Integration

CodingAgent is a separate generic programming-agent project. The agent calls it
via the `call_coding_agent` tool when a code modification is needed (e.g. adding
a missing metric to the script output).

Configuration priority:

```text
CLI argument:     --codingagent-path
Environment var:  CODINGAGENT_PATH
Config file:      agents.codingagent_path
Fallback:         importable `coding_agent` package, if available
```

`src/reproagent/integrations/codingagent.py` is the only module that resolves
the path, validates the checkout, imports the CodingAgent API, and orchestrates
patch requests (goal/constraint building, verification command wrapping).

ReproAgent owns environment creation, dependency installation, audit, hardware
context, and deciding when to call CodingAgent. CodingAgent owns minimal
repo-local code/config edits, verification inside the prepared environment,
and patch report/diff generation.

## 8. Public API

```python
from reproagent import ReproTask, ReproState, CommandPlan, CommandResult
from reproagent.controller import run_controller   # loop with full finalization
from reproagent.agent import run_task, resume_task  # CLI-facing wrappers

# isolated / copy / shared — exactly one repository source:
state = run_task(ReproTask(repo_url="https://...", workspace_dir=...))
state = run_task(ReproTask(copy_from="/local/worktree", workspace_dir=...))
state = run_task(ReproTask(external_repo_path="/shared/repo", workspace_dir=...))
# environment provisioning only:
state = run_task(ReproTask(repo_url="...", workspace_dir=..., setup_only=True))
state = resume_task(Path("/path/to/workspace"), "continue with 5 epochs")
```

`allow_code_delegation=False` (default True) disables the internal
call_coding_agent delegation: the run exits `blocked` with the coding
issues listed so the orchestrator can route a CodingAgent task and resume.

CLI:

```text
reproagent run       --repo ... | --copy-from ... | --external-repo ...
                     --workspace ... --experiment-goal ... [--setup-only]
reproagent resume    <workspace> --instruction ...
reproagent list      --root <dir>
reproagent status    <workspace>
```

## 9. Session and Workspace Layout

Every task writes into `<workspace_dir>/`:

```text
state.json         # full AgentState (resume reads this)
result.md          # final human-readable report
session.yaml       # session index card (status, bindings, key artifacts)
logs/              # per-command stdout/stderr, conda setup, audit, LLM traces
patches/           # CodingAgent patch outputs
repo/              # repository worktree (cwd for all commands; shared mode
                   # points at the external repo instead)
```

session.yaml `bindings` follows the execution-contract v1 sub-schema
(additive over the legacy flat keys conda_env/dataset_cache/pip_cache):

```yaml
bindings:
  repo: {path, origin, commit, mode}          # mode: isolated|copy|shared
  environment: {name, policy, certification}  # certification: experiment
    #   + certified_at + audit_artifact when the audit passed
  dataset_cache: /path
  pip_cache: /path
```

Resume semantics: same `task_id` → same conda env; steps are appended to the
previous state; `attempt_count` increments.

## 10. Adding New Code

- **New agent action**: add the JSON type to `AgentAction.action` in
  `models.py`, a handler in `controller/actions.py`, dispatch in
  `controller/loop.py`, and the tool description in `controller/prompts.py`
  (a prompt change is a behavior change — update the phase0 contract hash
  deliberately, never inside a refactor).
- **New runtime capability** (new backend, new cache layer): add a module in
  `runtime/`; keep `models.py` free of runtime imports.
- **New integration** (external agent/service): add a module in
  `integrations/`; nothing else may import the external module directly.
- **New repository context source**: extend `repository/context.py`.

## 11. Tests

```bash
cd /home/cyl/reproagent
conda activate reproagent
pytest -q                                        # full suite (112+)
pytest -q tests/test_phase0_contract.py          # behavior freeze locks
```

Phase 0 contract locks: public exports, CLI parameter set, SYSTEM_PROMPT and
initial-context hashes, persisted model fields. A refactor must never change
these; a deliberate behavior change must update them in its own commit.

## 12. Not Yet Implemented

```text
Docker backend
cloud / SSH execution backend
paper PDF parsing
automatic SOTA discovery
full benchmark / table reproduction planning
multi-GPU scheduling
native Claude / Gemini SDKs
```
