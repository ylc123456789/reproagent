# reproagent Architecture

## 1. Purpose

`reproagent` is a small LLM-first reproduction runner for machine learning projects.

Input:

```text
paper_url
repo_url
workspace_dir
experiment_goal
```

Output:

```text
result.md
state.json
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
  __init__.py
  models.py                    # shared Pydantic data structures
  controller.py                # agent loop — the core of reproagent
  llm.py                       # system prompt, turn prompt builder, LLM API calls
  runner.py                    # command safety checks, execution, live log streaming
  audit.py                     # post-setup environment audit (torch/tf/jax)
  env.py                       # conda environment creation and command wrapping
  context.py                   # clone repo and collect README/file-tree/hardware
  coding.py                    # CodingAgent patch orchestration
  hardware.py                  # lightweight hardware context collection
  text.py                      # text normalization for LLM output and reports
  report.py                    # write result.md and state.json
  main.py                      # CLI entry point
  integrations/
    codingagent.py             # CodingAgent location/import boundary
```

## 3. Environment Strategy

MVP uses conda only.

```text
environment.yml present → conda env create -n <env> -f environment.yml
otherwise               → conda create -n <env> python=<version> -y
```

All LLM-proposed commands run through:

```bash
conda run -n <env> bash -c "<command>"
```

## 4. Workflow (Agent Loop)

The controller runs a single loop. There are no fixed stages — the LLM decides
what to do next based on the current state.

```
initialize:
  → clone repo
  → create conda env
  → collect hardware / file-tree / README context

loop (max N steps):
  → build prompt from current state (compacted history + file cache)
  → LLM returns JSON action
  → execute action via one of four tools:
      run_commands      — execute shell commands
      audit_env          — check installed packages and GPU
      call_coding_agent  — delegate code modifications to CodingAgent
      finish             — write final report and end the run
  → observe result, update state, repeat
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

All prompt/response pairs are saved to `logs/` for debugging.

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
the path, validates the checkout, and imports the CodingAgent API.

ReproAgent owns environment creation, dependency installation, audit, hardware
context, and deciding when to call CodingAgent. CodingAgent owns minimal
repo-local code/config edits, verification inside the prepared environment,
and patch report/diff generation.

## 8. Not Yet Implemented

```text
Docker backend
cloud / SSH execution backend
paper PDF parsing
automatic SOTA discovery
full benchmark / table reproduction planning
multi-GPU scheduling
native Claude / Gemini SDKs
```
