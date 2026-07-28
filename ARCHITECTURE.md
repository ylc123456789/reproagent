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
```

The central goal is to let an LLM read repository context, paper reference, and the user-provided experiment goal, decide how a human would try to reproduce that target, execute the plan inside an isolated conda environment, and revise the plan after failures.

## 2. Project Path

Official development directory:

```text
/home/cyl/reproagent
```

Sibling projects:

```text
/home/cyl/AutoResearchClaw
/home/cyl/auto-researcher
/home/cyl/reproagent
```

Use Ubuntu-D for development, dependency installation, and command execution.

## 3. File Structure

Keep the project small:

```text
src/reproagent/
  __init__.py
  models.py    # shared Pydantic data structures
  context.py   # clone repo and collect README/file evidence
  env.py       # conda environment creation and command wrapping
  llm.py       # LLM prompts and OpenAI-compatible API calls
  runner.py    # command safety checks, execution, log capture
  report.py    # write result.md and state.json
  main.py      # CLI and top-level workflow
```

## 4. Environment Strategy

MVP uses conda only.

```text
environment.yml present -> conda env create -n <env> -f environment.yml
otherwise               -> conda create -n <env> python=<version> -y
```

All LLM-proposed commands run through:

```bash
conda run -n <env> bash -c "<command>"
```

This avoids fragile non-interactive `conda activate` behavior.

Docker is intentionally not implemented yet. If a repo contains a Dockerfile, the LLM can see it in the file tree, but the runner will still use conda for now.

## 5. Workflow

Outer workflow is linear:

```text
1. Build Context
2. Prepare Conda Environment
3. Environment Loop
4. Experiment Loop
5. Result Review
6. Write Report
```

Each loop can retry locally. A failed environment command does not restart the whole run; it returns to the environment planner with the latest logs. A failed experiment command returns to the experiment planner with the latest logs.

```text
Build Context
  -> Create/reuse conda env
  -> Env plan by LLM for the experiment goal
  -> Run env commands inside conda env
  -> if failed: send logs back to LLM and retry env stage
  -> Experiment plan by LLM for the experiment goal
  -> optionally ask user to confirm the experiment plan
  -> Run experiment commands inside conda env
  -> if failed: send logs back to LLM and retry experiment stage
  -> Result review
```

## 6. LLM API Strategy

`llm.py` supports OpenAI-compatible chat completions endpoints:

```text
api_base + /chat/completions
api_key_env
model
```

Examples:

```text
OpenAI:   api_base=https://api.openai.com/v1, api_key_env=OPENAI_API_KEY
DeepSeek: api_base=https://api.deepseek.com/v1, api_key_env=DEEPSEEK_API_KEY
```

It does not yet implement native Claude or Gemini SDKs.

## 7. LLM Memory / Session Design

API models should not be treated as having reliable permanent memory. `reproagent` stores the memory itself:

```text
state.json
context/context_summary.md
logs/*.stdout
logs/*.stderr
```

Within a stage, the system passes compact stage history and relevant tail logs back to the LLM.

## 8. Not Yet Doing

This MVP does not yet do:

```text
automatic SOTA discovery
automatic repository discovery
Docker backend
large dataset management
paper PDF table verification
full blackboard architecture
web UI
```

The next milestone is: pass a concrete reproduction target, use a DeepSeek/OpenAI-compatible API to plan setup/run commands, execute them in conda, and produce an honest result file with metrics, deviations, and logs.
