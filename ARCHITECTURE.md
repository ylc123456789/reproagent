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

The central goal is to let an LLM read repository context, paper reference, and the user-provided experiment goal, decide how a human would try to reproduce that target, probe the repository interface, produce a concrete execution plan, execute approved commands inside an isolated conda environment, and revise the plan after failures.

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
  runner.py    # command safety checks, execution, live log streaming
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
4. Probe Stage
5. Final Experiment Planning
6. Experiment Loop
7. Result Review
8. Write Report
```

Each loop can retry locally. A failed environment command does not restart the whole run; it returns to the environment planner with the latest logs. A failed experiment command returns to the experiment planner with the latest logs.

```text
Build Context
  -> Create/reuse conda env
  -> Env plan by LLM for the experiment goal
  -> Run env commands inside conda env
  -> if failed: send logs back to LLM and retry env stage
  -> Probe plan by LLM for safe interface discovery
  -> Run only help/config/listing/inline-inspection probe commands
  -> Final experiment plan by LLM for the experiment goal, using probe logs
  -> Validate final plan for obvious goal mismatches, unsupported metric claims, unsafe logging commands, and missing explicit budgets
  -> if validation marks needs_patch/blocked/unsafe: write the issue and do not execute experiment commands
  -> if --plan-only: write planned experiment and stop before training/evaluation
  -> optionally ask user to confirm the experiment plan
  -> Run experiment commands inside conda env with live output streaming
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

Within a stage, the system passes compact stage history and relevant tail logs back to the LLM. Probe logs are included before final experiment planning so the model can turn discovered CLI/config parameters into explicit commands instead of relying on script defaults.

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

The next milestone is: improve final-plan validation, especially checking that goal words such as bounded, GPU, loss, accuracy, or full reproduction are reflected in explicit commands/configs before expensive runs start.
