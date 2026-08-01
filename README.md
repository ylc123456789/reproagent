# reproagent

LLM-first machine learning paper reproduction runner.

Given a paper URL, repository URL, and concrete experiment goal, `reproagent` asks an LLM how to set up and run the project, first probes the repository experiment interface, asks an LLM to produce a concrete execution plan, executes approved commands inside an isolated conda environment, retries failed stages with log feedback, and writes a local reproduction result.

This project is intentionally small. It is not a full autonomous research agent yet.

## Development Location

Develop this project in Ubuntu-D:

```bash
cd /home/cyl/reproagent
```

Sibling projects:

```text
/home/cyl/AutoResearchClaw
/home/cyl/auto-researcher
/home/cyl/reproagent
```

## Install

`reproagent` itself can be installed in a small conda environment. Reproduced paper repositories still run in separate per-task conda environments created by `reproagent`.

```bash
cd /path/to/reproagent
conda env create -f environment.yml
conda activate reproagent
pip install -e .
pytest -q
```

To update an existing environment:

```bash
cd /path/to/reproagent
conda env update -f environment.yml --prune
conda activate reproagent
pip install -e .
```

Runtime requirement:

```text
conda must be available on PATH, or set REPROAGENT_CONDA_EXE=/path/to/conda
```

Use the conda setup above for local and server development so CLI behavior matches the reproduction backend. The controller environment should use Python 3.11 when ReproAgent, ExpAgent, and CodingAgent share one environment; per-task reproduction environments still use `--python-version` when no target repo environment file exists.

## Run with Mock LLM

Mock mode is useful for local CLI tests, but still requires conda because commands run through the conda backend.

```bash
reproagent run \
  --paper https://arxiv.org/abs/2301.07093 \
  --repo https://github.com/milesial/Pytorch-UNet \
  --workspace runs/unet-demo \
  --experiment-goal "Run a minimal import/version check for the repository." \
  --mock-llm
```

## Plan Without Running Experiments

Use `--plan-only` when you want the agent to set up the environment, run safe probe commands such as `--help`/config inspection, and write the final proposed experiment plan without starting training or evaluation.

```bash
reproagent run \
  --paper https://arxiv.org/abs/xxxx.xxxxx \
  --repo https://github.com/user/project \
  --workspace runs/plan-demo \
  --experiment-goal "Run a bounded GPU training experiment and report accuracy, loss, runtime, and deviations." \
  --api-base https://api.deepseek.com/v1 \
  --api-key-env DEEPSEEK_API_KEY \
  --model deepseek-v4pro \
  --plan-only
```

## Run with OpenAI

```bash
export OPENAI_API_KEY=...
reproagent run \
  --paper https://arxiv.org/abs/xxxx.xxxxx \
  --repo https://github.com/user/project \
  --workspace runs/demo \
  --experiment-goal "Run the paper's main bounded evaluation and report the primary metric." \
  --model gpt-4.1-mini
```

## Run with DeepSeek / OpenAI-compatible API

```bash
export DEEPSEEK_API_KEY=...
reproagent run \
  --paper https://arxiv.org/abs/xxxx.xxxxx \
  --repo https://github.com/user/project \
  --workspace runs/demo \
  --experiment-goal "Run the requested reproduction target and report metrics, logs, and any deviations." \
  --api-base https://api.deepseek.com/v1 \
  --api-key-env DEEPSEEK_API_KEY \
  --model deepseek-v4pro
```

Use the exact model name provided by your API provider.

## CodingAgent Location

`reproagent` can call CodingAgent when `--enable-coding-agent` is set and final-plan validation reports that a repo-local patch is needed. CodingAgent is treated as an external dependency, not as a fixed nested repository path.

Configure its checkout path in this priority order:

```text
CLI:             --codingagent-path /home/cyl/CodingAgent
Environment:     CODINGAGENT_PATH=/home/cyl/CodingAgent
Config file:     agents.codingagent_path
Fallback:        importable `coding_agent` package, if available
```

For a shared controller environment, install the external CodingAgent checkout into that same environment and pass its path. `reproagent` keeps `vendor/coding_agent/` only as a synchronized vendored source copy; `pip install -e .` installs only the `reproagent` package to avoid shadowing the global CodingAgent package.

Example config file:

```yaml
agents:
  codingagent_path: /home/cyl/CodingAgent
```

Relative CLI and environment paths resolve against the current working directory. Relative config values resolve against the config file directory.

Example:

```bash
reproagent run \
  --paper https://arxiv.org/abs/xxxx.xxxxx \
  --repo https://github.com/user/project \
  --workspace runs/demo \
  --experiment-goal "Run a bounded GPU evaluation and report metrics." \
  --enable-coding-agent \
  --codingagent-path /home/cyl/CodingAgent
```

## Execution Model

The MVP is conda-first:

```text
1. clone repo
2. collect README/docs/file-tree context
3. create a task-specific conda env
4. ask LLM for dependency setup commands needed for the experiment goal
5. run setup commands with conda run -n <env>
6. ask LLM for safe probe commands such as --help/config inspection
7. run probe commands and feed their logs back to the LLM
8. ask LLM for the final experiment/eval/demo plan that directly attempts the goal
9. optionally stop with --plan-only, or confirm before running experiment commands
10. run commands with conda run -n <env>
11. write result.md and state.json
```

LLM commands should not contain `conda activate`, `conda create`, or `conda run`; `reproagent` wraps commands itself.

## Output

Each run writes:

```text
runs/<task>/
  state.json
  result.md
  repo/
  logs/
  context/
```

## Current MVP Scope

The MVP focuses on using an LLM to understand how to run a repository. It does not yet promise full paper-table reproduction.

Not implemented yet:

```text
Docker backend
cloud/SSH execution backend
paper PDF parsing
automatic dataset/checkpoint management
automatic SOTA discovery
full benchmark/table reproduction planning
```
