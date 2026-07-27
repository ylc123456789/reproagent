# reproagent

LLM-first machine learning paper reproduction runner.

Given a paper URL and a repository URL, `reproagent` asks an LLM how to set up and run the project, executes the proposed commands inside an isolated conda environment, retries failed stages with log feedback, and writes a local reproduction result.

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

A Python venv also works for local development, but the conda setup above is the recommended path for servers.

## Run with Mock LLM

Mock mode is useful for CLI smoke tests, but still requires conda because commands run through the conda backend.

```bash
reproagent run \
  --paper https://arxiv.org/abs/2301.07093 \
  --repo https://github.com/milesial/Pytorch-UNet \
  --workspace runs/unet-demo \
  --mock-llm
```

## Run with OpenAI

```bash
export OPENAI_API_KEY=...
reproagent run \
  --paper https://arxiv.org/abs/xxxx.xxxxx \
  --repo https://github.com/user/project \
  --workspace runs/demo \
  --model gpt-4.1-mini
```

## Run with DeepSeek / OpenAI-compatible API

```bash
export DEEPSEEK_API_KEY=...
reproagent run \
  --paper https://arxiv.org/abs/xxxx.xxxxx \
  --repo https://github.com/user/project \
  --workspace runs/demo \
  --api-base https://api.deepseek.com/v1 \
  --api-key-env DEEPSEEK_API_KEY \
  --model deepseek-v4pro
```

Use the exact model name provided by your API provider.

## Execution Model

The MVP is conda-first:

```text
1. clone repo
2. collect README/docs/file-tree context
3. create a task-specific conda env
4. ask LLM for dependency setup commands
5. run setup commands with conda run -n <env>
6. ask LLM for experiment/eval/demo commands
7. run commands with conda run -n <env>
8. write result.md and state.json
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
```
