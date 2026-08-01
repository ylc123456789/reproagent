# Developer Handoff

This is the practical handoff note for continuing `reproagent` in another Codex/chat session or with another developer.

## Current Status

`reproagent` is an LLM-first machine-learning paper reproduction runner. The current MVP takes a paper URL, a git repo URL, a concrete experiment goal, and a workspace directory. It clones the repo, builds context, creates an isolated conda environment, asks an OpenAI-compatible LLM to plan dependency setup, probes the repository interface, validates the final experiment plan, optionally calls CodingAgent for repo-local patches, runs approved experiment commands, and writes `result.md` plus `state.json`.

The current implementation has been tested on AutoDL with `torchdiffeq` / Neural ODE MNIST. The successful path used GPU, let CodingAgent add minimal loss logging, ran a bounded 5-epoch MNIST experiment, streamed experiment progress, and recorded accuracy/loss/runtime/deviations.

## Important Paths

Local WSL development path:

```bash
cd /home/cyl/reproagent
```

Related local projects:

```text
/home/cyl/reproagent
/home/cyl/CodingAgent
/home/cyl/AutoResearchClaw
```

Cloud path used for testing:

```bash
cd /root/autodl-tmp/projects/reproagent
```

GitHub:

```text
reproagent:  https://github.com/ylc123456789/reproagent.git
CodingAgent: https://github.com/ylc123456789/CodingAgent.git
```

## Setup

```bash
cd /path/to/reproagent
conda env create -f environment.yml
conda activate reproagent
pip install -e .
pytest -q
```

For an existing environment:

```bash
conda activate reproagent
cd /path/to/reproagent
pip install -e .
pytest -q
```

On AutoDL, prefer putting development conda envs on the data disk:

```bash
conda config --add envs_dirs /root/autodl-tmp/conda-envs-dev
conda config --show envs_dirs
```

## Typical Cloud Run

```bash
reproagent run \
  --paper https://arxiv.org/abs/1806.07366 \
  --repo https://github.com/rtqichen/torchdiffeq.git \
  --workspace /root/autodl-tmp/projects/reproagent/runs/torchdiffeq-next \
  --experiment-goal "Run a bounded GPU MNIST ODE-Net experiment using examples/odenet_mnist.py and report test accuracy, training loss, runtime, and any deviations from the paper setup." \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --model deepseek-v4-pro \
  --timeout 3600 \
  --max-env-attempts 3 \
  --max-run-attempts 2 \
  --mirror-profile autodl \
  --enable-coding-agent \
  --max-coding-agent-steps 24 \
  --confirm-before-experiment
```

Useful switches:

```text
--plan-only                    stop after final experiment-plan generation
--confirm-before-experiment    print final commands and wait for y/N
--mirror-profile cn|autodl     prefer domestic/AutoDL dependency mirror guidance
--mirror-strict                block/fail instead of silently leaving the selected mirror profile
--enable-coding-agent          allow repo-local patches when validation says needs_patch
```

## Current Workflow

```text
collect context
  -> prepare task conda env
  -> LLM environment plan
  -> run environment commands
  -> environment audit
  -> LLM probe plan
  -> run safe probe commands
  -> LLM experiment plan
  -> validate experiment plan
     -> needs_config: replan experiment within max-run attempts
     -> needs_patch: run CodingAgent if enabled, then reprobe/replan
     -> blocked/unsafe: stop and report
  -> plan-only or optional user confirmation
  -> run experiment commands
  -> final review
  -> write state.json and result.md
```

LLM commands must be plain repository-root shell commands. They must not use `conda activate`, `conda create`, `conda run`, `cd`, `tee`, or shell log redirection. `reproagent` wraps commands with `conda run -n <env> bash -c <command>` and captures stdout/stderr itself.

## Module Map

```text
src/reproagent/main.py        CLI and top-level workflow
src/reproagent/models.py      Pydantic task/state/result models
src/reproagent/context.py     clone/reuse repo and collect README/file evidence
src/reproagent/hardware.py    hardware and CUDA/GPU context
src/reproagent/env.py         conda env creation, command wrapping, setup retries
src/reproagent/audit.py       post-env Python/pip/torch/CUDA audit
src/reproagent/llm.py         prompts, OpenAI-compatible API calls, mirror guidance
src/reproagent/validation.py  conservative final-plan validation
src/reproagent/runner.py      command safety, execution, logs, live experiment streaming
src/reproagent/coding.py      adapter from reproagent to vendored CodingAgent
src/reproagent/report.py      result.md and state.json writing
src/coding_agent/             vendored generic CodingAgent package
```

## CodingAgent Boundary

This chat/project owns `reproagent`. CodingAgent itself is a separate generic programming-agent project.

Do not customize `src/coding_agent/` for reproagent-specific behavior. If CodingAgent has a bug or missing feature, document the issue, fix it in `/home/cyl/CodingAgent` in the CodingAgent session, then sync the vendored copy into reproagent.

Current contract:

```text
reproagent owns environment creation, dependency installation, audit, hardware context, goal validation, and deciding when to call CodingAgent.
CodingAgent owns minimal repo-local code/config edits and verification inside the prepared environment.
CodingAgent must not install, upgrade, or remove dependencies.
```

Sync pattern after CodingAgent upstream updates:

```bash
cd /home/cyl/reproagent
rsync -a --delete /home/cyl/CodingAgent/src/coding_agent/ src/coding_agent/
pytest -q
git status --short
```

Review the diff before committing.

## AutoDL Notes

Use workspaces under the data disk, commonly:

```text
/root/autodl-tmp/projects/reproagent/runs/<run-name>
```

For slow PyTorch downloads, use:

```bash
--mirror-profile autodl
```

This tells the LLM to prefer AutoDL/domestic mirror behavior and avoid official PyTorch indexes unless necessary.

Disk checks:

```bash
du -h -d 1 /root/autodl-tmp/projects/reproagent/runs | sort -h
du -h -d 1 /root/autodl-tmp/conda-envs-dev 2>/dev/null | sort -h
du -h -d 1 /root/autodl-tmp/conda-envs 2>/dev/null | sort -h
conda env list
```

Delete old run folders and old task-created `repro_repro_*` envs deliberately. Do not remove active dev envs such as `reproagent`, `researchclaw`, `gpu`, or `base`.

## Known Good Test Target

```text
paper: https://arxiv.org/abs/1806.07366
repo:  https://github.com/rtqichen/torchdiffeq.git
goal:  bounded GPU MNIST ODE-Net experiment with test accuracy, training loss, runtime, and deviations
```

Expected behavior on a GPU AutoDL machine:

```text
GPU-capable PyTorch is installed
probe discovers examples/odenet_mnist.py CLI/logging behavior
validation may request a loss-logging patch
CodingAgent adds minimal logging without changing training semantics
reproagent replans and asks for confirmation if requested
experiment runs with live progress
result.md reports loss, train acc, test acc, runtime, and deviations from full-paper training
```

## Current Limitations

Not implemented yet:

```text
Docker execution backend
native SSH/cloud scheduler backend
paper PDF parsing or table extraction
automatic SOTA/repository discovery
dataset/checkpoint acquisition policy
full benchmark/table reproduction planning
native Claude/Gemini SDKs
```

Current validation is conservative. It catches common mismatches but does not prove the experiment is scientifically correct.

## Recommended Next Work

1. Improve the post-patch loop: after CodingAgent edits repo code/config, decide what probes must rerun, how to compare patches against the goal, and when to stop for human review.
2. Add an explicit environment reuse/cache policy. Task envs are isolated but large.
3. Ground final summaries more deterministically in parsed logs/metrics instead of relying only on the LLM final review.
4. Add dataset/checkpoint handling: storage path, mirrors, missing-data blocking, and cleanup.
5. Move from bounded MNIST toward a fuller benchmark/table-row reproduction target.

## Handoff Checklist

```bash
cd /home/cyl/reproagent
git status --short
pytest -q
```

For cloud run debugging, inspect or preserve:

```text
result.md
state.json
logs/*.stderr
logs/*.stdout
patches/coding_agent_*/patch_report.md
patches/coding_agent_*/diff.patch
```
