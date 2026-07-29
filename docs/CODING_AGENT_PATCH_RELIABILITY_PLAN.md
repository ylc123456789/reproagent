# CodingAgent Patch Reliability Plan

## Purpose

This document records the CodingAgent-side issue exposed by reproagent cloud run `torchdiffeq-codingagent-2` and defines the recommended fix scope for the separate CodingAgent project.

The reproagent repository should not carry a special fork of CodingAgent. Any CodingAgent implementation changes should be made in the standalone CodingAgent repository, then the vendored copy under `src/coding_agent/` can be synced back into reproagent.

## Current Failure

In the cloud run, reproagent correctly prepared the target conda environment and passed environment context to CodingAgent. CodingAgent verification also ran inside the prepared environment:

```text
/root/miniconda3/bin/conda run -n repro_repro_20260729_202404_4938e7 bash -c <command>
```

The failure was not an environment issue. CodingAgent generated a malformed unified diff and stopped when `git apply` failed:

```text
git apply failed:
error: corrupt patch at <stdin>:30
```

The intended patch was reasonable: add training-loss logging to `examples/odenet_mnist.py` so reproagent could satisfy the experiment goal. The patch failed because the generated unified diff was syntactically invalid, likely due to incorrect hunk line counts or malformed context.

## Required CodingAgent Fix

CodingAgent should make patch application reliable enough for autonomous code-edit tasks. The minimum viable fix is a patch validation and repair loop.

### 1. Validate Before Applying

Before applying a model-generated patch, run:

```bash
git apply --check <patch-file>
```

If validation passes, apply it normally.

If validation fails, do not stop immediately. Save the failed patch and send the exact error plus the failed patch back to the model for one or more repair attempts.

### 2. Save Failed Patch Artifacts

When patch validation or application fails, CodingAgent should persist:

- `failed_patch_<step>.patch`
- `failed_patch_<step>.stderr`
- the model action JSON that produced the patch

This makes downstream debugging much easier from reproagent artifacts.

### 3. Add Patch Repair Attempts

CodingAgent should support a small bounded retry loop, for example 2 repair attempts per patch action:

1. Generate patch.
2. Run `git apply --check`.
3. If invalid, ask the model to repair the patch using the original task, target file context, invalid patch, and `git apply --check` stderr.
4. Re-run `git apply --check`.
5. Apply only after validation succeeds.

If all repairs fail, report a structured failure instead of a generic controller stop.

### 4. Prefer Robust Edit Protocols

Long-term, CodingAgent should reduce dependence on raw unified diff generation. Good options:

- For small files, allow whole-file replacement after reading the file.
- For local changes, ask the model for structured edit operations such as `replace exact old text with new text`.
- Let the controller construct the final diff from file writes, then verify with `git diff`.
- Keep unified diff support, but treat it as one edit strategy, not the only strategy.

The key design idea: the LLM should decide the edit intent, but deterministic code should perform and validate the filesystem mutation whenever possible.

## ReproAgent Integration Boundary

reproagent should remain responsible for:

- cloning or reusing the target repository;
- creating and repairing the conda environment;
- collecting hardware, README, paper, and probe context;
- deciding when a code patch is needed for the reproduction goal;
- passing environment context and allowed verification commands to CodingAgent;
- recording CodingAgent status, report paths, residual risks, and whether experiment commands were executed.

CodingAgent should remain responsible for:

- understanding a code-edit task;
- inspecting repository files;
- making minimal repo-local code/config edits;
- validating edits with the provided verification commands;
- returning changed files, diff, summary, and residual risks.

CodingAgent should not install, upgrade, or remove dependencies when called by reproagent. If verification reveals dependency, CUDA, or package-version problems, CodingAgent should report them as environment issues for reproagent to handle.

## Expected Result After Fix

For the `torchdiffeq` MNIST case, after CodingAgent patch reliability is fixed, the expected flow is:

1. reproagent plans experiment and validation detects that training loss is not logged.
2. reproagent calls CodingAgent with the prepared conda environment context.
3. CodingAgent edits `examples/odenet_mnist.py` or creates a minimal run-specific wrapper/config so loss is logged without changing training semantics.
4. CodingAgent validates the patch and returns `status=completed`.
5. reproagent reruns probe, replans the experiment, asks for confirmation if enabled, and then executes the experiment commands.

## Non-Goals For This Fix

Do not add broad environment-management powers to CodingAgent yet. That can be designed later if needed, but the current integration is clearer and safer if environment repair stays in reproagent.

Do not make a reproagent-specific CodingAgent fork. Keep CodingAgent generic, then sync the vendored copy into reproagent.
