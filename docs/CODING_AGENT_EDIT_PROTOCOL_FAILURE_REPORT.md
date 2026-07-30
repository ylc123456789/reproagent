# CodingAgent Edit Protocol Failure Report

## Background

This report documents the failure observed when reproagent used the updated vendored CodingAgent during cloud run `torchdiffeq-codingagent-3`.

The previous CodingAgent issue was that a malformed unified diff caused `git apply` to fail immediately. CodingAgent was then updated with patch validation and repair attempts. The new run confirms that the repair loop is active, but also shows that relying on LLM-generated unified diffs remains too fragile.

## Run Under Investigation

reproagent run workspace:

```text
/root/autodl-tmp/projects/reproagent/runs/torchdiffeq-codingagent-3
```

Downloaded artifact path on Windows:

```text
E:/agent26/reproagent/torchdiffeq-codingagent-3
```

Target repository:

```text
https://github.com/rtqichen/torchdiffeq.git
commit 657943acefa826ef04c025ebeb1ff5e9d60dc268
```

Experiment goal:

```text
Run a bounded GPU MNIST ODE-Net experiment using examples/odenet_mnist.py and report test accuracy, training loss, runtime, and any deviations from the paper setup.
```

## What Worked

### reproagent Environment Setup Worked

reproagent created a conda environment and installed a GPU-capable PyTorch build:

```text
Conda env: repro_repro_20260730_090619_fd8e3c
Torch: version=2.6.0+cu124, compiled_cuda=12.4, cuda_available=True, device_count=1
GPU: NVIDIA GeForce RTX 4090 D
```

This confirms the failure is not due to missing GPU or dependency setup.

### reproagent-to-CodingAgent Environment Bridge Worked

CodingAgent was given the prepared environment context, and verification commands were wrapped through conda:

```text
/root/miniconda3/bin/conda run -n repro_repro_20260730_090619_fd8e3c bash -c 'python examples/odenet_mnist.py --help'
```

The verification command passed.

### CodingAgent Patch Repair Loop Was Active

The updated CodingAgent did not stop after the first malformed patch. It saved failed patch artifacts and attempted repair:

```text
failed_patch_04_01.patch
failed_patch_04_01.stderr
failed_patch_04_02.patch
failed_patch_04_02.stderr
failed_patch_04_03.patch
failed_patch_04_03.stderr
```

This confirms the vendored CodingAgent update was loaded by reproagent.

### reproagent Failure Reporting Worked

reproagent did not continue to run experiment commands after CodingAgent failed. The final summary correctly stated:

```text
Experiment commands were not executed because CodingAgent did not complete the required patch.
```

So the remaining issue is inside CodingAgent's edit mechanism, not reproagent's final reporting.

## Current Failure

CodingAgent failed after the initial patch plus two repair attempts:

```text
patch failed validation/application after 2 repair attempt(s):
error: corrupt patch at <stdin>:37

---
error: patch fragment without header at <stdin>:25: @@ -370,7 +370,11 @@ if __name__ == '__main__':

---
error: patch fragment without header at <stdin>:24: @@ -370,7 +372,8 @@ if __name__ == '__main__':
```

The failed patches were saved under:

```text
patches/coding_agent_01/logs/failed_patch_04_01.patch
patches/coding_agent_01/logs/failed_patch_04_02.patch
patches/coding_agent_01/logs/failed_patch_04_03.patch
```

## Important Observation

The edit intent was good. CodingAgent correctly tried to add training-loss and runtime logging to `examples/odenet_mnist.py`, which is exactly what reproagent needed.

The problem is not task understanding. The problem is the edit application protocol.

The generated patch attempted to:

- create a `loss_meter = RunningAverageMeter()`;
- update it with `loss.item()` during training;
- include `Loss {:.4f}` in the epoch log line;
- reset the meter after each epoch log;
- optionally log total runtime.

However, the unified diff hunk headers and context did not match the real file. `git apply --check` therefore rejected the patch.

## Root Cause

The current CodingAgent repair strategy still depends on the LLM producing a syntactically valid unified diff. This is unreliable because:

1. LLMs frequently produce incorrect hunk line numbers.
2. LLMs may mix real context with approximate context.
3. Even after being shown `git apply --check` errors, the model may keep producing patch fragments without valid headers.
4. A semantically correct edit can fail purely because the diff serialization is invalid.

The failure mode is structural: asking the model to repair raw unified diff text is not robust enough for autonomous code-edit tasks.

## Recommended CodingAgent Fix

CodingAgent should add a more deterministic edit protocol and stop treating raw unified diff as the only mutation path.

### 1. Add Structured Text Replacement

Support an action such as:

```json
{
  "action": "replace_text",
  "path": "examples/odenet_mnist.py",
  "old_text": "exact text copied from the current file",
  "new_text": "replacement text"
}
```

Controller behavior:

1. Read the target file.
2. Verify `old_text` occurs exactly once.
3. Replace it with `new_text`.
4. Write the file.
5. Run `git diff` to produce the final diff artifact.
6. If `old_text` occurs zero or multiple times, ask the LLM for a corrected replacement with local file context.

This avoids hunk line-number errors entirely.

### 2. Add Insert-Before / Insert-After Operations

Support actions such as:

```json
{
  "action": "insert_after",
  "path": "examples/odenet_mnist.py",
  "anchor_text": "    b_nfe_meter = RunningAverageMeter()",
  "insert_text": "    loss_meter = RunningAverageMeter()\n"
}
```

Rules:

- `anchor_text` must match exactly once.
- If not, fail with a structured message and ask for repair.
- The controller performs the edit, not the model.

This is useful for small instrumentation patches like logging metrics.

### 3. Keep Unified Diff As A Fallback, Not The Main Path

`apply_patch` can remain, but CodingAgent should prefer structured edit actions when:

- the file has already been read;
- the patch touches a small number of local snippets;
- the task is logging/config/control-flow instrumentation;
- previous diff validation failed.

### 4. Improve Repair After Diff Failure

When unified diff validation fails, the next repair attempt should not necessarily ask for another unified diff. Instead, CodingAgent should ask the model to convert the intended change into structured edit operations.

Example repair instruction:

```text
The unified diff failed to apply. Do not produce another diff. Produce exact replace_text or insert_after operations using anchors copied from the current file context.
```

### 5. Persist More Debug Information

The latest run already saved failed patches. Continue doing that, but also save:

- the exact current file snippets used for repair;
- the model repair response JSON;
- whether the failure was syntax-level, context mismatch, safety violation, or ambiguous anchor.

## Minimal Implementation Plan For CodingAgent

1. Extend `ControllerAction` with new actions:
   - `replace_text`
   - `insert_before`
   - `insert_after`

2. Add deterministic editor functions:
   - `replace_text_once(repo_root, path, old_text, new_text, allowed_paths)`
   - `insert_before_anchor(repo_root, path, anchor_text, insert_text, allowed_paths)`
   - `insert_after_anchor(repo_root, path, anchor_text, insert_text, allowed_paths)`

3. Update the controller prompt:
   - Prefer structured edit actions after reading a file.
   - Use `apply_patch` only for larger multi-file changes or when structured edits are unsuitable.
   - After `git apply` failure, prefer structured edits for repair.

4. Update state/report artifacts:
   - Save action JSON for structured edits.
   - Save resulting `git diff` after deterministic edits.
   - Include structured edit failures in residual risks.

5. Add tests:
   - exact one-match `replace_text` succeeds;
   - zero-match and multi-match replacements fail safely;
   - `insert_after` succeeds with exact anchor;
   - malformed unified diff can be recovered by a structured edit action;
   - generated final diff contains expected code.

## Expected Result After Fix

For the torchdiffeq MNIST example, CodingAgent should be able to apply the logging change using structured edits even if the model cannot produce a valid unified diff.

After CodingAgent returns `status=completed`, reproagent should:

1. rerun probe;
2. replan the experiment;
3. ask for confirmation if `--confirm-before-experiment` is enabled;
4. execute the bounded MNIST experiment;
5. report test accuracy, logged loss, runtime, and deviations.

## Non-Goals

Do not move conda or dependency management into CodingAgent yet. In reproagent integration, environment preparation and repair should stay in reproagent.

Do not make reproagent-specific hacks inside CodingAgent. The structured edit protocol should be generic and useful for any coding task.
