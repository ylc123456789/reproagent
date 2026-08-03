# CodingAgent `insert_after` Anchor Ambiguity Bug

## Summary

`insert_after` with a short anchor matched the wrong closing parenthesis in
nested-paren structures, inserting code inside the outer `logger.info(...)` call
instead of after it, causing `SyntaxError`.

## Reproduction

See run `torchdiffeq-external-codingagent-20260802-123916`.
All logs: `patches/coding_agent_01/logs/`

## Step-by-step Trace

| Step | Action | Anchor / Old Text | Result |
|------|--------|-------------------|--------|
| 02 | `insert_after` | `f_nfe_meter = RunningAverageMeter()\n b_nfe_meter = RunningAverageMeter()` | OK |
| 03 | `insert_after` | `loss = criterion(logits, y)\n\n if is_odenet:` | Wrong indent |
| 06 | `replace_text` | Fixed indent from step 03 | OK |
| 08 | `replace_text` | Updated logger format string (full block) | OK |
| **09** | `run_command` | `--help` | **PASSED** |
| 10-13 | `read_file` | Re-reading file | — |
| 14 | `insert_before` | `end = time.time()` | OK |
| 15 | `read_file` | Re-reading file | — |
| **16** | `insert_after` | **`"                    )"`** | **BUG — see below** |
| 17 | `finish` + verify | Final `--help` check | **FAILED SyntaxError** |

## Root Cause: Action 16

```json
{
  "action": "insert_after",
  "anchor_text": "                    )",
  "insert_text": "\n    train_time = time.time() - train_start\n    logger.info('Total training time: {:.2f} minutes'.format(train_time / 60))"
}
```

At this point the target file contained:

```python
                logger.info(                              # 16 spaces + "logger.info("
                    "...".format(                          # 20 spaces + "...".format("
                        ...                               # args
                    )                                     # 20 spaces + ")" — .format() close
                )                                         # 16 spaces + ")" — logger.info() close
```

The anchor `"                    )"` (20 spaces + `)`) matched the `.format()`
close (first occurrence), **not** the `logger.info()` close (only 16 spaces).

Result: the inserted code landed between `.format()`'s `)` and `logger.info()`'s `)`:

```python
                logger.info(
                    "...".format(
                        ...
                    )                                     # anchor matched here

    train_time = time.time() - train_start
    logger.info('Total training time: {:.2f} minutes'.format(train_time / 60))
                )                                         # original close still present
```

This produces a syntactically invalid construct where `train_time = ...` and
`logger.info(...)` appear as malformed extra arguments to the outer
`logger.info(...)` call.

Python error:
```
SyntaxError: invalid syntax. Perhaps you forgot a comma?
```

## Why Did Step 09 Pass?

Step 09 ran **before** actions 14-16. At that point the file was valid — only
the loss_meter logging changes were applied, and the logger format string update
was correct. The syntax-breaking edit (action 16) came later.

## Two Defects

### 1. Tool: `insert_after` anchor too short / ambiguous

A 20-character anchor consisting only of spaces and `)` is inherently ambiguous
in any file with nested parentheses at similar indentation. The tool should:

- **Prefer longer anchors** — include at least one preceding line for context.
- **Warn on multiple matches** — if the anchor matches N > 1 locations, either
  reject the operation or require the caller to specify `occurrence_index`.
- **Validate after insert** — do a quick syntax check (e.g. `python -m py_compile`)
  after any edit and report syntax errors immediately instead of waiting for
  finish verification.

### 2. Agent: No incremental verification after late-stage edits

The agent re-read the file (action 15), chose the wrong anchor for action 16,
and then called `finish` without running another `--help` check. The finish
verification then failed. The agent should:

- Run verification after **every** edit, not just after early ones.
- If `read_file` shows nested `)` at multiple levels, use context from
  preceding lines to disambiguate the anchor.

## Recommended Fix Priority

1. **Tool fix**: `insert_after` should reject anchors shorter than ~30 chars
   when the file contains similar indentation patterns (or require
   `occurrence_index`).
2. **Tool fix**: Auto-run `python -m py_compile <file>` after every edit.
3. **Agent prompt**: Instruct the agent to use multi-line anchors that
   include the line above the target insertion point.
