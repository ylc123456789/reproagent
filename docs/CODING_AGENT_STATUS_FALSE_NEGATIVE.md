# CodingAgent 报告状态误判：非关键验证失败导致整体标记为 failed

## 问题

CodingAgent 在完成代码修改并成功运行实验后，报告的整体状态被标记为 `failed`，尽管所有实质性工作都已正确完成。

## 复现

Run: `torchdiffeq-external-codingagent-20260803-171450`
产物: `patches/coding_agent_01/`

## 操作时间线

```
Step 1  replace_text   加 loss_meter 初始化                    ✅
Step 2  replace_text   加 loss_meter.update()                  ✅
Step 3  replace_text   改 logger 格式字符串，加入 Loss 字段      ✅
Step 4  run_command    grep 'Loss.*Test Acc' → returncode=1   ❌
Step 5  read_file      读文件确认修改正确                        —
Step 6  run_command    1-epoch 训练验证 → exit=0                ✅
Step 7  run_command    5-epoch 训练验证 → exit=0                ✅
                       结果: Test Acc 98.88%, Loss 0.0387, 4m21s
Step 8  finish         status="completed"                      ✅
```

最终 patch_report.md 的 `## Status` 字段是 `failed`。finish action 的 `status` 是 `"completed"`。两者矛盾。

## 根因

CodingAgent 的报告生成器在计算最终状态时，遍历了所有 verification 结果。Step 4 的 grep 返回了非零退出码（returncode=1），导致整体状态被判为 `failed`。

但 Step 4 失败的原因不是代码改错了，而是 grep pattern 本身有问题：`Loss.*Test Acc` 中 `Loss` 和 `Test Acc` 在源码里被 Python 的隐式字符串拼接分到了两行：

```python
"Epoch {:04d} | ... | Loss {:.4f} | "     # 第 1 行
"Train Acc {:.4f} | Test Acc {:.4f}"       # 第 2 行
```

默认 `grep` 不跨行匹配，所以永远匹配不到。

## 影响

对 reproagent 的影响是致命的：`_run_coding_agent_patch_cycle` 检查 `result.status != "completed"` 来判断 CodingAgent 是否成功。因为 CodingAgent 报告了 `failed`，reproagent 认为补丁失败，整个实验阶段终止，即使所有指标已经正确产出。

## 修复建议

1. **报告状态判定逻辑**：不要因为某一条 verification 失败就判整体 failed。区分"代码修改失败"和"某条验证命令没匹配到"——前者是 failed，后者最多是 warning。

2. **finish 时的状态应覆盖报告状态**：如果 agent 在 finish action 里标记 `status="completed"`，那最终报告的状态应该以 finish 为准，而不是被中间某条 verification 覆盖。

3. **verification 失败应分级**：
   - 代码修改类验证（py_compile、语法检查）失败 → 严重，可判 failed
   - 功能验证（实际跑脚本/实验）失败 → 严重，可判 failed
   - grep/cat 类文本查找失败 → 非严重，不改变最终状态（或最多 warning）

## 相关证据文件

- `patches/coding_agent_01/logs/action_04.json` — 失败的 grep 命令
- `patches/coding_agent_01/logs/action_08.json` — finish action，status="completed"
- `patches/coding_agent_01/patch_report.md` — 最终报告，Status: failed
