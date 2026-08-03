# 实验计划校验重构方案

## 一、现在是怎么工作的

### 1.1 整体流程

reproagent 一次完整运行总共 7 步：

```
第1步  收集上下文      git clone 仓库，读取 README，采集硬件信息
第2步  创建 conda 环境   用 environment.yml 或者裸 python=3.10 创建
第3步  环境安装循环      让 LLM 规划装什么依赖 → 执行 pip install → 审计检查
第4步  探测循环          让 LLM 规划怎么探查仓库（--help、grep、cat 等）
第5步  [可选] 只出计划   如果加了 --plan-only，到这里就停了
第6步  实验规划循环      让 LLM 规划实验命令 → 两层校验 → 执行或重试
第7步  最终总结          LLM 写报告
```

其中第 6 步（实验规划循环）是目前问题最多的地方，下面展开说。

### 1.2 实验规划循环的详细流程

```
第1次尝试:
  LLM 生成实验计划
    → 硬规则校验（validation.py，11 条正则/字符串检查）
    → LLM 语义校验（review_experiment_plan_semantics）
    → 合并所有问题，算出 feasibility（可行性等级）
    
    如果 feasibility == ready_to_run → 执行命令 ✅
    如果 feasibility == needs_config → 让 LLM 重新规划（还有次数的话）
    如果 feasibility == needs_patch  → 调用 CodingAgent 打补丁，然后重新探测
    如果 feasibility == blocked      → 直接结束，不给重试 ❌
    如果 feasibility == unsafe       → 直接结束 ❌

第2次尝试（如果第1次没过）:
  LLM 根据上次的反馈重新生成计划
    → 同样的两层校验
    → 同样的 feasibility 判断
    ...
最多 3 次（--max-run-attempts）
```

### 1.3 两层校验具体做了什么

**第一层：硬规则（validation.py，纯代码，不用 LLM）**

共 11 条规则，全部是正则表达式或字符串匹配：

| 规则 | 怎么检查 | 判什么等级 |
|------|---------|-----------|
| 是否 bounded | goal 里有 "bounded"，命令里有没有 `--nepochs` 关键词 | needs_config |
| 是否用 GPU | goal 里有 "gpu"，有没有 `--gpu` 或 cuda 证据 | needs_config |
| 是否输出 loss | goal 里有 "loss"，probe 日志里有没有 logger/print | **needs_patch** |
| 是否用猜测语气 | plan 里有 "assume"、"likely" 等词 | needs_config |
| 是否用 shell 重定向 | 命令里有 `tee`、`2>&1`、` >`、` >>`、`cd ` | **blocked** |
| 是否只有检查命令 | 实验阶段只有 grep/cat/--help 没有训练命令 | needs_config |
| 是否包含 --help | 实验命令里有 --help（应该放探测阶段） | needs_config |
| 入口脚本是否缺失 | goal 里提到的 .py 文件没出现在命令里 | needs_config |
| CLI flag 是否正确 | 命令里的 --xxx 在 probe 的 --help 输出里不存在 | needs_config |
| CLI flag 缺值 | 命令用了 --xxx 但没给值 | needs_config |
| LLM 语义审查结果 | LLM 返回的 issue 文本里匹配关键词推断等级 | 视关键词而定 |

**第二层：LLM 语义审查（review_experiment_plan_semantics）**

把计划发给 LLM，让它判断是否满足 goal 要求。LLM 返回问题列表，然后代码根据问题里的关键词推断可行性等级（比如问题里有 "patch" 就判 needs_patch）。

### 1.4 可行性等级与重试规则

```
ready_to_run            → 直接执行
needs_config            → 可以重试（LLM 重新规划）
needs_patch             → 调用 CodingAgent 打补丁
blocked                 → 直接终止，不能重试 ❌
unsafe_or_too_expensive → 直接终止，不能重试 ❌
```

**关键问题**：只有 `needs_config` 能触发重新规划。一旦被判定为 `blocked` 或 `unsafe`，即使还有剩余尝试次数，流程也直接结束。

### 1.5 Runner 安全检测（这一层是合理的，不改）

在 `runner.py` 里，每个命令执行前会过安全检查：

```
阻止的命令: sudo、rm -rf、| bash、shutdown、reboot、conda activate
阻止的行为: 路径穿越（../ 跳出仓库目录）
探测阶段额外限制: 只能执行检查类命令（--help、grep、cat 等），不能跑训练
```

这是兜底安全层，无论 LLM 生成什么命令都会过这一关。**这层保持不变**。

---

## 二、目前存在的问题

### 2.1 假阳性（不该拦的拦了）

**案例 1**：LLM 生成了一个 inline Python 脚本，里面有 `if x > 0`（比较运算符），被规则 `" >"` 匹配成 shell 重定向 → 判 `blocked` → 流程死亡。

```
命令里有:  if args.max_iters > 0
规则匹配:               ^^    ← " >" 击中了空格+大于号
规则认为:  shell 重定向（其实是 Python 比较运算符）
```

**案例 2**：脚本本来就有 `logger.info(...Loss...)` 输出 loss，但 probe 日志太长/格式复杂，规则没解析到 → 判 `needs_patch` → 不必要的 CodingAgent 调用。

### 2.2 假阴性（该拦的没拦）

**案例 3**：goal 要求 "bounded"，LLM 生成了 `--nepochs 160`。规则只检查 `--nepochs` 关键词是否存在，不检查值 → 160 epoch 被放行，但 160 根本不 bounded。

### 2.3 架构问题

| 问题 | 后果 |
|------|------|
| `blocked` 不可重试 | LLM 随手写了个带 ` >` 的命令就毁掉整个 run |
| 两层校验互相打架 | 硬规则说 needs_config，LLM 说 needs_patch，合并后到底听谁的？规则写死了不可调 |
| 越补规则越多 | 每发现一个新 case 就加一条正则，复杂度越来越高，假阳性越来越多 |
| `_semantic_issue_feasibility` 用关键词匹配推断 LLM 意图 | LLM 说"可能缺 loss"→ 因为含 "patch" 关键词就判 needs_patch，但 LLM 可能只是不确定 |

---

## 三、改后的方案

### 3.1 核心原则

> **LLM 负责思考和判断。硬代码只管安全和基础设施。**

分工如下：

| 层 | 负责什么 | 谁来做 |
|----|---------|--------|
| 基础设施 | git clone、conda 环境、子进程执行、文件读写 | 代码 |
| 安全兜底 | sudo、rm -rf、管道 bash、路径穿越、探测阶段限制 | 代码（runner.py） |
| 规划 | 装什么依赖、探测什么、实验命令怎么跑 | LLM |
| 校验 | 这个计划对吗？满足 goal 了吗？格式对吗？flag 对吗？ | **LLM**（改用 LLM） |
| 修正 | 根据校验反馈修正计划 | **LLM**（改用 LLM） |

关键变化：**把校验工作从硬代码还给 LLM**——一个 LLM 调用同时做规划和审视。硬代码只守最底线的安全。

### 3.2 新流程

```
第6步 实验规划循环（改后）:

  LLM 生成实验计划
    → Runner 安全检测（sudo/rm -rf 等，碰到就直接停）
    → LLM 自审视（一次调用，检查所有质量要求）
       返回: { ready: true/false, issues: [...], feasibility: "..." }
    
    如果 ready → 执行命令 ✅
    如果 needs_patch → CodingAgent 打补丁 → 重新探测 → 重新规划
    如果 needs_config → 把问题喂给 LLM 让它改 → 重新规划（有次数就继续）
    如果 blocked → 也是把问题喂给 LLM → 重新规划（有次数就继续）
    
    只有 Runner 安全检测触发时才直接停（unsafe）
```

**核心变化**：`needs_config` 和 `blocked` 都可以重试。只有硬安全规则（sudo、rm -rf）才会导致立即终止。

### 3.3 LLM 自审视做了什么

一次 LLM 调用覆盖之前两层校验的全部内容，prompt 里明确列出来让 LLM 逐条检查：

**目标对齐**：
- goal 说 bounded → 命令里的 --nepochs 值真的小吗？（10 以内，不是 160）
- goal 说 GPU → 命令有 GPU flag 或者脚本默认 CUDA 吗？
- goal 要求某个指标 → probe 证据显示脚本会输出那个指标吗？
- goal 要求 runtime → 命令有 time 或在脚本输出里能看到吗？
- goal 提到特定脚本 → 命令里真的跑了那个脚本吗？

**命令格式**：
- 实验阶段不能有 --help（那是探测阶段的事）
- 不能用 cd、tee、shell 重定向（runner 已经处理了）
- 不能只有检查类命令（grep/cat），必须有实际的训练/评估命令

**CLI 正确性**：
- 命令里用的 --flag 在 probe 的 --help 输出里存在吗？
- flag 需要的值给了吗？
- flag 名字拼对了吗？

**判定标准**：
- 只在真正不可行时判 `blocked`（缺失数据、缺失检查点、硬件完全不兼容）
- 格式问题、flag 问题一律判 `needs_config`（LLM 改一下就好，别卡死）
- 需要改代码才能满足 goal → `needs_patch`

### 3.4 可行性等级（简化后）

```
ready_to_run  → 执行
needs_config  → 重试（LLM 修正计划）         ← 所有格式/flag 问题走这条
needs_patch   → CodingAgent 打补丁           ← 真正需要改代码才走这条
unsafe        → 直接停（只来自 Runner 安全检测） ← 硬安全规则
```

---

## 四、具体要改哪些文件

### 4.1 删除：`src/reproagent/validation.py`

整个文件删除，约 370 行。

包含要删的内容：所有 11 条硬规则、`ValidationIssue`、`collect_experiment_validation_issues`、`validate_experiment_plan`、`annotate_plan_with_validation_issues`、以及所有辅助函数（`_has_gpu_execution_evidence`、`_loss_logging_is_uncertain`、`_cli_argument_issues`、`_command_flags`、`_help_options` 等）。

### 4.2 修改：`src/reproagent/llm.py`

**改写 `review_experiment_plan_semantics` → 新的 `review_experiment_plan`**

旧的只做语义对齐检查，新的覆盖所有质量检查。新函数返回结构化结果：

```python
# 返回值示例
{
    "ready": False,
    "issues": [
        "命令使用了 --batch-size，但 probe 的 --help 输出里只有 --batch_size（下划线不是连字符）",
        "goal 要求 bounded 运行，但 --nepochs 160 是脚本默认值，不是 bounded。应该改成 --nepochs 5 或 10"
    ],
    "feasibility": "needs_config"
}
```

**更新 `revise_after_failure`**

在 prompt 里加上前一轮的校验反馈，让 LLM 知道上次哪里不对，这次针对性地改。

### 4.3 修改：`src/reproagent/main.py`

需要改的地方：

1. **删除 import**：不再 `from .validation import ...`

2. **改 `_validate_experiment_plan`**：
   ```python
   # 旧: 先硬规则 → 再 LLM 语义 → 合并
   # 新: 只调用 LLM 自审视
   def _validate_experiment_plan(state, plan):
       result = review_experiment_plan(state, plan)
       return apply_review_to_plan(plan, result)
   ```

3. **改 `_run_stage_loop` 实验部分**：
   - 去掉 `blocked` 直接退出的逻辑
   - 所有非 ready 状态都走重试（还有次数就 continue）
   - 只有 runner 安全检测的 unsafe 才直接 `return False`

4. **保留不变**：`_run_coding_agent_patch_cycle`、`_confirm_experiment`、`_stage_succeeded` 等都不动。

### 4.4 修改测试

可能有个别测试引用了 `validation.py` 的函数，需要同步更新。测试数量少（目前 95 个），改动量不会大。

---

## 五、为什么不这样做有风险，这样做是对的

### 现在的做法（硬规则为主）

```
LLM 产生问题 → 硬代码拦截 → 问题被阻断，LLM 没机会修正 → run 失败
```

每次 LLM 产生一个格式问题（比如 `--batch-size` 写成连字符、Python 代码里有 `> 0`），硬规则直接判 blocked，整个 run 就砸了。而这些问题是 LLM 完全有能力自己修正的——只要给它看反馈。

### 改后的做法（LLM 自修复）

```
LLM 产生问题 → LLM 审视发现问题 → 喂回给 LLM 修正 → 修正后执行
```

质量检查交给 LLM，修正也交给 LLM。硬代码只在安全底线兜底。这样：

- LLM 犯了格式错误 → LLM 看到反馈 → LLM 自己改 → 下一次就对了
- LLM 用了 `--nepochs 160` 但 goal 要 bounded → LLM 审视时发现 → 主动改成 5 → 合格
- LLM 用了不存在的 `--batch-size` → LLM 审视时对照 help 输出 → 改成 `--batch_size` → 合格

**不会卡死的闭环**：只要还有重试次数，问题就能被修正。

---

## 六、实施顺序

1. `llm.py` — 写新的 `review_experiment_plan`，综合 prompt
2. `main.py` — 重新接线，去掉 validation.py 的调用
3. `main.py` — 改重试逻辑（所有非 ready 都可重试）
4. `validation.py` — 删除
5. 跑测试 → 修测试 → 确认通过
6. 用 torchdiffeq case 真实验证
