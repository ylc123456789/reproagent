# 架构重构方案：从线性阶段改为 Agent Loop

## 一、为什么需要换架构

当前的线性架构：

```
collect context → env setup → audit → probe → experiment plan → validate → run → done
                  ↑ 失败可重试(最多3次)    ↑          ↑ 失败可重试(最多3次)
```

每层问题：

| 阶段 | 问题 |
|------|------|
| 环境配置 | LLM 不知道脚本 import 了哪些依赖、镜像可能不可用、传递依赖缺失 |
| 审计 | 只检查 torch，不检查其他关键依赖。漏了也发现不了 |
| 探测 | 依赖不全会导致 `--help` 直接报错，探测没产出 |
| 实验规划 | 依赖探测结果，探测没产出 → 实验规划就瞎猜 |
| 运行 | 依赖不全会炸，但此时已无法回环境阶段——只能失败退出 |
| 校验 | 删了硬规则但还是依赖 LLM，LLM 有时不敢拍板 |

根本矛盾：**环境问题可能在任何阶段暴露，但线性架构不允许回头**。

Agent Loop 的做法：一个 LLM 持续观察状态、决定下一步、执行、观察结果、再决定。装依赖 → import 失败 → 缺 torchvision → 补装 → import 成功 → 探测 → 实验 → 完成。没有"阶段"的概念，只有连续的行动。

## 二、参考项目

| 项目 | 做法 | 参考什么 |
|------|------|---------|
| SWE-agent | 简单 agent loop + 工具（bash/edit/submit），LLM 决定何时做什么 | 工具设计、状态管理 |
| OpenHands | Agent + 工具，能来回切换读写代码和运行命令 | 动作格式、观察反馈 |
| Claude Code | 工具调用循环，read→write→bash→observe→repeat | 单一 agent，没有阶段分隔 |
| AutoGPT | 老牌 agent loop，但过于复杂 | 不参考，太复杂 |

共性：**一个循环**，LLM 有多个工具，根据观察结果决定下一步。不预设阶段，让 LLM 判断该做什么。

## 三、新架构总览

```
                     ┌──────────────────────────┐
                     │     Controller Loop       │
                     │                          │
初始化 ──────────────→│  1. 收集状态快照          │
  - clone repo       │  2. LLM 分析 + 决定行动    │←──────────┐
  - 创建 conda env   │  3. 执行行动               │           │
  - 采集硬件信息      │  4. 记录结果               │           │
  - 采集文件树/README │  5. 判断是否结束           │──→ done   │
                     │    否则 → 回到步骤 1        │           │
                     └──────────────────────────┘           │
                                                              │
                     可用的工具:                               │
                     ┌─────────────────────┐                  │
                     │ run_commands()      │ 执行shell命令     │
                     │ audit_env()         │ 检查环境状态      │
                     │ call_coding_agent() │ 委托代码修改      │
                     │ finish()            │ 结束任务          │
                     └─────────────────────┘                  │
                                                              │
                     每个工具执行后返回结果 ─────────────────────┘
```

### 3.1 与当前代码的对应

| 当前模块 | 改动 |
|---------|------|
| `context.py` | **保留** — clone repo、收集 README/文件树/硬件，初始化时跑一次 |
| `env.py` | **保留** — 创建 conda 环境、`build_backend_command`，变成工具 `setup_conda()` |
| `runner.py` | **保留** — `run_commands()`、安全检测，变成工具 `run_commands()` |
| `audit.py` | **保留** — `audit_environment()`，变成工具 `audit_env()` |
| `coding.py` | **保留** — `run_coding_agent_for_patch()`，变成工具 `call_coding_agent()` |
| `report.py` | **保留** — `write_result()`，在 finish 时调用 |
| `models.py` | **小改** — 新增 Action/Observation 模型 |
| `main.py` | **重写** — 替换为 controller loop（约 200 行） |
| `llm.py` | **重写** — 替换为单一 controller prompt + `_call_llm_text` 保留（约 200 行） |
| `text.py` | **保留** |

### 3.2 Controller Loop 伪代码

```python
def run_controller(task):
    # 初始化（一次性）
    repo_context = collect_context(task)
    conda_env = ensure_environment(task)
    
    # 初始 prompt
    history = [build_initial_prompt(task, repo_context, conda_env)]
    
    for step in range(max_steps):
        # LLM 决定下一步
        response = call_llm(history)
        action = parse_action(response)      # {action, commands, reasoning, ...}
        
        # 执行
        if action.type == "run_commands":
            result = run_commands(action.commands, ...)
            history.append(format_result(result))
        elif action.type == "audit_env":
            result = audit_environment(...)
            history.append(format_audit(result))
        elif action.type == "call_coding_agent":
            result = run_coding_agent_for_patch(...)
            history.append(format_ca_result(result))
        elif action.type == "finish":
            write_result(state)
            return result
        
        # 超过预算强制结束
        if step >= max_steps:
            force_finish()
```

### 3.3 LLM 的 Prompt 结构

```
System:
You are a machine learning reproduction engineer. Your task is to reproduce
an ML paper experiment in a prepared conda environment.

Available tools:
- run_commands(commands, stage_hint) — execute shell commands
- audit_env() — check installed packages and GPU  
- call_coding_agent(goal, issues) — delegate code modifications to a coding agent
- finish(status, summary) — end the task

Safety rules:
- Never use sudo, rm -rf, curl|bash, shutdown/reboot
- Do NOT use cd, tee, shell redirection — the runner handles this

Context:
Paper: <url>
Repo: <url> (already cloned at <path>, commit <hash>)
Experiment goal: <goal>
Hardware: <nvidia-smi output>
Conda env: <name> (freshly created, no packages installed)
File tree:
<tree>

READ ME / docs:
<readme excerpt>

---

## Step 1

You: <assistant decides what to do first>
```

之后每次 step 都是追加观察结果 + LLM 响应，形成自然的对话历史。LLM 能看到前面所有的行动和结果。

### 3.4 动作的 JSON 格式

```json
{
  "thinking": "The environment is empty. I need to install PyTorch with CUDA support, torchvision, and the repo itself.",
  "action": "run_commands",
  "stage_hint": "environment",
  "commands": [
    "pip install torch==2.6.0 torchvision==0.21.0",
    "pip install -e .",
    "python -c \"import torch; print(torch.cuda.is_available())\""
  ]
}
```

```json
{
  "thinking": "torch is installed but torchvision failed. Let me check the environment state.",
  "action": "audit_env"
}
```

```json
{
  "thinking": "torchversion is missing because the mirror doesn't have it. Let me install it from the default index.",
  "action": "run_commands",
  "stage_hint": "environment",
  "commands": ["pip install torchvision", "python examples/odenet_mnist.py --help"]
}
```

```json
{
  "thinking": "The script needs modification to print training loss. Delegating to CodingAgent.",
  "action": "call_coding_agent",
  "goal": "Add training loss logging to examples/odenet_mnist.py",
  "issues": ["Script does not print training loss per epoch"]
}
```

```json
{
  "thinking": "Experiment completed. Test accuracy 99.05%, runtime 14 minutes. Training loss was added by CodingAgent.",
  "action": "finish",
  "status": "completed",
  "summary": "Bounded GPU MNIST ODE-Net experiment completed with all metrics collected."
}
```

### 3.5 关键设计决策

**1. 不采用 Function Calling API**

用 JSON 文本格式而不是 OpenAI function calling。原因：兼容所有 LLM（DeepSeek、任意 OpenAI-compatible API）。

**2. 工具数量保持 4 个**

`run_commands` / `audit_env` / `call_coding_agent` / `finish`。不再细分 "只读命令" "写命令"，LLM 自己判断该跑什么。安全由 runner.py 兜底。

**3. 对话历史代替硬状态**

不构建复杂的状态对象。每次 LLM 调用时，history 包含从初始 prompt 到当前步骤的所有行动和结果。LLM 自己从历史中理解当前状态——就像人看终端输出一样。

**4. CodingAgent 的调用时机由 LLM 决定**

不再需要 validation 判断 needs_patch。LLM 在跑实验时发现脚本不输出 loss → 主动决定 call_coding_agent → CodingAgent 修改代码 → 回来后 LLM 继续跑实验。

**5. 审计变成可选工具**

LLM 可以在任何时候调用 `audit_env` 来了解环境中装了哪些包、GPU 是否可用。不必在固定位置跑。

**6. 最大步数兜底**

设 `max_steps`（比如 30）防止 LLM 死循环。接近上限时系统强制要求 finish。

## 四、与当前架构的对比

| 维度 | 当前（线性+重试） | 新（Agent Loop） |
|------|-----------------|-----------------|
| LLM 调用次数 | 固定 ~6-8 次（每阶段 1-3 次） | 灵活，通常 ~8-15 步 |
| 环境失败后能不能回修 | 不能，只能同阶段重试 3 次 | 能，LLM 随时跑 pip install |
| CodingAgent 触发 | validation 判 needs_patch | LLM 自己判断 |
| 探测和信息收集 | 一个专门的 probe 阶段 | 随时可以，LLM 觉得需要就跑 |
| 状态追踪 | 分散在 ReproState 各字段 | 对话历史，LLM 自己理解 |
| 新增依赖检查 | 改审计代码 | LLM 看到 import 报错就自己补 |
| 安全 | runner.py | runner.py（保留不动） |
| 代码量 | main.py 489行 + llm.py 498行 | controller.py ~200行 + llm.py ~200行 |

## 五、实施步骤

### 第一步：新建 `controller.py`

约 200 行。实现 controller loop + 四个工具函数。逻辑：

- `run_controller(task)` — 主循环
- `_execute_action(action, state)` — 分发到对应工具
- `_build_history(action, result)` — 格式化观察追加到 history

### 第二步：重写 `llm.py`

约 200 行。保留 `_call_llm_text`、`_openai_compatible_text`、`_base_context`。新增：

- `_system_prompt()` — 系统指令 + 工具说明
- `_format_observation(result)` — 将工具结果格式化为文本

删除所有旧的 stage-specific prompt（`plan_environment`、`plan_probe`、`plan_experiment`、`revise_after_failure`、`review_experiment_plan`、`final_review`、`apply_review_to_plan`）。

### 第三步：修改 `models.py`

新增 Action、Observation 类型。删除不再需要的（CommandPlan？保留，因为 runner 和 coding 还用）。

### 第四步：重写 `main.py`

从 489 行缩减到约 80 行。只保留：CLI 解析、创建 ReproTask、调用 `run_controller(task)`、打印结果。

### 第五步：更新测试

mock_llm 模式下用一个预定义的 action 序列代替 LLM 调用。测试 controller loop 本身（能正确执行工具、能处理 finish）。

### 第六步：真实测试

用 torchdiffeq case 验证：环境安装 → probe → 发现需要 patch → CodingAgent → 跑实验 → 完成。

## 六、风险和缓解

| 风险 | 缓解 |
|------|------|
| LLM 可能在循环里打转 | max_steps 硬限制，接近上限时在 prompt 里警告 |
| LLM 可能执行危险命令 | runner.py 安全检测保留不动 |
| 对话历史可能过长 | 裁剪：保留最近 20 步 + 初始 context |
| LLM 可能瞎调 CodingAgent | CodingAgent 自身有 max_steps 和 task 约束 |
| 结果不可复现 | 每次行动记录到日志，可回放 |
