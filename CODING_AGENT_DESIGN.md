# 通用编程 Agent 开发文档

本文档描述一个可独立维护、也可嵌入 `reproagent` 的通用编程 agent。它不是专门为某一篇论文复现写死的 patcher，而是面向科研自动化系统的通用代码修改与验证能力层。

## 1. 定位

目标是实现一个轻量版 Claude Code / Codex / Aider / SWE-agent 类模块：

- 输入：代码仓库路径、代码任务目标、约束、上下文、验证命令。
- 输出：代码改动、diff、验证结果、日志、是否完成、失败原因。
- 用途：
  - 复现实验时修改原项目代码，例如补充 metric logging、修复环境兼容性、添加小规模配置。
  - 原创实验时编写训练脚本、评估脚本、消融实验脚本。
  - 大科研 agent 中作为统一的“代码执行与修改”子模块。
  - 未来可拆成单独 GitHub 项目维护。

在 `reproagent` 中，coding agent 暂时可以作为一个独立文件夹存在，不需要像第三方依赖一样安装。后续稳定后再考虑单独仓库或 pip 包。

## 2. 参考项目

设计上建议参考以下项目，但不要一开始做成重平台：

- Aider：重点参考 repo map、architect/editor 分工、精确 diff 编辑。
- SWE-agent / mini-SWE-agent：重点参考“执行命令 - 观察反馈 - 修改代码 - 再验证”的闭环。
- OpenHands：重点参考长期方向，如隔离环境、任务状态、可视化、SDK，但当前 MVP 不照搬复杂平台。

核心取舍：

- 采用 LLM-first，而不是写死大量规则。
- 用清晰的状态和文件产物保证可读、可调试。
- 所有代码修改必须有 diff 和验证日志。
- 第一版只支持单仓库、单 agent、线性多轮循环，不做复杂多 agent 协作。

## 3. 模块边界

coding agent 只负责代码相关任务：

- 阅读代码和文档。
- 制定代码修改计划。
- 修改工作副本。
- 运行验证命令。
- 根据失败日志迭代。
- 生成 patch report。

它不负责：

- 搜索论文或 SOTA。
- 判断完整科研路线是否合理。
- 长期实验调度。
- 云服务器资源购买或排队。
- 最终论文写作。

硬件不足、实验太贵、数据缺失等问题可以由上层系统判断。coding agent 可以辅助写 profiling 脚本或小规模配置，但不应该成为资源规划模块。

## 4. 建议目录结构

第一版可以放在 reproagent 仓库中：

```text
coding_agent/
  __init__.py
  models.py          # TaskSpec, AgentState, EditPlan, PatchReport 等数据结构
  context.py         # 构建 repo map、读取相关文件、裁剪上下文
  llm.py             # OpenAI-compatible LLM 调用
  planner.py         # Architect：生成修改计划
  editor.py          # Editor：生成 diff 或文件级编辑
  apply.py           # 应用 patch，检查路径安全
  runner.py          # 运行验证命令，捕获 stdout/stderr
  reviewer.py        # 根据 diff 与验证结果判断是否完成
  agent.py           # 主循环入口
  report.py          # 写 patch_report.md、state.json、diff.patch
  safety.py          # 命令与文件修改安全策略
tests/
  test_coding_agent_*.py
```

如果之后单独开仓库，可以保持同样结构：

```text
coding-agent/
  src/coding_agent/
  tests/
  README.md
  pyproject.toml
```

## 5. 核心数据结构

建议先用 Pydantic，便于 JSON 记录和后续集成。

```python
class CodeTaskSpec(BaseModel):
    repo_path: Path
    task_goal: str
    constraints: list[str] = []
    verify_commands: list[str] = []
    allowed_paths: list[str] = []
    max_iterations: int = 3
    timeout_seconds: int = 900
    api_base: str
    api_key_env: str
    model: str

class EditPlan(BaseModel):
    summary: str
    target_files: list[str]
    allowed_edit_type: Literal["logging_only", "config_only", "bugfix", "new_file", "general"]
    risks: list[str] = []
    verification: list[str] = []
    needs_user_input: list[str] = []
    feasibility: Literal["ready_to_edit", "needs_context", "blocked", "unsafe"]

class PatchReport(BaseModel):
    status: Literal["completed", "failed", "blocked", "needs_user_input"]
    changed_files: list[str]
    diff_path: Path | None
    verification_results: list[CommandResult]
    summary: str
    residual_risks: list[str] = []
```

第一版不需要一口气把字段设计得很重，但必须保留：

- task goal
- constraints
- changed files
- diff
- verification commands/results
- final status
- residual risks

## 6. 工作流

推荐采用 Architect / Editor / Verifier / Reviewer 四段式循环。

```text
CodeTaskSpec
  ↓
Context Builder
  ↓
Architect: 生成 EditPlan
  ↓
Editor: 生成 patch
  ↓
Patch Applier: 应用 patch 并记录 diff
  ↓
Verifier: 运行验证命令
  ↓
Reviewer: 判断是否完成
  ↓
如果失败且未超过 max_iterations，则带日志进入下一轮
```

### 6.1 Context Builder

输入仓库路径和任务目标，生成最小必要上下文：

- 文件树，忽略 `.git`、`__pycache__`、大数据、模型权重、虚拟环境。
- README / docs / config 文件摘要。
- 与任务关键词相关的代码文件。
- 最近验证失败日志。

第一版可以用简单文本检索：

- `rg` 查找关键词。
- 根据文件后缀和大小过滤。
- 只读取前 N KB 或目标函数附近片段。

后续可参考 Aider 的 repo map，加入代码符号索引。

### 6.2 Architect

Architect 负责回答：

- 需要改哪些文件？
- 为什么要改？
- 能不能不改代码，只改配置或命令？
- 这次修改会不会改变实验语义？
- 如何验证？

输出 `EditPlan`，不直接写代码。

对于科研复现场景，Architect 必须区分：

- 只补日志/输出：低风险。
- 只新增配置：低到中风险。
- 修运行错误：中风险。
- 改模型结构、训练逻辑、数据处理：高风险，默认需要用户确认。

### 6.3 Editor

Editor 只根据 EditPlan 生成补丁。

第一版建议支持两种编辑方式：

1. 生成 unified diff。
2. 对小文件允许 whole-file rewrite。

优先 unified diff，因为：

- 便于审查。
- 便于保存产物。
- 便于回滚。

补丁必须满足：

- 只能修改 `repo_path` 内的文件。
- 不能改 `.git`、虚拟环境、数据目录、权重文件。
- 不能写绝对路径。
- 不能做删除大目录等危险操作。

### 6.4 Patch Applier

Patch Applier 负责安全应用补丁：

- 解析 diff。
- 检查所有路径是否在允许范围内。
- 应用 patch。
- 保存 `diff.patch`。
- 保存修改前后文件列表。

第一版可以调用 `git diff` 获取最终 diff，但不要依赖目标仓库一定干净。

如果目标仓库本来有未提交改动：

- 记录初始 diff 为 `initial_diff.patch`。
- 只报告本轮新增 diff。
- 不回滚用户已有改动。

### 6.5 Verifier

Verifier 运行验证命令：

- 命令由上层传入，或由 Architect 建议。
- 必须设置 cwd 为 repo root。
- 必须捕获 stdout/stderr 到日志文件。
- 终端只输出关键进度。
- 禁止危险命令，如 `rm -rf`、`sudo`、`shutdown`、写系统目录。

验证命令示例：

```bash
python examples/odenet_mnist.py --help
python examples/odenet_mnist.py --debug --nepochs 1 --gpu 0
pytest -q
```

### 6.6 Reviewer

Reviewer 根据以下信息判断结果：

- 任务目标。
- 约束。
- diff。
- 验证命令结果。
- 日志尾部。

输出：

- completed
- failed
- blocked
- needs_user_input

Reviewer 必须说明残余风险，例如：

- 只验证了 debug run，未运行 full experiment。
- patch 只补了日志，不保证指标与论文一致。
- 修改可能改变训练时间，但不改变训练语义。

## 7. 安全策略

第一版必须实现的安全策略：

- 只允许修改 repo root 内文件。
- 默认禁止修改 `.git/`、虚拟环境、数据集、模型权重、缓存目录。
- 默认禁止危险 shell 命令：
  - `rm -rf`
  - `sudo`
  - `chmod -R`
  - `chown -R`
  - `shutdown`
  - `reboot`
  - `curl ... | bash`
  - `wget ... | bash`
- 默认禁止静默改变科研语义：
  - 模型结构
  - loss function
  - optimizer
  - dataset split
  - evaluation metric

如果任务确实需要高风险修改，agent 应返回 `needs_user_input`，而不是自动改。

## 8. 与 reproagent 的集成方式

reproagent 不应该直接把 patch 逻辑写死在主流程里，而是通过清晰接口调用 coding agent。

触发条件：

- experiment plan validation 返回 `needs_patch`。
- environment audit/revision 多次失败，且问题看起来是代码兼容性。
- 用户明确要求修改原项目代码来完成目标。

示例：

```python
from coding_agent import run_code_task

report = run_code_task(CodeTaskSpec(
    repo_path=state.repo_context.repo_path,
    task_goal=(
        "Modify the repository minimally so examples/odenet_mnist.py reports "
        "training loss during bounded GPU MNIST ODE-Net runs without changing "
        "model architecture, optimizer, dataset, or evaluation semantics."
    ),
    constraints=[
        "Do not change model architecture.",
        "Do not change optimizer, learning rate, or dataset split.",
        "Prefer adding logging only.",
        "Keep the patch minimal and easy to review.",
    ],
    verify_commands=[
        "python examples/odenet_mnist.py --help",
        "python examples/odenet_mnist.py --debug --nepochs 1 --gpu 0",
    ],
    max_iterations=2,
    timeout_seconds=900,
))
```

reproagent 需要把 coding agent 的结果写入自己的产物：

```text
runs/<task>/
  result.md
  state.json
  patches/
    initial_diff.patch
    coding_agent.patch
    coding_agent_report.md
  logs/
    coding_agent_*.stdout
    coding_agent_*.stderr
```

## 9. MVP 范围

第一版只做一个垂直闭环，不追求覆盖所有代码任务。

推荐 MVP：

> 给定一个 Python ML repo，任务是“最小修改代码以输出某个缺失 metric”，coding agent 能定位训练循环，添加日志，运行 debug 验证，输出 diff 和报告。

必须支持：

- 读取 repo context。
- LLM 生成 edit plan。
- LLM 生成 patch。
- 安全应用 patch。
- 运行验证命令。
- 失败后最多重试一轮。
- 写 `patch_report.md`、`state.json`、`diff.patch`。

暂不支持：

- 多 agent 并发。
- 自动处理大型重构。
- 自动设计复杂新模型。
- 自动购买或切换云机器。
- 自动修改论文级核心算法。

## 10. 第一批测试场景

### 10.1 单文件补日志

输入：一个训练脚本只输出 accuracy，不输出 loss。

目标：添加 loss logging。

验收：

- diff 只改训练脚本。
- 不改训练语义。
- debug run 输出 loss。

### 10.2 新增小规模配置

输入：项目只有 full config，没有 small config。

目标：新增 `configs/reproagent_debug.yaml`。

验收：

- 不改原配置。
- 新配置能跑。
- 报告说明这是 debug/bounded config，不是 full reproduction。

### 10.3 修 API 兼容性

输入：旧代码因新版本库 API 变化失败。

目标：最小兼容性修复。

验收：

- 原功能恢复。
- 报告说明依赖版本差异。

### 10.4 硬件不足时生成 profiling 脚本

输入：full run 超出当前 GPU 显存。

目标：新增一个小 profiling 命令或脚本，估计 batch size / memory。

验收：

- 不声称 full reproduction 完成。
- 报告给出资源建议。

## 11. 产物格式

每次 coding agent run 生成：

```text
coding_agent_run/
  state.json
  patch_report.md
  diff.patch
  initial_diff.patch
  logs/
    plan.json
    edit_response.txt
    verify_01.stdout
    verify_01.stderr
```

`patch_report.md` 至少包含：

- 任务目标。
- 修改摘要。
- 修改文件。
- diff 路径。
- 验证命令与结果。
- 是否完成。
- 残余风险。
- 是否需要用户进一步确认。

## 12. 开发顺序

建议按以下顺序实现：

1. 建 `coding_agent/` 目录和数据模型。
2. 实现 repo context builder。
3. 实现 LLM planner，先只生成 EditPlan。
4. 实现 LLM editor，生成 unified diff。
5. 实现 patch 安全检查和应用。
6. 实现 verifier。
7. 实现 reviewer。
8. 跑 toy repo 单元测试。
9. 用 torchdiffeq 的 training loss 场景做真实测试。
10. 再考虑接入 reproagent 主流程。

## 13. 当前 torchdiffeq 集成目标

当前 `reproagent` 中已有一个真实触发点：

- 用户目标：bounded GPU MNIST ODE-Net experiment，报告 test accuracy、training loss、runtime、deviations。
- 当前问题：原 `examples/odenet_mnist.py` 能输出 test accuracy，但 probe 没证明它输出 training loss。
- validation 结果：`needs_patch`。

coding agent 第一版可以围绕这个问题验证：

```text
Modify examples/odenet_mnist.py minimally so the bounded GPU MNIST ODE-Net run reports training loss and runtime, without changing model architecture, optimizer, dataset, evaluation metric, or default full-experiment behavior.
```

期望输出：

- 一个小 diff。
- debug run 能看到 loss。
- `patch_report.md` 清楚说明只补日志，不改变训练语义。

## 14. 重要原则

- 可读性优先于“大而全”。
- 每次修改都要有 diff。
- 每次验证都要有日志。
- 不确定就停下来，不要悄悄改科研语义。
- 上层 agent 负责研究目标，coding agent 负责代码动作。
- 第一版要小，但接口要像未来独立项目一样清楚。
