# 代码风格统一方案：按 CodingAgent 标准重构

## 目标

将 reproagent 的 prompt/context/LLM 三层对齐到 CodingAgent 的模块划分风格。

## 现状 vs 目标

```
CodingAgent                          ReproAgent (现在)        ReproAgent (改后)
─────────────────────────────────    ─────────────────────    ─────────────────────
controller/prompts.py  ←prompt管理  llm.py (全部混在一起)     prompts.py ← NEW
context.py             ←上下文收集  context.py                context.py (不改)
context_policy.py      ←上下文预算  models.py (ContextPolicy)  context_policy.py ← NEW
llm.py                 ←纯API调用   llm.py                   llm.py (精简)
```

## 具体操作

### 1. 新建 `prompts.py`（~270 行）

从 `llm.py` 搬出所有"非 API"的内容：

```
prompts.py:
  SYSTEM_PROMPT                         # 系统提示词
  build_initial_context()               # 初始上下文构建
  build_turn_prompt()                   # 每轮 prompt 重建
  _env_line()                           # 环境信息格式化
  _cache_line()                         # 数据集缓存提示
  _mirror_block()                       # 镜像策略块
  _compact_audit()                      # 审计压缩
  _compact_step_line()                  # 步骤压缩
  _command_snippet()                    # 命令摘要
  _format_step_full()                   # 最新步骤完整输出
  _mock_response()                      # 测试用 mock
```

### 2. `llm.py` 精简到只保留 API 层（~80 行）

```
llm.py:
  call_llm()                            # 入口（对 controller 的唯一接口）
  _openai_compatible()                  # API 调用 + 重试逻辑
  _chat_completions_url()              # URL 构建
  _write_llm_trace()                    # 日志记录
```

`call_llm(task, system, user)` 签名不变。`system` 参数接收 `SYSTEM_PROMPT`，`user` 参数接收 `build_initial_context()` 或 `build_turn_prompt()` 的结果。

controller 改为从 `prompts` 导入 prompt 相关函数，从 `llm` 导入 `call_llm`。

### 3. 新建 `context_policy.py`（~50 行）

从 `models.py` 搬出：

```
context_policy.py:
  MODEL_CONTEXT_WINDOWS                 # 模型上下文窗口表
  ContextPolicy                         # 上下文预算策略类
```

跟 CodingAgent 的 `context_policy.py` 结构完全对应。

### 4. `models.py` 清理

删除 `ContextPolicy` 和 `MODEL_CONTEXT_WINDOWS`（移到 context_policy.py）。其余不变。

### 5. `controller.py` 更新 import

```
# 改前
from .llm import SYSTEM_PROMPT, build_initial_context, build_turn_prompt, call_llm
from .models import ..., ContextPolicy, ...

# 改后
from .prompts import SYSTEM_PROMPT, build_initial_context, build_turn_prompt
from .llm import call_llm
from .context_policy import ContextPolicy
from .models import ...
```

## 不改的文件

| 文件 | 原因 |
|------|------|
| `context.py` | 已经跟 CodingAgent 对齐 |
| `controller.py` | 只改 import，不改逻辑 |
| `runner.py`、`audit.py`、`coding.py`、`env.py`、`report.py` | 不受影响 |
| 所有测试文件 | 只更新 import |

## 文件行数对比

| 文件 | 改前 | 改后 |
|------|------|------|
| `llm.py` | 379 行 | ~80 行 |
| `prompts.py` | — | ~270 行 |
| `context_policy.py` | — | ~50 行 |
| `models.py` | 242 行 | ~195 行 |
| `controller.py` | 387 行 | ~390 行（仅 import 变动） |

## 影响

- 功能：**零影响**。所有函数只是换了一个文件，逻辑一行不改
- 测试：仅更新 import 路径
- 命令行接口：不变
- 对外 API：`from reproagent.llm import call_llm` 仍然可用（重导出自 prompts）
