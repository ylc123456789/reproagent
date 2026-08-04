# 代码审查：非通用化实现

审查日期：2026-08-04

---

## 问题列表

### 1. OMP_NUM_THREADS 硬编码为 16

**文件**：`runner.py:196`

```python
env["OMP_NUM_THREADS"] = "16"
```

在 192 核服务器上太保守（只用 16/192），在 4 核机器上太激进（超额订阅）。应自适应：

```python
env["OMP_NUM_THREADS"] = str(min(16, os.cpu_count() or 4))
```

---

### 2. audit.py 只检查 PyTorch，不检查其他框架

**文件**：`audit.py:33-43`

PROBE_CODE 硬编码了 `import torch` 和 `torch.cuda.is_available()`。如果目标项目用 TensorFlow、JAX、或纯 NumPy 项目，审计没有任何框架相关信息。

另外 GPU 检测也假设 NVIDIA（`nvidia-smi`）。Apple Silicon MPS、AMD ROCm 无法检测。

**影响**：中等。对现有测试用例（PyTorch + NVIDIA）没问题，换项目可能漏报。

---

### 3. coding.py 传给 CodingAgent 的验证命令永远为空

**文件**：`models.py:163` + `coding.py:105-117`

`AgentState.probe_attempts` 返回 `[]`（空列表）。`_verification_commands` 遍历 `state.probe_attempts` 收集 `--help` 命令 → 永远收集不到 → CodingAgent 拿不到任何预置的验证命令，全靠自己生成。

**影响**：低。CodingAgent 有自己的验证逻辑，不影响正确性，但没有利用 reproagent 已探明的脚本接口信息。

---

### 4. build_initial_context 没使用 ContextPolicy 的 readme_chars

**文件**：`llm.py:127`

```python
{repo_context.readme_text[:16000]}
```

硬编码了 16000 字符，但 `ContextPolicy.readme_chars` 有模型感知的值（30000/16000/8000）。两者不一致——policy 设了但没用。

---

### 5. build_turn_prompt 丢了 paper_url 和 repo_url

**文件**：`llm.py:145-148`

```
# build_turn_prompt 的 Task 段：
Goal: {experiment_goal}
Timeout: ... | Steps used: ...
```

只保留了 goal，丢掉了 paper URL 和 repo URL。agent 在第 2 步之后无法再看到"我在复现哪篇论文、哪个仓库"。虽然 goal 足够驱动决策，但丢失了完整任务上下文。

---

### 6. _MAX_STEPS 和 _FORCE_FINISH_AFTER 是死代码

**文件**：`controller.py:35-36`

```python
_MAX_STEPS = 30
_FORCE_FINISH_AFTER = 28
```

定义了但从未使用。实际限制来自 `task.max_steps`（默认 30）。

---

### 7. _update_file_cache 的正则匹配不全

**文件**：`controller.py:139`

```python
m = re.match(r"(?:cat|head(?:\s+-n\s+\d+)?|tail(?:\s+-n\s+\d+)?|sed\s+[^ ]+)\s+(.+)", cmd)
```

只匹配 `cat`/`head`/`tail`/`sed`。`grep` 的输出（也是文件内容）不会被缓存。`python foo.py --help` 的输出也不会。不过这不是错误——grep 的输出通常不是完整文件内容，缓存意义不大。

---

### 8. _command_env 的 PIP_CACHE_DIR 绑定到 workspace

**文件**：`runner.py:190-192`

```python
pip_cache_dir = workspace / ".cache" / "pip"
```

每个 run 独享 pip 缓存，跨 run 不能复用。如果网络慢，同一包被反复下载。可以加一个 `--pip-cache-dir` 参数让用户指定共享缓存目录。

---

### 9. report.py 保留了大量没用到的旧函数

**文件**：`report.py:10-174`

`write_result`、`save_state`、`_stage_lines`、`_planned_experiment_lines`、`_coding_agent_lines` 这些函数为 `ReproState`（旧架构）服务，新架构只用 `write_agent_result`。

但 `_coding_agent_lines` 被 `write_agent_result` 复用（line 230），所以不能全删。`_stage_lines` / `_planned_experiment_lines` 可以删除。

---

### 10. context.py 的文件树 limit 硬编码

**文件**：`context.py:241`

```python
def _file_tree(repo_path: Path, limit: int = 250) -> str:
```

250 是固定的，不受 ContextPolicy 控制。大的 monorepo 可能需要截断更多。

---

## 严重度汇总

| 等级 | 问题 | 影响 |
|------|------|------|
| 🔴 中 | OMP_NUM_THREADS 硬编码 | 性能受影响 |
| 🟡 低 | audit.py 只检查 PyTorch | 非 PyTorch 项目审计不准 |
| 🟡 低 | CodingAgent 验证命令空 | 不影响正确性 |
| 🟡 低 | build_initial_context 未用 policy readme_chars | 与大窗口模型不匹配 |
| 🟡 低 | build_turn_prompt 丢 paper/repo URL | agent 丢失任务上下文 |
| ⚪ 洁癖 | _MAX_STEPS 死代码 | 无影响 |
| ⚪ 洁癖 | _update_file_cache 正则不全 | 边际收益 |
| ⚪ 洁癖 | PIP_CACHE_DIR 隔离 | 需用户显式配置 |
| ⚪ 洁癖 | report.py 旧函数残留 | 代码臃肿 |
| ⚪ 洁癖 | _file_tree limit 硬编码 | 大仓库可能溢出 |

---

## 不需要改的

| 项 | 原因 |
|----|------|
| `_mirror_block` 硬编码 aliyun URL | 镜像本身就是配置项，"autodl" profile 的行为就是使用 aliyun |
| `MODEL_CONTEXT_WINDOWS` 硬编码模型名 | 这是模型能力表，新增模型只需加一行 |
| `_command_env` 默认 `api_base=https://api.openai.com/v1` | 此字段是 CLI 参数，默认值 = 最常用的 |
| Legacy types (ReproState, CommandPlan, StageResult) 不能删 | `env.py`、`audit.py`、`coding.py` 的参数签名依赖它们 |
| `_is_probe_command` 白名单有限 | 安全策略，宁紧勿松 |
