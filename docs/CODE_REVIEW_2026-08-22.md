# ReproAgent 只读审查报告（四模块整理 · Phase B）

> 日期：2026-08-22
> 分支：codex/readability-cleanup（自 main @ 6054b0b 创建）
> 范围：src/reproagent 全部生产代码 + tests 对应关系
> 状态：只读审查，未修改任何生产代码

## 1. 真实入口

| 入口 | 位置 | 说明 |
|---|---|---|
| CLI | `reproagent.main:main` | 子命令 run / resume / list / status / inspect / prune |
| 公共 API | `reproagent.__init__` | 导出 ReproTask / ReproState / CommandPlan / CommandResult（Phase 0 锁定） |
| 编排模式 | `reproagent.agent.run_task` / `resume_task` | ResAgent 通过 `reproagent.controller.run_controller` 直接调用（adapters/reproagent.py:169,257），并 import `reproagent.models.{ReproTask,AgentState}` |
| 资源管理 | `reproagent.runtime.env_manager.delete_environment` | ResAgent cleanup.py:258 消费（跨模块契约） |
| 能力卡 | `agent.yaml` | ResAgent CapabilityRegistry 加载（能力词表冻结） |

无其他入口（ExpAgent/CodingAgent 不 import reproagent；CodingAgent 仅共享 `_vendor/env_contract_v1.py` 字节一致副本）。

## 2. 主流程调用图（真实生产路径）

```text
main.run ──► agent.run_task ──► controller.loop.run_controller
                                    │
   (非 mock) ├─ repository.context.collect_context      # setup_workspace(三种模式) + git/file-tree/README/hardware
             │    └─ runtime.hardware.collect_hardware_text
             ├─ controller.loop.ensure_environment_for_controller
             │    └─ runtime.environment.ensure_environment
             │         ├─ [content_addressed] _ensure_content_addressed ── runtime.env_identity / env_manager（确定性，无 LLM）
             │         └─ [legacy 默认] env_name 绑定 → 命名 → conda create（未变）
             ├─ runtime.dataset_cache.prepare_dataset_links（best-effort，不致命）
             └─ loop：prompts.build_initial_context/build_turn_prompt
                    ─► llm.call_llm（mock 或 OpenAI 兼容）
                    ─► actions._parse_action ─► 三工具：
                          _tool_run_commands  ── runtime.runner.run_commands（安全策略+白名单+门控）
                          _tool_audit_env     ── runtime.audit.audit_environment
                                                 └─ env_manager.check_task_spec_compliance + finalize_manifest_after_audit
                          _tool_call_coding_agent ── integrations.codingagent.run_coding_agent_for_patch
             └─ finish → report.write_agent_result（result.md + result.json + 证据冻结 + state.json）
                      → session.write_session_card（session.yaml）
```

resume：`agent.resume_task` 读 state.json 重建 ReproTask（透传全部 M2 字段）→ 同一 loop，复用 workspace/env。

## 3. 主要文件职责表

| 文件 | 职责 | 存在理由 |
|---|---|---|
| controller/loop.py | 状态机主循环（init/resume、步骤预算、finish 门控、报告落盘） | 唯一主循环 |
| controller/actions.py | LLM action 解析 + 三个工具执行 + 认证门控/变更策略 | 决策边界：门控全部确定性 |
| controller/prompts.py | SYSTEM_PROMPT + 上下文构建 | Prompt 唯一所在地 |
| runtime/runner.py | 命令安全分析（shlex）、白名单、执行/日志/超时 | 唯一命令执行通道 |
| runtime/environment.py | conda 发现、legacy 环境创建/绑定、content-addressed 复用或创建 | 环境准备唯一所有者 |
| runtime/env_manager.py | manifest 状态机、创建锁、库存采集、审计定稿、清理 plan/apply | M2 资源管理唯一所有者 |
| runtime/env_identity.py | spec 收集（任务级） | 指纹算法本体在 _vendor |
| _vendor/env_contract_v1.py | ENVIRONMENT_*_V1 契约算法（与 CodingAgent 字节一致） | 跨仓唯一实现，禁改 |
| runtime/audit.py | 环境审计探针（python/pip/torch/GPU） | 认证门控的数据来源 |
| runtime/dataset_cache.py | 数据集缓存桥接（scan→resolve→link→render） | 确定性桥接，LLM 不可替代 |
| repository/context.py | workspace 三模式 + clone/缓存 + 上下文收集 | 仓库来源唯一所有者 |
| report.py | result.md/result.json/证据冻结/state.json | 证据冻结唯一所有者 |
| session.py | session.yaml 写/扫描/状态 | 跨模块发现契约 |
| integrations/codingagent.py | CodingAgent 定位/导入/补丁编排 | 唯一跨模块边界 |
| llm.py | OpenAI 兼容 HTTP 传输 + 重试 + trace | 纯传输层 |
| models.py | 全部 Pydantic 模型（含 legacy） | 序列化契约 |
| text.py | 文本规范化（mojibake 修复） | 报告安全 |
| main.py | CLI | 唯一 CLI |

## 4. 状态读写位置

| 状态 | 写 | 读 |
|---|---|---|
| AgentState（内存/state.json） | loop.py（每步、finalize）；report._save_agent_state | prompts.build_turn_prompt、actions 门控、session/report |
| manifest.json | env_manager（原子写） | environment._ensure_content_addressed、CLI inspect/prune、session bindings |
| 创建锁 | env_manager | environment 创建路径、plan_cleanup |
| session.yaml | session.write_session_card（loop 结束） | ResAgent 扫描、CLI list/status |
| result.json/result.md | report.write_agent_result | ResAgent adapter、ExpAgent |
| 环境变量 REPROAGENT_DATASET_CACHE | loop（设置/恢复） | runner._command_env、session._resolve_pip_cache |
| os.environ（PIP_CACHE_DIR 等） | runner._command_env（每命令子进程） | 子进程 |

## 5. 外部输入输出与副作用

- 输入：CLI 参数 / ReproTask（ResAgent adapter 构造）；repo（clone/copy/external）；conda；LLM API。
- 输出：workspace（state.json、result.md/json、session.yaml、logs、patches/、evidence/）；resource_root（environments/<env_id>/、locks/、conda-envs/）；conda 环境本体。
- 副作用：conda create、pip 缓存目录创建、数据集符号链接、os.environ 变更（受控恢复）。
- 确定性边界：fingerprint/manifest/锁/门控/清理候选/数据集桥接全部为纯代码，LLM 不参与任何决策。

## 6. 问题清单（证据见行号）

### A. Split Mainline / Legacy 并存（高）

**A1. legacy 报告写入路径 `write_result` + `save_state`**
- 位置：report.py:13-93（write_result）、report.py:92（调用 save_state）
- 现状：与主线 `write_agent_result`（report.py:126）平行存在，是旧 linear-workflow 的渲染器。
- 真实调用路径：生产代码 0 调用（loop.py:267 只走 write_agent_result）；唯一消费者是 tests/test_report.py:36-90 三个测试。跨模块已验证：ResAgent/ExpAgent/CodingAgent 均无 import（grep 证据）。
- 为什么是问题：同一"写 result.md + state.json"存在两套实现，新开发者需要理解哪条是真。
- 是否改变行为：删除后生产行为不变（无生产调用者）。
- 最小处理方案：删除 write_result/save_state 及 test_report.py 中三个专属测试（其断言的行为在 write_agent_result 测试中已覆盖：experiment goal/codingagent path/version 渲染）。
- 删除风险：低（见上方调用路径证据）。
- 验收：全量测试绿；git diff --check。

**A2. legacy 状态模型 ReproState 充当生产函数签名 + AgentState 三个兼容 property**
- 位置：models.py:198-211（property 块）；models.py:241-255（ReproState）；environment.py:19、audit.py:58、codingagent.py:201、report.py:13,21（签名）
- 现状：生产代码从不实例化 ReproState（src/ 内 grep 零 `ReproState(`）；所有生产调用传入 AgentState，靠 property 别名（environment_audit→last_audit、coding_agent_results→coding_results、probe_attempts→[]）鸭子类型桥接。
- 真实调用路径：
  - property 的唯二消费者在 integrations/codingagent.py:205（coding_agent_results）、285-286（environment_audit）、302（probe_attempts，且因恒空是死循环）。
  - 跨模块已验证：ResAgent adapter 只访问 result_state.status/final_summary/steps/repo_context（adapters/reproagent.py:209-227,293-295），不用三个 property。
- 为什么是问题：旧线性 workflow 的模型与 agent-loop 模型并存，同一状态两种表达；property 别名掩盖了类型不一致（函数签名说 ReproState、实际传 AgentState）。
- 是否改变行为：签名修正与字段直用零行为变化；删除 property 不影响序列化（pydantic property 不进入 model_dump/state.json）；唯一对外可见变化是 Python 属性访问 `AgentState().environment_audit` 等——已验证无外部消费者。
- 最小处理方案（分两个提交，先签名后删 property）：
  1. environment.py/audit.py/codingagent.py 签名 `ReproState`→`AgentState`；codingagent.py 直用 last_audit/coding_results；删除 probe_attempts 死循环（codingagent.py:302-306）；
  2. 删除 models.py 三个 property；相关测试构造迁移 ReproState→AgentState。
- 删除风险：中——测试构造迁移涉及 test_env.py、test_audit.py、test_codingagent_integration.py、test_content_addressed_flow.py、test_report.py；每处仅换载体类型，行为断言不变。
- 验收：全量测试绿；test_phase0_contract（模型字段列表）不受影响（property 不在 model_fields）。
- 注意：ReproState / CommandPlan / StageResult 定义**保留**（`reproagent.__all__` 为 Phase 0 公共契约锁定；CommandPlan 仍被 actions.py:272-277 构造使用；StageResult 无任何实例化但属 ReproState 定义的一部分，随模型保留）。

### B. 死代码（安全删除）

| ID | 位置 | 证据（全仓 grep 含 tests，零引用） |
|---|---|---|
| B1 | text.py:90-101 normalize_plan_text / normalize_text_list | src + tests 均无 import/调用 |
| B2 | prompts.py:291-295 _cache_line | 零引用；docstring 自标 Deprecated |
| B3 | session.py:118-120 update_session_card | 零引用（write_session_card 的一次别名包装） |
| B4 | env_identity.py:83-100 _probe_nvidia_smi | 生产零调用——_accelerator_spec（env_identity.py:120）实际走 _contract.probe_gpu_usable；仅 tests/test_env_identity.py:166-228 引用 |
| B5 | codingagent.py:302-306 probe_attempts 循环 | AgentState.probe_attempts property 恒空列表，循环体不可达（并入 A2 删除） |
| B6 | context/__init__.py:10-12 legacy 重导出（clone_repo/collect_context/setup_workspace） | 原判定"零消费者"并删除。**已修正（RP2）**：外部消费者（flat context.py 时代的公共导入面）仍依赖该路径，按接口稳定性要求恢复——从 repository.context 重导出、不复制实现，identity 测试锁定；这不是恢复旧线性 workflow（该 workflow 的生产代码已不存在） |

### C. 重复实现（Redundancy）

| ID | 位置 | 现状 |
|---|---|---|
| C1 | env_manager.py:369-371 `_task_run_id` vs environment.py:237-239 `_run_id` | 逐字重复的 parent_run.run_id 提取 |
| C2 | env_manager.py:29-30 `utcnow` vs session.py:263-264 `_utcnow` | 逐字重复的 UTC 时间戳格式化 |
| C3 | env_identity.py:103-106 `_constraint_cuda_variant`、135-137 `_dependency_files` | 单一调用点的 vendor 委托薄壳，可内联（与 B4 合并为"vendor 委托收敛"一个主题提交） |

### D. Correctness / Test Gap

**D1. 加速器测试的 monkeypatch 打在死代码上，测试通过依赖宿主 GPU（高）**
- 位置：tests/test_env_identity.py:173-201（monkeypatch `module._probe_nvidia_smi`）；生产路径 env_identity.py:120 调用 `_contract.probe_gpu_usable()`。
- 现状：monkeypatch 不作用于真实路径；`test_accelerator_requires_gpu_with_gpu_yields_cuda` 与 `test_accelerator_variant_from_explicit_constraint` 只有在宿主机器存在可用 nvidia-smi 时才通过（本机 WSL 有 GPU 直通，恰好通过；无 GPU CI 上必失败）。
- 最小处理方案：删除 _probe_nvidia_smi（B4）后，测试 monkeypatch 目标改为 `env_identity._contract.probe_gpu_usable`；`test_probe_nvidia_smi_robust_header_parse` 改为直接断言 vendor `probe_gpu_usable` 的 banner 规则（fake nvidia-smi 脚本保留）。
- 风险：低（纯测试修正 + 死代码删除）。
- 验收：该测试文件在任何机器（含无 GPU）上绿。

### E. 行为风险修改（本轮只报告，单独审批）

**E1. CLI `--timeout` 默认 1800 与模型默认 3600 不一致**
- 位置：main.py:39（default=1800）vs models.py:34（timeout_seconds: int = 3600）。
- 现状：commit cfa8843 按 CAPABILITY_TEST_FINDINGS_L2_SEBLOCK 要求把默认超时提到 3600，但只改了模型默认；CLI 用户仍拿到 1800。同一概念两个默认值，且与文档意图（3600s）相悖。
- 是否改变行为：CLI 对齐到 3600 会改变 CLI 行为（属方案 Phase C 表中 CLI 类）。
- 建议：本轮不改或经批准后一行对齐。

**E2. python 版本身份规则与 vendored 契约的 `select_python_version` 不一致**（**已修复：RP1**）
- 根因：本地 `_normalize_python_version` 只取 `task.python_version` 的 major.minor（缺省 3.10），完全不看仓库的 environment.yml pin。结果是：同一仓库换 Python 声明版本，spec identity 不变——环境身份与实际仓库 Python 要求脱节；且同一条"Python 版本选择"规则在 vendor 契约（task > environment.yml pin > default）与本地实现（仅 task/default）之间漂移。
- 修复：`collect_environment_spec` 改调 `_contract.select_python_version(task.python_version, repo_path)`（唯一规则，vendor 字节一致文件），删除本地 `_normalize_python_version` 及其 re import。测试覆盖：显式优先 / environment.yml pin 被读取 / 两者皆无走契约默认 / 指纹随 Python 版本变化。契约文件未改动（跨仓字节一致不变）。

### F. 观察项（不做修改）

- models.py 注释"compatibility aliases for infrastructure modules"已不准确（消费者仅本仓 codingagent.py，A2 处理）。
- 若干 docstring 笔误（"Write write blocked result" runner.py:407、"Load load config" codingagent.py:116、"Return whether has parent directory traversal" runner.py:206 等）——随所在提交顺带修正，不单独提交。
- `_INTERNAL_ACTIONS`、setup 白名单、认证门控、变更策略全部集中于单一实现（runner.py / actions.py），无第二套——符合主线要求。
- agent.yaml 能力词表与 V2 契约一致；session bindings 职责清晰；fingerprint/manifest/lock/清理候选全部为确定性代码。

## 7. 测试与生产路径对应关系（Explore 子代理全量核对）

- 当前 20 个测试文件 / 224 用例，无 conftest；fixtures 仅 `tests/fixtures/m2/`（P0 契约复制）。
- 生产模块覆盖矩阵：22 个模块被直接 import；7 个无直接测试：各空壳 `__init__`、`agent.run_task/resume_task`（CLI 真入口零测试，测试全直连 run_controller）、`llm.call_llm` 真实 HTTP 路径（仅 mock 分支被测）、`context/__init__` legacy shim（对应测试已随重构删除）。
- 行为级测试约 40–45%（test_workflow 15 条主线+门控、test_session 12 条 run_controller、test_content_addressed_flow 14 条 ensure_environment 含并发、test_audit 6 条、test_context 17 条真实 git）；实现细节测试约 55–60%（runner 谓词、env_identity golden、env_manager 状态机等）。真实进程级集成约 10 条。
- 已核对：当前 20 个文件的全部 import 与 monkeypatch 目标均存在于 src——**没有 import 已删模块的死测试**。
- 环境与耦合注意：
  - `tests/__pycache__` 残留已删除文件（test_validation.py、test_compat_shims.py）的 .pyc 与 `.pytest_cache` 旧 nodeid（281 vs 224）——非 git 跟踪，顺手清理即可；
  - test_vendor_contract.py 依赖 sibling ResAgent 仓库的契约文件（跨仓字节一致测试，有意为之，保留）；
  - test_models.py:8-14 保留 unittest.TestCase 风格残留（含无用的 `unittest.main()` 块）——安全整理候选；
  - test_hardware.py 与 D1 加速器测试依赖宿主 GPU 环境（D1 将修复，hardware 冒烟保留）。
- 覆盖缺口（本轮不新增大规模测试；仅补「证明行为不变」的最小入口测试）：`agent.run_task` 公共入口零测试——Phase D 补 1 条 mock-LLM 冒烟（同时满足 Phase E 公共入口导入测试验收）。

## 8. 处理计划（Phase C/D 预案，待用户确认后执行）

推荐提交序列（每提交一个主题，全量测试绿）：

1. `remove dead code: text helpers, prompts._cache_line, session.update_session_card, context legacy re-exports`（B1/B2/B3/B6）
2. `converge vendor delegation in env_identity: drop _probe_nvidia_smi and thin wrappers`（B4+C3）+ 测试修正（D1：monkeypatch 目标改 `_contract.probe_gpu_usable`）
3. `retire legacy report writer write_result/save_state`（A1，含删 3 测试）
4. `type agent-loop functions as AgentState; drop legacy compatibility properties`（A2，签名+直用字段+删 property+删死循环+测试载体迁移；补 agent.run_task mock 冒烟）
5. `dedupe run_id/utcnow helpers`（C1/C2）
6. （若批准）`align CLI default timeout to 3600s`（E1，行为修改单独提交）

## 9. 执行结果（Phase D/E 已完成，2026-08-22）

实际提交（codex/readability-cleanup，自 main @ 6054b0b）：

| commit | 主题 | 变更 |
|---|---|---|
| 9b18a5c | remove dead code（B1/B2/B3/B6） | text×2、prompts._cache_line、session.update_session_card、context legacy 重导出；5 文件 |
| d163b30 | converge vendor delegation in env_identity（B4/C3/D1） | 删 _probe_nvidia_smi 与薄壳 + 未用常量；测试 monkeypatch 指向真实路径 `_contract.probe_gpu_usable`，任何机器可跑；3 文件 |
| 5e5ba8a | retire legacy report writer（A1） | 删 write_result/save_state + 3 测试；-141 行 |
| da66eae | type agent-loop functions as AgentState（A2） | environment/audit/codingagent 签名；删 3 个兼容 property；删 probe_attempts 死循环；测试载体迁移；补 agent.run_task 入口冒烟；10 文件 |
| 7235ac8 | dedupe run_id/utcnow helpers（C1/C2） | environment.task_run_id 唯一实现；session 复用 env_manager.utcnow；3 文件 |
| 4cf9e96 | align CLI default timeout to 3600s（E1，已批准） | main.py + resume 兜底 + 防回归测试；3 文件 |

验收（Phase E）：

- 全量测试：224 → **223 passed**（-3 legacy writer 测试、+2：run_task 入口冒烟、CLI 默认一致性）；`git diff --check` 通过。
- 公共入口导入：`reproagent` / controller / runtime / integrations / _vendor 全部模块导入通过，`__all__` 契约不变。
- CLI smoke：`reproagent --help` / `run --help` / `inspect --help` 正常。
- 依赖：pyproject.toml 未动（pydantic + pyyaml），无新增生产依赖。
- 规模：24 文件变更，+299/-316（其中 ~190 行为本报告）；生产代码 5628 → 5453 行（-175，-3.1%）；生产文件数 27 → 27。
- 未处理风险：E2（python 版本身份规则 vs vendored select_python_version）按方案 §8 只报告，由总体审查决定；StageResult 随 ReproState 定义保留（公共契约）。
- 工作区：干净；未合并默认分支，等待总体审查。

## 10. 后续任务单执行（RP1–RP3，2026-08-22）

| commit | 主题 |
|---|---|
| 1b9d1f4 | RP1：spec 收集改用 `_contract.select_python_version`（task > environment.yml pin > 契约默认）；删 `_normalize_python_version`；4 个新测试 |
| 363487e | RP2：恢复 `reproagent.context` 公共重导出（指向 repository.context 实现，identity 测试锁定）；新增 tests/test_public_imports.py |
| 本提交 | RP3：报告补充根因与保留理由 |

### 旧 workflow 与公共兼容入口的区别

- **旧线性 workflow**（plan → validate → stage 执行）：生产代码已完全删除，本任务单不恢复任何部分。
- **公共兼容入口**（`reproagent.context` 重导出、`reproagent.__all__` 中的 ReproState/CommandPlan）：只是接口稳定性承诺——外部 import 不因内部重组而断裂。它们指向当前唯一主线实现（repository.context、agent-loop 状态机），不携带任何旧行为。
- 判定标准：兼容入口必须是"重导出当前实现"（RP2 的 identity 测试锁死这一点）；任何要求复制旧渲染/旧状态机逻辑的都是 workflow 恢复，拒绝。

### 最终测试结果（2026-08-22）

- **229 passed**（RP1 +4、RP2 +2；相对整理基线 224：-3 legacy writer、+8 新增）
- `git diff --check` 通过；`from reproagent.context import clone_repo, collect_context, setup_workspace` 验证通过
- environment.yml python pin 进入 ENVIRONMENT_SPEC_V1 与 spec_fingerprint（test_python_version_reads_environment_yml_pin / test_python_version_joins_identity）
- Prompt、外部契约（agent.yaml / __all__ / 模型字段顺序）、依赖（pyproject.toml）均未改动
- 分支 codex/readability-cleanup 已推送，未合并 main
