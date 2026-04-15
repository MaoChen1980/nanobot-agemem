# TaskTree 分层规划 — 实现任务

## 目标

将 nanobot 从单层扁平 ReAct loop 升级为 TaskTree + Depth-First + Replan 模式。

**设计文档**：`docs/tasktree-design.md`

---

## 实现顺序

### Phase 1：骨架（无外部依赖，先跑通流程）

- [x] **T1.1** 实现 `TaskNode` dataclass + `TaskStatus` enum + `RootCause` + `WorkspaceState` + `Artifact`
- [x] **T1.2** 实现 `TaskTree` 类（add_child / get_path / pick_deepest_pending / is_done / mark_* / to_dict / from_dict）
- [x] **T1.3** 实现 `NodeResult` dataclass（成功结果）
- [x] **T1.4** 实现 `FailureReport` dataclass
- [x] **T1.5** 实现 `ConstraintSet` dataclass

### Phase 2：接口抽象层

- [x] **T2.1** 定义 `ExecutionAgent` Protocol（scheduler.py 内）
- [x] **T2.2** 定义 `ConstraintAgent` Protocol（scheduler.py 内）
- [x] **T2.3** 定义 `VerificationAgent` Protocol + `VerificationResult`
- [x] **T2.4** 定义 `SubgoalParser` Protocol + `LLMSubgoalParser`

### Phase 3：Scheduler 状态机

- [x] **T3.1** 实现 `TaskScheduler` 主循环
- [x] **T3.2** 实现 `pick_deepest_pending` 的 depth-first 逻辑（tree.py 内）
- [x] **T3.3** 实现父节点 replan 逻辑（MAX_CHILDREN=10 边界检查）
- [x] **T3.4** 实现失败上报链（子节点失败 → 父节点决策 → 向上传播）+ bubble_up
- [x] **T3.5** 实现 `build_node_context()` + `build_result_from_agent_response()` + `build_failure_from_error()`

### Phase 4：现有系统集成

- [x] **T4.1** 封装现有 `AgentRunner` 为 `DefaultExecutionAgent`
- [x] **T4.2** 实现 `DefaultConstraintAgent`（同模型 LLM call，返回 ConstraintSet）
- [x] **T4.3** 实现 `LLMVerificationAgent`（LLM_JUDGE 策略）
- [x] **T4.4** TaskTreeService（独立服务 + /taskstatus + /taskcancel + 用户介入 + 断后延续）
- [x] **T4.5** 节点执行结果写入 SessionManager（SessionPersistenceCallback）

### Phase 5：Memory 集成

- [x] **T5.1** 节点失败时查 MemoryRetriever 找相似 root cause 经验（DefaultConstraintAgent._apply_memory_veto）
- [x] **T5.2** 相似 root cause 出现 → 收紧 failure_count_limit（N=10，相似失败越多允许重试越少）
- [x] **T5.3** 节点完成后 summary 写入 MemoryStore（MemoryCallback）

---

## 设计决策（已确认）

| 参数 | 值 |
|------|---|
| MAX_CHILDREN | 10 |
| Constraint Agent 模型 | 同模型 |
| Verification 策略 | LLM_JUDGE |
| 节点 token 预算 | 无限（由 depth 约束） |

---

## 文件结构（已实现）

```
nanobot/
    bus/
        __init__.py      # MessageBus, RouterBus, InboundMessage, OutboundMessage
        router.py        # RouterBus（路由层：predicates → 独立 consumer 队列）
    agent/tasktree/
        __init__.py          # 统一导出
        models.py            # TaskNode, TaskStatus, NodeResult, FailureReport,
                             # Artifact, RootCause, WorkspaceState, ConstraintSet
        tree.py              # TaskTree（add_child, pick_deepest_pending, bubble_up, serialize）
        scheduler.py         # Scheduler 主循环 + 全部 Protocols
                             # (ExecutionAgent, ConstraintAgent, SubgoalParser,
                             #  SchedulerCallbacks, VerificationAgent)
                             # 支持 resume_tree 参数断后延续
        context.py           # build_node_context() + build_result/failure_from_*()
        callbacks.py         # SessionPersistenceCallback（写入 session + tree checkpoint 保存/恢复）
        memory_callback.py   # MemoryCallback（写入 AgeMem MemoryStoreV2）
        service.py           # TaskTreeService（独立服务：
                             #  submit / cancel / get_status / request_user_input / 断后延续）
        execution/
            __init__.py
            default.py      # DefaultExecutionAgent（封装 AgentRunner）
            constraint.py   # DefaultConstraintAgent（LLM + MemoryRetriever guided veto，N=10）
            verification.py # LLMVerificationAgent（LLM_JUDGE 策略）
            subgoal.py      # SubgoalParser, LLMSubgoalParser（JSON/numbered/markdown list 检测）
    nanobot.py               # RouterBus 集成 + run_tasktree / get_task_status / cancel_task
```

**已全部实现** ✓

---

## CLI 接入（已完成）

- [x] `nanobot agent -m "/plantask <goal>"`：单消息模式，提交后 polling 等待完成
- [x] `nanobot agent`（交互模式）：RouterBus + TaskTreeService，识别 `/plantask` 自动路由
- [x] `/taskstatus`：交互模式下查询 TaskTree 进度
- [x] `/taskcancel`：交互模式下取消运行中任务
- [x] RouterBus.fallback 队列：AgentLoop 读非路由消息，与 TaskTreeService 隔离

**注**：`gateway` / `serve` 命令已接入 RouterBus（见 T6）。

---

## Scheduler 修复（已补）

- [x] `_execute_node()` 传递 `tree=self._tree` 到 `execution_agent.execute()`，确保每个节点（含根节点）context 带完整 task block（Root Goal / Parent Task / Parent Result / Constraints / Your Task）

---

## 待攻克的开放问题

- [x] Workspace 污染处理（TaskNode + NodeResult 新增 `workspace_state: WorkspaceState` 字段；DefaultExecutionAgent._extract_artifacts() 扫描 tool_calls 检测写文件/环境修改；mark_pending() 时重置；mark_done() 时从 result 同步；snapshot/可逆标记方案暂不实现，先用"dirty"状态提示用户）
- [x] 根节点第一次拆解 prompt（`[Decomposition Instruction]` block 注入 task block，包含结构化输出格式要求和"不要问问题"的指令）
- [x] Root Context 组装增强（新增 `[Root Planning Context]` 块标识 depth=0、workspace_state；Decomposition Instruction 更详细，包含 JSON/numbered/markdown 格式示例、最大 subtask 数=10、不自己执行只规划等规则）
- [x] Verification 失败后子节点 replan prompt 生成（verification phase 在 Scheduler.run() 末尾执行；失败时 mark_pending + 注入 `verification_failure` 到 node context；`max_verification_retries=3` 防止无限循环）

---

## 待完成任务

### T6：gateway / serve 接入 RouterBus

- [x] `gateway` 命令（`nanobot gateway`）：改为 RouterBus + AgentLoop + TaskTreeService，识别 `/plantask` 路由到 TaskTreeService
- [x] `serve` 命令（`nanobot serve`）：同上
- [x] 两个命令共用水总线、session manager、provider 等共享组件

### T7：Rich console double output

- [x] `_consume_outbound()` 里同一 OutboundMessage 被渲染多次（可能是 Rich console ANSI capture 对长文本的渲染行为）
- [x] 根因：`_consume_outbound()` 没有过滤 `_tasktree_progress` 消息，或者 `_print_agent_response` 被多次调用

### T8：request_user_input CLI 端未接入

- [x] `request_user_input()` 发出 `🤔 需要你的介入：xxx` 后，CLI 侧没有接收和处理
- [x] 交互模式下检测 `metadata._tasktree_needs_input: True`，暂停输入，等待用户响应，调用 `tasktree_service.submit_user_input(chat_id, response)`

### T8.1：TaskTree 后台任务人机交互

- [x] TaskTree 后台执行中发 `🤔 需要你的介入：xxx` 到 CLI 输出（通过 `_tasktree_needs_input: True` metadata）
- [x] CLI 交互模式检测到此 metadata 时：停止 spinner，输出问题，等待用户输入一行文本
- [x] 用户输入后调用 `tasktree_service.submit_user_input(chat_id, response)`
- [x] TaskTree 收到响应后继续执行（ExecutionAgent.await `_input_results`）
- [x] 支持多轮交互（同一个节点可能多次请求用户输入）

### T8.2：任务确认与信息丰富（TaskTree 启动阶段）

- [x] 根节点执行前：LLM 重新复述任务（paraphrase），生成结构化任务描述
- [x] 通过 CLI 输出复述后的任务描述，询问用户确认或补充
- [x] 用户可以：确认 / 修改 / 补充额外信息 / 取消任务
- [x] 确认后的任务描述作为 Root Goal 注入 TaskTree，取代原始模糊输入
- [x] 降低因任务不清晰、理解偏差导致的执行失败概率
- [x] 设计要点：这是一个 pre-execution 阶段，在 Scheduler.run() 的根节点执行前插入 LLM 复述 + 人机讨论循环

### T9：单元测试

- [x] `tests/unit/agent/tasktree/test_models.py`：TaskNode / TaskStatus / RootCause / Artifact / NodeResult / FailureReport / ConstraintSet 序列化与反序列化
- [x] `tests/unit/agent/tasktree/test_tree.py`：TaskTree.add_child / pick_deepest_pending / mark_* / bubble_up / is_done / to_dict / from_dict
- [x] `tests/unit/agent/tasktree/test_scheduler.py`：Scheduler 主循环（mock ExecutionAgent/ConstraintAgent/SubgoalParser），depth-first 调度、bubble_up、replan、verification 重试
- [x] `tests/unit/agent/tasktree/test_context.py`：`build_node_context()`（根节点有 Decomposition Instruction / 子节点有 Parent Result / 失败重试有 verification_failure block）
- [x] `tests/unit/agent/tasktree/test_subgoal.py`：`LLMSubgoalParser.parse()`（JSON / numbered list / markdown list / 无结构）
- [x] `tests/unit/agent/tasktree/test_execution.py`：`DefaultExecutionAgent._extract_artifacts()`（write_file / edit_file 映射）
- [x] `tests/unit/agent/tasktree/test_verification_agent.py`：`LLMVerificationAgent.verify()` / `_parse_verification_response` / `_build_verification_prompt`
- [x] `tests/unit/agent/tasktree/test_constraint_agent.py`：`DefaultConstraintAgent.get_constraints()` / `_apply_memory_veto` / `_parse_constraint_response`
- [x] `tests/unit/agent/tasktree/test_memory_callback.py`：`MemoryCallback.on_node_done/failed/blocked` / 异常处理
- [x] `tests/unit/bus/test_router.py`：`RouterBus.register_route` / `start_router` / `_route_loop`（fallback 队列、predicate 匹配）

### T10：集成测试

- [x] `tests/integration/tasktree/test_end_to_end.py`：
  - RouterBus 路由：/plantask 消息 → TaskTreeService，非 /plantask → AgentLoop
  - TaskTree 完整执行：submit → depth-first 执行 → verification → 返回结果
  - 断后延续：checkpoint 保存后恢复，继续执行
  - `/taskstatus` / `/taskcancel` 命令
  - `request_user_input` 流程：TaskTree 请求用户输入 → CLI 接收并响应 → 继续执行
- [x] `tests/integration/tasktree/test_verification_retry.py`：
  - Verification 失败 → 重试指定节点 → 重新验证（最多 3 次）
  - Verification 通过 → 返回最终结果
- [x] `tests/integration/tasktree/test_replan_flow.py`：
  - 节点失败（非 constraint_veto）→ 父节点 spawn sibling
  - MAX_CHILDREN 边界：已达上限则父节点失败向上传播
  - bubble_up：所有子节点 terminal 后评估父节点状态
