# TaskTree 分层规划 — 设计讨论记录

> 本文档记录从"扁平 ReAct loop 问题"到"TaskTree 分层规划"的设计迭代过程。

---

## 问题起点

**用户观察到的问题**：

1. Nanobot 和 LLM 交互是扁平结构，无法处理长任务
2. 跑多了就跑偏，陷入细节
3. 无法约束细节的颗粒度

**根本原因**：单层 ReAct loop 本质是线性循环，每步把所有历史、记忆、skill 全量塞入 context，没有任务分解、目标追踪、深度控制的机制。

---

## 初始提案：TaskTree + Depth-First + Replan

### 核心思路

```
TaskTree 是锚 — 初始目标不变，最后回溯验证
Depth-first 是路径 — 最小 context 路径，避免干扰
Replan 是逃生舱 — 在 tree 约束下保留灵活应变能力
摘要上传是缩小 context — 避免污染其他分支
```

---

## 设计攻击与迭代（摘要）

| 攻击点 | 最终决定 |
|--------|---------|
| Tree 结构能动态删节点吗？ | **不可以** |
| Tree 执行模式 | **父子链式深度执行**，不是 DFS 搜索回溯 |
| Replan 谁来驱动？ | **父节点驱动**，根据子节点失败 JSON 决定 |
| Failure Report 格式？ | **固定 JSON Schema** |
| Failure 重试边界？ | **LLM 决定 + Constraint Agent 硬否决**（两个独立 Agent） |
| Root 验证方式？ | **Execution Agent / Verification Agent 分离** |
| 验证失败后怎么处理？ | **仅重跑失败分支，不重跑整个 tree** |

---

## 最终架构

```
用户任务 (Root Goal)
    │
    ▼
TaskScheduler（状态机，驱动执行）
    │
    ├─► Constraint Agent（获取约束）→ 约束包 ConstraintSet
    │
    ├─► Execution Agent（执行节点）→ NodeResult 或 FailureReport
    │
    └─► Verification Agent（独立验证）
            ├─ 验证通过 → 任务完成
            └─ 验证失败 → 仅重跑失败分支
```

---

## 接口设计

### IPC 决策

- **IPC 方式**：async function call，复用现有 asyncio 模式，不引入 MQ
- **Execution Agent**：protocol/interface，底层封装现有 `AgentLoop.process_direct()`
- **Constraint Agent**：同步 LLM call（小 prompt + 快模型），执行前获取
- **Verification Agent**：独立 LLM call，所有 execution 完成后的单独阶段
- **子节点生成**：父节点 LLM 输出包含子节点列表，Scheduler 解析并插入树

---

### 数据结构

#### TaskNode

```python
class TaskNode:
    id: str                           # 格式："A1", "A2"，唯一标识
    goal: str                         # 任务描述
    status: TaskStatus                # pending | running | done | failed | blocked
    parent_id: str | None             # 父节点 ID，root 为 None
    children: list[str]               # 子节点 ID 列表
    depth: int                        # 深度，root = 0
    result: NodeResult | None         # 成功结果
    failure: FailureReport | None     # 失败报告
    replan_count: int                # 被父节点 replan 的次数
    created_at: float                 # 创建时间戳
```

#### NodeResult（成功结果）

```json
{
  "node_id": "A1",
  "status": "done",
  "summary": "完成了用户认证模块的 API 封装",
  "artifacts": [
    {
      "type": "file_written",
      "path": "src/auth/api.py",
      "description": "认证 API 封装，包含 login/logout/refresh"
    }
  ],
  "constraints_respected": true,
  "token_spent": 3200
}
```

#### FailureReport（失败报告，固定 Schema）

```json
{
  "node_id": "A1",
  "status": "failed",
  "root_cause": "api_timeout | file_not_found | constraint_veto | no_remaining_options | max_replan_reached",
  "summary": "第三方 API 超时，重试 3 次后仍不可用",
  "tried": ["尝试官方 API", "尝试备用 endpoint", "尝试降级 mock"],
  "remaining_options": [],
  "impact_on_parent": "A 的后续子节点不依赖此 API，可跳过",
  "constraint_veto": false,
  "workspace_state": "partial",
  "token_spent": 1800
}
```

**`root_cause` 枚举**：
- `api_timeout`：外部 API 超时
- `file_not_found`：文件/资源不存在
- `constraint_veto`：被 Constraint Agent 否决
- `no_remaining_options`：已穷尽所有替代方案
- `max_replan_reached`：replan 次数超限
- `unknown`：未知原因

**`workspace_state`**：`clean | partial | dirty`

#### ConstraintSet

```python
@dataclass
class ConstraintSet:
    max_depth: int           # 该节点最大深度（从 root 算起）
    forbidden_actions: list[str]   # 禁止的操作，如 ["delete_file", "rm_rf"]
    failure_count_limit: int # 同 root cause 最多重试次数（默认 2）
```

---

### Agent 接口

```python
class ExecutionAgent(Protocol):
    async def execute(
        self,
        node: TaskNode,
        constraints: ConstraintSet,
        parent_context: NodeResult | None,
    ) -> NodeResult | FailureReport:
        """执行一个节点。成功→NodeResult，失败→FailureReport"""


class ConstraintAgent(Protocol):
    async def get_constraints(
        self,
        node: TaskNode,
        parent_result: NodeResult | None,
        history_failures: list[FailureReport],
    ) -> ConstraintSet:
        """根据节点信息+历史失败，生成约束包"""


class VerificationResult:
    passed: bool
    failed_nodes: list[str]
    reason: str
    evidence: list[str]


class VerificationAgent(Protocol):
    async def verify(
        self,
        root_goal: str,
        results: dict[str, NodeResult],
    ) -> VerificationResult:
        """验证所有执行结果是否满足 root_goal"""
```

---

### Scheduler 执行流程

```
Scheduler.run(root_goal: str) → VerificationResult

1. tree = TaskTree(); root = tree.add_child(goal=root_goal)

2. while not tree.is_done():
       node = tree.pick_deepest_pending()

       parent_result = parent.result if parent else None
       constraints = await constraint_agent.get_constraints(node, parent_result, failures)
       result = await execution_agent.execute(node, constraints, parent_result)

       if success:
           tree.mark_done(node.id, result)
           # 如果 node 还有未执行的 children，继续调度下一个
       else:
           if result.constraint_veto:
               tree.mark_blocked(node.id)
               # 向上报 parent，parent 决定是否 replan
           else:
               # 父节点 replan 新的 sibling child
               # node.status = failed

3. all_results = {n.id: n.result for n in tree.nodes.values() if n.result}
   verification = await verification_agent.verify(root_goal, all_results)

4. if not verification.passed:
       for failed_node_id in verification.failed_nodes:
           parent.replan(failed_node_id)  # 重新生成子节点
       goto 2

5. return verification
```

---

## 设计决策（已确认）

| # | 问题 | 决定 |
|---|------|------|
| MAX_CHILDREN | 父节点最多生成几个子节点 | **10** |
| Constraint Agent 模型 | 用什么模型 | **同模型**（与 Execution Agent 相同） |
| Verification 策略 | 初期验证方式 | **LLM_JUDGE**（后续可扩展到 test_run 等） |
| 节点 token 预算 | 每个节点最多消耗 token | **无限**（由 depth 约束） |

---

## Context 组装原则：最小依赖

子节点 context 只携带必要信息，遵循**默认不带、按需注入**原则。

### 子节点 context 构成

```
子节点 A1a 的 context =
│
├── 1. Root Goal（锚定，始终携带）
│
├── 2. 父节点 result（result.summary + artifacts）
│       只传父节点一级，不累积传递
│
├── 3. 父节点 goal
│
├── 4. 约束包（ConstraintSet）
│
└── 5. 必要的兄弟节点依赖（仅在显式依赖时注入）
```

### 注入规则

| 信息 | 默认 | 何时注入 |
|------|------|---------|
| 兄弟节点信息 | **不带** | 当前节点的 goal 明确依赖某个 sibling 的输出时 |
| 失败 context | **不带** | 父节点 replan 决策需要参考时（由父节点 context 携带） |
| 祖先进度 | **不带** | 只通过父节点 result.summary 间接体现 |
| Root Goal | **始终带** | 每个节点 context 第一位，作为目标锚点 |

---

## 与现有系统集成

| 现有组件 | 集成方式 |
|---------|---------|
| `ContextBuilder` | 每个节点 context 由 `build_node_context()` 构建 |
| `AgentLoop` | 被 TaskScheduler 包装，execute() 封装 process_direct() |
| `SessionManager` | 每个节点执行后写入 session |
| `MemoryRetriever` | 节点失败时检索相似 root cause 经验 |
| `skills.py` | Execution Agent 调用时注入 skill context |

---

## 接入方案 B：RouterBus 集成

### 架构

```
User Message
    │
    ▼
RouterBus  ────────────────────────────
    │                                  │
    ├── predicate: content.startswith("/plantask")
    │   or metadata._tasktree_task      │
    ▼                                  ▼
TaskTreeService                  AgentLoop
(独立服务，background)           (普通对话)
    │
    ├── Scheduler.run()
    ├── pick_deepest_pending()
    ├── depth-first 执行
    └── progress → bus (🚀/✅/❌/🚫)
            │
            ▼
        OutboundMessage (channel="cli")
```

### 集成要点

- **RouterBus**（`nanobot/bus/router.py`）：扩展 `MessageBus`，支持 predicate-based 路由
  - `register_route(consumer_id, predicate) → asyncio.Queue[InboundMessage]`
  - 每个 consumer 有独立队列，避免竞争
  - `start_router()` / `stop_router()` 管理路由循环

- **TaskTreeService**（`nanobot/agent/tasktree/service.py`）：独立服务
  - `submit(inbound)`：非阻塞，background task 执行
  - `cancel(chat_id)`：取消运行中任务
  - `get_status(chat_id)`：返回树状进度（实时调度状态 or 历史 session 消息）
  - `request_user_input(chat_id, question)`：暂停执行，等待用户输入
  - `submit_user_input(chat_id, response)`：用户响应后继续执行
  - 断后延续：checkpoint 保存到 session metadata，`submit()` 时检查并恢复

### 通知机制

| 事件 | Emoji | Metadata |
|------|-------|---------|
| 节点开始 | 🚀 | `_tasktree_progress: true` |
| 节点完成 | ✅ | `_tasktree_progress: true` |
| 节点失败 | ❌ | `_tasktree_progress: true` |
| 节点被阻止 | 🚫 | `_tasktree_progress: true` |
| 需要用户输入 | 🤔 | `_tasktree_needs_input: true` |
| 任务取消 | 🛑 | `_tasktree_cancelled: true` |

### 断后延续（Resume）

- `SessionPersistenceCallback` 在每个节点状态变化后调用 `save_checkpoint(tree)`
- Checkpoint 存储在 session metadata 的 `_tasktree_checkpoint` 字段
- `submit()` 时 `session_cb.load_checkpoint()` 有值则传入 `scheduler.run(resume_tree=saved_tree)`
- Scheduler 从 checkpoint 恢复树状态，继续执行 pending 节点

### 文件结构

```
nanobot/
    bus/
        __init__.py      # MessageBus, RouterBus, InboundMessage, OutboundMessage
        router.py        # RouterBus（predicates → 独立 consumer 队列）
    agent/tasktree/
        __init__.py          # 统一导出
        models.py            # TaskNode, TaskStatus, NodeResult, FailureReport,
                             # Artifact, RootCause, WorkspaceState, ConstraintSet
        tree.py              # TaskTree
        scheduler.py         # Scheduler 主循环 + Protocols
        context.py           # build_node_context()
        callbacks.py         # SessionPersistenceCallback
        memory_callback.py   # MemoryCallback
        service.py          # TaskTreeService
        execution/
            default.py      # DefaultExecutionAgent
            constraint.py   # DefaultConstraintAgent
            verification.py # LLMVerificationAgent
            subgoal.py      # LLMSubgoalParser
    nanobot.py               # Nanobot.from_config() 创建 RouterBus + TaskTreeService
```
