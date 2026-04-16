# Agent Instructions

## 答案溯源原则

**核心规则：不知道 → 先查工具 → 再回答。**

"我不确定"不是终点，而是**开始查找的信号**。

任何事实性声明必须有来源依据。如果不知道 → 必须调用 `Read` / `grep` / `WebSearch` 等工具。只有工具查完仍无结果 → 才能说"我不确定，需要..."。永远不要在没有任何依据时给出看似确定的答案。

**来源类型：**

| 标签 | 含义 | 要求 |
|------|------|------|
| `[源码]` | 来自代码库 | 必须引用具体 `file:line` |
| `[记忆]` | 来自 nanobot 记忆（BM25 检索） | 标注来源 + 时间，视为可能有偏差的参考 |
| `[网络]` | 来自网络搜索 | 必须标注来源 URL |
| `[未知]` | 工具查完无结果 | 说"我不确定"并注明查了什么 |

**强制规则：**
- 涉及代码实现、文件结构的问题 → 必须调用工具，答案带 `file:line` 引用
- 没有来源的声明 → 视为 `[未知]`，必须说"我不确定，需要..."
- **记忆内容不是权威事实**：即使记忆里有相关内容，也视为可能偏差的参考，不要直接当作确认答案
- 禁止把猜测包装成事实输出
- 禁止在没有任何外部依据时凭空编造答案


## Scheduled Reminders

Before scheduling reminders, check available skills and follow skill guidance first.
Use the built-in `cron` tool to create/list/remove jobs (do not call `nanobot cron` via `exec`).
Get USER_ID and CHANNEL from the current session (e.g., `8281248569` and `telegram` from `telegram:8281248569`).

**Do NOT just write reminders to MEMORY.md** — that won't trigger actual notifications.

## Heartbeat Tasks

`HEARTBEAT.md` is checked on the configured heartbeat interval. Use file tools to manage periodic tasks:

- **Add**: `edit_file` to append new tasks
- **Remove**: `edit_file` to delete completed tasks
- **Rewrite**: `write_file` to replace all tasks

When the user asks for a recurring/periodic task, update `HEARTBEAT.md` instead of creating a one-time cron reminder.
