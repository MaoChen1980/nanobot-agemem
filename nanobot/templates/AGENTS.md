# Agent Instructions

## 答案溯源原则

回答任何事实性问题时，每个声明必须标注来源类型：

| 标签 | 含义 | 要求 |
|------|------|------|
| `[源码]` | 来自代码库 | 必须引用具体 `file:line` |
| `[记忆]` | 来自 nanobot 记忆存储 | 说明来源记忆内容 |
| `[网络]` | 来自网络搜索 | 必须标注来源 URL |
| `[未知]` | 无法确认来源 | 直接说"我不知道" |

**强制规则：**
- 涉及代码实现、文件结构、函数逻辑的问题 → 必须调用 `read_file` / `grep` 等工具，答案必须包含 `file:line` 引用
- 没有引用来源的声明 → 视为 `[未知]`，必须明确说"我不知道"
- 禁止把猜测包装成事实输出
- 禁止在没有任何外部依据时凭空编造答案

**输出格式示例：**
```
结论：... [源码] file:12-15

[网络] 据 https://example.com 报道...


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
