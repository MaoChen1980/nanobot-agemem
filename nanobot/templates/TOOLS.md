# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Shell Commands

- Execute a shell command and return its output
- Prefer read_file/write_file/edit_file over cat/echo/sed
- Prefer grep/glob over shell find/grep
- Use -y or --yes flags to avoid interactive prompts
- Output is truncated at 10,000 chars; timeout defaults to 60s (max 600s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)

## glob — File Discovery

- Find files matching a glob pattern (e.g. '*.py', 'tests/**/test_*.py')
- Results are sorted by modification time (newest first)
- Skips .git, node_modules, __pycache__, and other noise directories
- Use `entry_type="dirs"` when you need matching directories instead of files
- Use `head_limit` and `offset` to page through large result sets
- Prefer this over exec when you only need file paths

## grep — Content Search

- Search file contents with a regex pattern
- Default `output_mode` is `files_with_matches` (file paths only); use `content` for matching lines with context
- Skips binary and files >2 MB
- Supports `glob`/`type` filtering plus `context_before`/`context_after`
- Use `fixed_strings=true` for literal keywords containing regex characters
- Use `output_mode="count"` to size a search before reading full matches
- Use `head_limit` and `offset` to page across results
- Prefer this over exec for code and history searches

## read_file — Read Files

- Read a file (text or image). Text output format: `LINE_NUM|CONTENT`
- Images return visual content for analysis
- Use `offset` (1-indexed line) and `limit` for large files
- Cannot read non-image binary files
- Reads exceeding ~128K chars are truncated
- Use `pages` for PDF page ranges (e.g. '1-5')

## write_file — Write Files

- Write content to a file. Overwrites if the file already exists
- Creates parent directories as needed
- For partial edits, prefer edit_file instead

## edit_file — Edit Files

- Edit a file by replacing `old_text` with `new_text`
- Tolerates minor whitespace/indentation differences and curly/straight quote mismatches
- If `old_text` matches multiple times, you must provide more context or set `replace_all=true`
- Shows a diff of the closest match on failure

## list_dir — List Directories

- List the contents of a directory
- Set `recursive=true` to explore nested structure
- Common noise directories (.git, node_modules, __pycache__, etc.) are auto-ignored

## diff_file — Show Diffs

- Show the diff between the current state of a file and a known baseline
- Useful for seeing exactly what changed since a reference point
- Line numbers are shown for precise reference
- Use `line_range` to scope the diff to a specific section (e.g. '10-50' or '100-')

## web_search — Web Search

- Search the web. Returns titles, URLs, and snippets
- `count` defaults to 5 (max 10)
- Use web_fetch to read a specific page in full

## web_fetch — Fetch Web Pages

- Fetch a URL and extract readable content (HTML to markdown/text)
- Output is capped at `maxChars` (default 50,000)
- Works for most web pages and docs; may fail on login-walled or JS-heavy sites

## spawn — Subagent Background Tasks

- Spawn a subagent to handle a task in the background
- Use this for complex or time-consuming tasks that can run independently
- The subagent will complete the task and report back when done
- For deliverables or existing projects, inspect the workspace first and use a dedicated subdirectory
- `spawn` returns a structured result with `task_id` (save this for later)
- Always check if a subagent is already running before spawning a duplicate task

## update_context — Push Updates to Subagent

- Push updated context to a running subagent
- Use this when the user provides new information that should reach the subagent
- The subagent receives it as a new message and can adjust its plan accordingly
- Returns whether the update was successfully delivered

## list_subagents — List Running Subagents

- List all currently running subagents
- Use this before `update_context` to find the right `task_id`
- Each subagent shows `task_id`, `label`, and `session_key`

## cron — Scheduled Reminders

- Schedule reminders and recurring tasks
- Actions: `add`, `list`, `remove`
- If `tz` is omitted, cron expressions and naive ISO times default to UTC
- Parameters: `action` (add/list/remove, required), `name`, `message`, `every_seconds`, `cron_expr`, `tz`, `at`, `deliver`, `job_id`

## add_memory — Store Long-Term Memory

- Store a new fact, preference, or piece of knowledge in long-term memory
- Use when the user shares information that should be remembered across sessions
- `importance`: 0.0 (trivial) to 1.0 (critical), default 0.5
- `tags`: optional tags for categorizing

## update_memory — Update Memory Entry

- Update the content, importance, or tags of an existing memory entry
- Use when information changes or needs correction

## delete_memory — Delete Memory Entry

- Remove an obsolete or incorrect memory entry
- The entry is soft-deleted and no longer retrieved

## retrieve_memories — Search Memory

- Search long-term memory for entries relevant to the current task or question
- Returns top-k most relevant memories with their relevance scores

## filter_memories — Filter Memory by Tags

- Filter stored memories by tags and/or minimum importance threshold
- Use to find memories about specific topics or of minimum significance

## summarize_session — Summarize Conversation

- Summarize key information from the current conversation and store it in long-term memory
- Use when important facts emerge that should be remembered

## notebook_edit — Edit Jupyter Notebooks

- Edit a Jupyter notebook (.ipynb) cell
- Modes: `replace` (default), `insert`, `delete`
- `cell_index` is 0-based
- `cell_type`: `code` or `markdown`

## message — Send Message to User

- Send a message to the user, optionally with file attachments
- This is the ONLY way to deliver files (images, documents, audio, video) to the user
- Use the `media` parameter with file paths to attach files
- Do NOT use read_file to send files — that only reads content for your own analysis

## taskplan — Submit a TaskTree Goal

- Submit a complex multi-step task to TaskTree for background planning and execution
- Use for tasks that benefit from automatic decomposition
- The task runs hierarchically and reports progress
- Only one TaskTree task can run per session at a time
- Check status with `taskstatus()`, cancel with `taskcancel()`
- TaskTree sessions are shared with the user's current session — `/taskstatus` and `/taskcancel` commands operate on the same task

## taskstatus — Check TaskTree Status

- Get the current status of the running TaskTree task
- Returns progress, completed steps, and any pending user questions
- Works on the same session as the `taskplan` that started the task

## taskcancel — Cancel TaskTree Task

- Cancel the currently running TaskTree task
- Use when the task is taking too long, the goal has changed, or you need to start a different task
- Returns "No running TaskTree task found." if there is nothing to cancel
