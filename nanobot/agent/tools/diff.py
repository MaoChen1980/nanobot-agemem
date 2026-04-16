"""DiffFile tool for showing differences between files."""

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema


def _resolve_path(path: str | None, workspace: Path, allowed_dir: Path | None) -> Path | None:
    """Resolve and validate a file path."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = workspace / p
    else:
        p = p.resolve()
    if allowed_dir:
        try:
            p.relative_to(allowed_dir)
        except ValueError:
            raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return p


@tool_parameters(
    tool_parameters_schema(
        path=StringSchema("The file to show diff for"),
        baseline=StringSchema(
            "The baseline content to compare against. "
            "Can be a file path (prefixed with 'file:') or raw content string. "
            "Examples: 'file:/path/to/baseline.txt', 'previous content here'",
        ),
        line_range=StringSchema(
            "Optional line range to scope the diff, e.g. '10-50' or '100-' for lines 100 to end. "
            "If omitted, shows full diff.",
            nullable=True,
        ),
        required=["path", "baseline"],
    )
)
class DiffFileTool(Tool):
    """Show a line-by-line diff between a file and a baseline."""

    def __init__(self, workspace: Path, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "diff_file"

    @property
    def description(self) -> str:
        return (
            "Show the diff between the current state of a file and a known baseline. "
            "Useful for seeing exactly what changed in a file since a reference point. "
            "Line numbers are shown for precise reference. "
            "Use line_range to scope the diff to a specific section."
        )

    async def execute(
        self,
        path: str | None = None,
        baseline: str | None = None,
        line_range: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Compute and return the diff."""
        if not path:
            return "Error: path is required"
        if not baseline:
            return "Error: baseline is required"

        try:
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
        except PermissionError as e:
            return f"Error: {e}"
        if not file_path or not file_path.exists():
            return f"Error: file not found: {path}"

        current_content = file_path.read_text(encoding="utf-8")

        # Resolve baseline: 'file:/path' or raw content
        if baseline.startswith("file:"):
            try:
                baseline_path = _resolve_path(baseline[5:], self._workspace, self._allowed_dir)
            except PermissionError as e:
                return f"Error: baseline {e}"
            if not baseline_path or not baseline_path.exists():
                return f"Error: baseline file not found: {baseline[5:]}"
            baseline_content = baseline_path.read_text(encoding="utf-8")
        else:
            baseline_content = baseline

        # Parse line_range
        start_line: int | None = None
        end_line: int | None = None
        if line_range:
            import re
            if "-" in line_range:
                parts = line_range.split("-", 1)
                if parts[0]:
                    start_line = int(parts[0])
                if parts[1]:
                    end_line = int(parts[1])
            else:
                start_line = int(line_range)
                end_line = None

        # Compute diff
        import difflib

        current_lines = current_content.splitlines(keepends=True)
        baseline_lines = baseline_content.splitlines(keepends=True)

        diff_lines: list[str] = []
        for line in difflib.unified_diff(
            baseline_lines,
            current_lines,
            fromfile=f"{path} (baseline)",
            tofile=f"{path} (current)",
            lineterm="",
        ):
            diff_lines.append(line)

        if not diff_lines:
            return "No differences found."

        # Apply line_range filter if specified
        if start_line is not None or end_line is not None:
            filtered: list[str] = []
            in_range = False
            for dl in diff_lines:
                if dl.startswith("@@"):
                    m = re.search(r"\+(\d+)", dl)
                    if m:
                        line_num = int(m.group(1))
                        start = start_line if start_line else 1
                        end = end_line if end_line else float("inf")
                        in_range = start <= line_num <= end
                    else:
                        in_range = False
                    if in_range:
                        filtered.append(dl)
                elif in_range:
                    filtered.append(dl)
            diff_lines = filtered

        if not diff_lines:
            return "No differences in the specified line range."

        return "".join(diff_lines)
