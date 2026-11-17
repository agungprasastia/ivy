"""File system manipulation tools (read, write, list, patch, search)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool

if TYPE_CHECKING:
    from ivy.tools.registry import ToolRegistry


class ReadFileTool(BaseTool):
    """Tool to read content from a local file."""

    name = "read_file"
    description = "Read text content from a specified file path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the file."},
            "start_line": {"type": "integer", "description": "Optional starting line number (1-based)."},
            "end_line": {"type": "integer", "description": "Optional ending line number (inclusive)."},
        },
        "required": ["path"],
    }

    async def execute(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        file_path = Path(path).resolve()
        if not file_path.exists():
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"File not found: {path}",
            )
        if not file_path.is_file():
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Path is not a regular file: {path}",
            )

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            if start_line is not None or end_line is not None:
                start_idx = max(0, (start_line - 1) if start_line else 0)
                end_idx = end_line if end_line is not None else len(lines)
                content = "".join(lines[start_idx:end_idx])
            else:
                content = "".join(lines)

            return ToolResult(
                tool_name=self.name,
                status="success",
                output=content,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Failed to read file: {exc}",
            )


class WriteFileTool(BaseTool):
    """Tool to write or append text content to a local file."""

    name = "write_file"
    description = "Write text content to a file, creating parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the destination file."},
            "content": {"type": "string", "description": "Text content to write."},
            "append": {"type": "boolean", "description": "If true, appends content instead of overwriting."},
        },
        "required": ["path", "content"],
    }

    async def execute(
        self,
        path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        file_path = Path(path).resolve()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Successfully wrote {len(content)} characters to {path}",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Failed to write file: {exc}",
            )


class ListDirectoryTool(BaseTool):
    """Tool to list entries in a directory."""

    name = "list_directory"
    description = "List files and subdirectories within a directory path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to directory to list.", "default": "."},
        },
        "required": [],
    }

    async def execute(self, path: str = ".", **kwargs: Any) -> ToolResult:
        dir_path = Path(path).resolve()
        if not dir_path.exists():
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Directory not found: {path}",
            )
        if not dir_path.is_dir():
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Path is not a directory: {path}",
            )

        try:
            items = []
            for entry in os.scandir(dir_path):
                kind = "DIR" if entry.is_dir() else "FILE"
                size = entry.stat().st_size if entry.is_file() else 0
                items.append(f"{kind:<5} {size:>10} bytes  {entry.name}")

            output = "\n".join(sorted(items))
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=output if output else "(empty directory)",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Failed to list directory: {exc}",
            )


class PatchFileTool(BaseTool):
    """Tool to replace exact text segments in a file."""

    name = "patch_file"
    description = "Replace an exact target text sequence with new replacement text in a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to modify."},
            "target": {"type": "string", "description": "Exact text substring to replace."},
            "replacement": {"type": "string", "description": "New replacement text substring."},
        },
        "required": ["path", "target", "replacement"],
    }

    async def execute(
        self,
        path: str,
        target: str,
        replacement: str,
        **kwargs: Any,
    ) -> ToolResult:
        file_path = Path(path).resolve()
        if not file_path.exists():
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"File not found: {path}",
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if target not in content:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"Target substring not found in file: {path}",
                )

            updated = content.replace(target, replacement)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated)

            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Successfully patched {path}",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Failed to patch file: {exc}",
            )


class SearchFilesTool(BaseTool):
    """Tool to search text patterns recursively across files in a directory."""

    name = "search_files"
    description = "Search for a regex pattern across files in a directory."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression pattern to search for."},
            "path": {"type": "string", "description": "Root directory to search.", "default": "."},
            "max_results": {"type": "integer", "description": "Maximum matching lines to return.", "default": 50},
        },
        "required": ["pattern"],
    }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 50,
        **kwargs: Any,
    ) -> ToolResult:
        root = Path(path).resolve()
        if not root.exists():
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Path not found: {path}",
            )

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Invalid regex pattern: {exc}",
            )

        matches: list[str] = []
        skip_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}

        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                for filename in sorted(filenames):
                    file_path = Path(dirpath) / filename
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            for lineno, line in enumerate(f, start=1):
                                if regex.search(line):
                                    rel = file_path.relative_to(root)
                                    matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                                    if len(matches) >= max_results:
                                        return ToolResult(
                                            tool_name=self.name,
                                            status="success",
                                            output="\n".join(matches),
                                        )
                    except OSError:
                        continue

            return ToolResult(
                tool_name=self.name,
                status="success",
                output="\n".join(matches) if matches else "No matches found.",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Search failed: {exc}",
            )


class FileTools:
    """Helper facade for all file tools."""

    def __init__(self) -> None:
        self.read_tool = ReadFileTool()
        self.write_tool = WriteFileTool()
        self.list_tool = ListDirectoryTool()
        self.patch_tool = PatchFileTool()
        self.search_tool = SearchFilesTool()

    def register_all(self, registry: ToolRegistry) -> None:
        """Register read, write, list, patch, and search file tools into registry."""
        registry.register(self.read_tool)
        registry.register(self.write_tool)
        registry.register(self.list_tool)
        registry.register(self.patch_tool)
        registry.register(self.search_tool)
