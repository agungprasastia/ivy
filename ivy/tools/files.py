"""File system manipulation tools (read, write, list, patch, search)."""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool
from ivy.tools.path_security import should_skip_sensitive_entry, validate_file_path

if TYPE_CHECKING:
    from ivy.tools.checkpoint import FileCheckpointStore
    from ivy.tools.registry import ToolRegistry


def _normalize_workspace_root(workspace_root: str | Path | None) -> Path | None:
    if workspace_root is None:
        return None
    return Path(workspace_root).expanduser().resolve()


def _build_diff_metadata(path_label: str, before: str, after: str) -> dict[str, Any]:
    """Compute bounded unified diff stats for file-editing tool results."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{path_label}",
            tofile=f"b/{path_label}",
            n=2,
        )
    )
    added = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))
    diff_text = "".join(diff_lines[:120])
    return {
        "path": path_label,
        "diff": diff_text,
        "added": added,
        "removed": removed,
    }


class ReadFileTool(BaseTool):
    """Tool to read content from a local file."""

    name = "read_file"
    concurrent_safe = True
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

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        read_tracker: set[Path] | None = None,
    ) -> None:
        self.workspace_root = _normalize_workspace_root(workspace_root)
        self.read_tracker = read_tracker

    async def execute(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        file_path, err = validate_file_path(
            path,
            mode="read",
            workspace_root=self.workspace_root,
        )
        if err is not None or file_path is None:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=err or f"Invalid path: {path}",
            )
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

            if self.read_tracker is not None:
                self.read_tracker.add(file_path)

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

    def __init__(
        self,
        checkpoint_store: FileCheckpointStore | None = None,
        session_id_getter: Callable[[], str] | None = None,
        workspace_root: str | Path | None = None,
        read_tracker: set[Path] | None = None,
        require_read_before_overwrite: bool = False,
    ) -> None:
        self.checkpoint_store = checkpoint_store
        self.session_id_getter = session_id_getter
        self.workspace_root = _normalize_workspace_root(workspace_root)
        self.read_tracker = read_tracker
        self.require_read_before_overwrite = require_read_before_overwrite

    async def execute(
        self,
        path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        file_path, err = validate_file_path(
            path,
            mode="write",
            workspace_root=self.workspace_root,
        )
        if err is not None or file_path is None:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=err or f"Invalid path: {path}",
            )
        try:
            existed_before = file_path.exists() and file_path.is_file()
            if (
                existed_before
                and self.require_read_before_overwrite
                and self.read_tracker is not None
                and file_path not in self.read_tracker
            ):
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"Must read existing file with read_file before overwriting: {path}",
                )
            before_text = ""
            if existed_before:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    before_text = f.read()
            after_text = (before_text + content) if append else content

            cp_id: int | None = None
            if self.checkpoint_store is not None:
                sid = self.session_id_getter() if self.session_id_getter else "cli:main"
                cp = self.checkpoint_store.record(
                    session_id=sid,
                    path=str(file_path),
                    display_path=path,
                    tool_name=self.name,
                    existed_before=existed_before,
                    old_content=before_text,
                    new_content=after_text,
                )
                cp_id = cp.id

            file_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)

            if self.read_tracker is not None:
                self.read_tracker.add(file_path)

            meta = _build_diff_metadata(path, before_text, after_text)
            if cp_id is not None:
                meta["checkpoint_id"] = cp_id
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Successfully wrote {len(content)} characters to {path}",
                metadata=meta,
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
    concurrent_safe = True
    description = "List files and subdirectories within a directory path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to directory to list.", "default": "."},
        },
        "required": [],
    }

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = _normalize_workspace_root(workspace_root)

    async def execute(self, path: str = ".", **kwargs: Any) -> ToolResult:
        dir_path, err = validate_file_path(
            path,
            mode="list",
            workspace_root=self.workspace_root,
        )
        if err is not None or dir_path is None:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=err or f"Invalid path: {path}",
            )
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

    def __init__(
        self,
        checkpoint_store: FileCheckpointStore | None = None,
        session_id_getter: Callable[[], str] | None = None,
        workspace_root: str | Path | None = None,
        read_tracker: set[Path] | None = None,
        require_read_before_overwrite: bool = False,
    ) -> None:
        self.checkpoint_store = checkpoint_store
        self.session_id_getter = session_id_getter
        self.workspace_root = _normalize_workspace_root(workspace_root)
        self.read_tracker = read_tracker
        self.require_read_before_overwrite = require_read_before_overwrite

    async def execute(
        self,
        path: str,
        target: str,
        replacement: str,
        **kwargs: Any,
    ) -> ToolResult:
        file_path, err = validate_file_path(
            path,
            mode="patch",
            workspace_root=self.workspace_root,
        )
        if err is not None or file_path is None:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=err or f"Invalid path: {path}",
            )
        if not file_path.exists():
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"File not found: {path}",
            )
        if (
            self.require_read_before_overwrite
            and self.read_tracker is not None
            and file_path not in self.read_tracker
        ):
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Must read file with read_file before patching: {path}",
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

            cp_id: int | None = None
            if self.checkpoint_store is not None:
                sid = self.session_id_getter() if self.session_id_getter else "cli:main"
                cp = self.checkpoint_store.record(
                    session_id=sid,
                    path=str(file_path),
                    display_path=path,
                    tool_name=self.name,
                    existed_before=True,
                    old_content=content,
                    new_content=updated,
                )
                cp_id = cp.id

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated)

            meta = _build_diff_metadata(path, content, updated)
            if cp_id is not None:
                meta["checkpoint_id"] = cp_id
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Successfully patched {path}",
                metadata=meta,
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
    concurrent_safe = True
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

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = _normalize_workspace_root(workspace_root)

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 50,
        **kwargs: Any,
    ) -> ToolResult:
        root, err = validate_file_path(
            path,
            mode="search",
            workspace_root=self.workspace_root,
        )
        if err is not None or root is None:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=err or f"Invalid path: {path}",
            )
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
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in skip_dirs and not should_skip_sensitive_entry(Path(dirpath) / d)
                ]
                for filename in sorted(filenames):
                    file_path = Path(dirpath) / filename
                    if should_skip_sensitive_entry(file_path):
                        continue
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

    def __init__(
        self,
        checkpoint_store: FileCheckpointStore | None = None,
        session_id_getter: Callable[[], str] | None = None,
        workspace_root: str | Path | None = None,
        read_tracker: set[Path] | None = None,
        require_read_before_overwrite: bool = False,
    ) -> None:
        self._read_tracker: set[Path] = read_tracker if read_tracker is not None else set()
        self.read_tool = ReadFileTool(
            workspace_root=workspace_root,
            read_tracker=self._read_tracker,
        )
        self.write_tool = WriteFileTool(
            checkpoint_store=checkpoint_store,
            session_id_getter=session_id_getter,
            workspace_root=workspace_root,
            read_tracker=self._read_tracker,
            require_read_before_overwrite=require_read_before_overwrite,
        )
        self.list_tool = ListDirectoryTool(workspace_root=workspace_root)
        self.patch_tool = PatchFileTool(
            checkpoint_store=checkpoint_store,
            session_id_getter=session_id_getter,
            workspace_root=workspace_root,
            read_tracker=self._read_tracker,
            require_read_before_overwrite=require_read_before_overwrite,
        )
        self.search_tool = SearchFilesTool(workspace_root=workspace_root)

    def register_all(self, registry: ToolRegistry) -> None:
        """Register read, write, list, patch, and search file tools into registry."""
        registry.register(self.read_tool)
        registry.register(self.write_tool)
        registry.register(self.list_tool)
        registry.register(self.patch_tool)
        registry.register(self.search_tool)
