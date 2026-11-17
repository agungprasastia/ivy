"""Tools ecosystem including terminal, file operations, code runner, web, and safety guard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ivy.tools.approval import SafetyApprovalPolicy
    from ivy.tools.base import BaseTool
    from ivy.tools.files import (
        FileTools,
        ListDirectoryTool,
        PatchFileTool,
        ReadFileTool,
        SearchFilesTool,
        WriteFileTool,
    )
    from ivy.tools.registry import ToolRegistry
    from ivy.tools.runner import CodeRunnerTool
    from ivy.tools.terminal import TerminalTool
    from ivy.tools.web import WebFetchTool, WebSearchTool

__all__ = [
    "BaseTool",
    "SafetyApprovalPolicy",
    "TerminalTool",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "PatchFileTool",
    "SearchFilesTool",
    "FileTools",
    "CodeRunnerTool",
    "WebFetchTool",
    "WebSearchTool",
    "ToolRegistry",
]

_EXPORT_MAP = {
    "BaseTool": ("ivy.tools.base", "BaseTool"),
    "SafetyApprovalPolicy": ("ivy.tools.approval", "SafetyApprovalPolicy"),
    "TerminalTool": ("ivy.tools.terminal", "TerminalTool"),
    "ReadFileTool": ("ivy.tools.files", "ReadFileTool"),
    "WriteFileTool": ("ivy.tools.files", "WriteFileTool"),
    "ListDirectoryTool": ("ivy.tools.files", "ListDirectoryTool"),
    "PatchFileTool": ("ivy.tools.files", "PatchFileTool"),
    "SearchFilesTool": ("ivy.tools.files", "SearchFilesTool"),
    "FileTools": ("ivy.tools.files", "FileTools"),
    "CodeRunnerTool": ("ivy.tools.runner", "CodeRunnerTool"),
    "WebFetchTool": ("ivy.tools.web", "WebFetchTool"),
    "WebSearchTool": ("ivy.tools.web", "WebSearchTool"),
    "ToolRegistry": ("ivy.tools.registry", "ToolRegistry"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        mod_path, attr = _EXPORT_MAP[name]
        module = __import__(mod_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
