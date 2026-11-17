"""Tools ecosystem including terminal, file operations, code runner, web, and safety guard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether_agent.tools.approval import SafetyApprovalPolicy
    from aether_agent.tools.base import BaseTool
    from aether_agent.tools.files import (
        FileTools,
        ListDirectoryTool,
        PatchFileTool,
        ReadFileTool,
        SearchFilesTool,
        WriteFileTool,
    )
    from aether_agent.tools.registry import ToolRegistry
    from aether_agent.tools.runner import CodeRunnerTool
    from aether_agent.tools.terminal import TerminalTool
    from aether_agent.tools.web import WebFetchTool, WebSearchTool

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
    "BaseTool": ("aether_agent.tools.base", "BaseTool"),
    "SafetyApprovalPolicy": ("aether_agent.tools.approval", "SafetyApprovalPolicy"),
    "TerminalTool": ("aether_agent.tools.terminal", "TerminalTool"),
    "ReadFileTool": ("aether_agent.tools.files", "ReadFileTool"),
    "WriteFileTool": ("aether_agent.tools.files", "WriteFileTool"),
    "ListDirectoryTool": ("aether_agent.tools.files", "ListDirectoryTool"),
    "PatchFileTool": ("aether_agent.tools.files", "PatchFileTool"),
    "SearchFilesTool": ("aether_agent.tools.files", "SearchFilesTool"),
    "FileTools": ("aether_agent.tools.files", "FileTools"),
    "CodeRunnerTool": ("aether_agent.tools.runner", "CodeRunnerTool"),
    "WebFetchTool": ("aether_agent.tools.web", "WebFetchTool"),
    "WebSearchTool": ("aether_agent.tools.web", "WebSearchTool"),
    "ToolRegistry": ("aether_agent.tools.registry", "ToolRegistry"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        mod_path, attr = _EXPORT_MAP[name]
        module = __import__(mod_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
