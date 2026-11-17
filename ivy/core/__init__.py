"""Core data types, LLM adapter, ReAct orchestrator, and subagent runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ivy.core.agent import IvyAgent
    from ivy.core.llm import LLMClient
    from ivy.core.subagent import SubagentRunner
    from ivy.core.types import (
        ChatMessage,
        Role,
        SessionState,
        ToolCall,
        ToolResult,
    )

__all__ = [
    "IvyAgent",
    "LLMClient",
    "SubagentRunner",
    "Role",
    "ChatMessage",
    "ToolCall",
    "ToolResult",
    "SessionState",
]

_EXPORT_MAP = {
    "IvyAgent": ("ivy.core.agent", "IvyAgent"),
    "LLMClient": ("ivy.core.llm", "LLMClient"),
    "SubagentRunner": ("ivy.core.subagent", "SubagentRunner"),
    "Role": ("ivy.core.types", "Role"),
    "ChatMessage": ("ivy.core.types", "ChatMessage"),
    "ToolCall": ("ivy.core.types", "ToolCall"),
    "ToolResult": ("ivy.core.types", "ToolResult"),
    "SessionState": ("ivy.core.types", "SessionState"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        mod_path, attr = _EXPORT_MAP[name]
        module = __import__(mod_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
