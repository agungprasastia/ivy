"""Core data types, LLM adapter, ReAct orchestrator, and subagent runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether_agent.core.agent import AetherAgent
    from aether_agent.core.llm import LLMClient
    from aether_agent.core.subagent import SubagentRunner
    from aether_agent.core.types import (
        ChatMessage,
        Role,
        SessionState,
        ToolCall,
        ToolResult,
    )

__all__ = [
    "AetherAgent",
    "LLMClient",
    "SubagentRunner",
    "Role",
    "ChatMessage",
    "ToolCall",
    "ToolResult",
    "SessionState",
]

_EXPORT_MAP = {
    "AetherAgent": ("aether_agent.core.agent", "AetherAgent"),
    "LLMClient": ("aether_agent.core.llm", "LLMClient"),
    "SubagentRunner": ("aether_agent.core.subagent", "SubagentRunner"),
    "Role": ("aether_agent.core.types", "Role"),
    "ChatMessage": ("aether_agent.core.types", "ChatMessage"),
    "ToolCall": ("aether_agent.core.types", "ToolCall"),
    "ToolResult": ("aether_agent.core.types", "ToolResult"),
    "SessionState": ("aether_agent.core.types", "SessionState"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        mod_path, attr = _EXPORT_MAP[name]
        module = __import__(mod_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
