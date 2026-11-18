"""Long-term user preference & profile memory tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool

if TYPE_CHECKING:
    from ivy.memory.store import MemoryStore


class RememberPreferenceTool(BaseTool):
    """Persist durable user preferences and project facts across sessions."""

    name = "remember_preference"
    description = (
        "Save or update a persistent cross-session user preference or fact "
        "(e.g. coding style, language, environment details) in long-term profile memory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Short snake_case preference key (e.g. 'preferred_language').",
            },
            "value": {
                "type": "string",
                "description": "Preference value or concise guideline to remember.",
            },
        },
        "required": ["key", "value"],
    }

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    async def execute(self, key: str = "", value: str = "", **kwargs: Any) -> ToolResult:
        clean_key = key.strip()
        clean_val = value.strip()
        if not clean_key or not clean_val:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error="Both 'key' and 'value' must be non-empty.",
            )
        await self.memory_store.set_user_preference(clean_key, clean_val)
        return ToolResult(
            tool_name=self.name,
            status="success",
            output=f"Saved preference {clean_key}={clean_val}",
            metadata={"key": clean_key, "value": clean_val},
        )
