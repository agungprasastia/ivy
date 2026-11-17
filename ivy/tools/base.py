"""Base abstractions for tools and parameter schemas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ivy.core.types import ToolResult


class BaseTool(ABC):
    """Abstract base class for all callable tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with keyword arguments."""

    def get_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible tool JSON schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
