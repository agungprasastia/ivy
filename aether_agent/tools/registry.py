"""Central registry for managing and invoking tools."""

from __future__ import annotations

from typing import Any

from aether_agent.core.types import ToolResult
from aether_agent.tools.base import BaseTool


class ToolRegistry:
    """Registry maintaining available tools and dispatching execution."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a new tool instance."""
        if not tool.name:
            raise ValueError("Tool name must not be empty")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    def subset(self, allowed_names: list[str] | set[str]) -> ToolRegistry:
        """Create a restricted registry containing only the specified tool names."""
        sub = ToolRegistry()
        for name in allowed_names:
            tool = self.get(name)
            if tool is not None:
                sub.register(tool)
        return sub

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Dispatch execution to the appropriate tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status="error",
                error=f"Tool '{tool_name}' is not registered.",
            )

        try:
            result = await tool.execute(**arguments)
            if result.tool_call_id is None and tool_call_id is not None:
                result.tool_call_id = tool_call_id
            if result.tool_name is None:
                result.tool_name = tool_name
            return result
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status="error",
                error=f"Unexpected error executing {tool_name}: {exc}",
            )
