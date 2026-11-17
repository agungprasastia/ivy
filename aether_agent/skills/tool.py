"""Tool allowing the agent to read full instructions of discovered skills."""

from __future__ import annotations

from typing import Any

from aether_agent.core.types import ToolResult
from aether_agent.skills.engine import SkillEngine
from aether_agent.tools.base import BaseTool


class ReadSkillTool(BaseTool):
    """Retrieve the full markdown instructions for a registered skill."""

    name = "read_skill"
    description = "Read the detailed instructions of an available skill by name."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The exact name of the skill to load."},
        },
        "required": ["name"],
    }

    def __init__(self, skill_engine: SkillEngine) -> None:
        self.skill_engine = skill_engine

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        instructions = self.skill_engine.read_skill(name)
        if instructions is None:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Skill '{name}' not found.",
            )
        return ToolResult(
            tool_name=self.name,
            status="success",
            output=instructions,
        )
