"""Self-directed skill management tool allowing the agent to CRUD SKILL.md entries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ivy.core.types import ToolResult
from ivy.skills.format import SkillDefinition
from ivy.skills.guard import format_scan_rejection, scan_skill_content
from ivy.tools.base import BaseTool

if TYPE_CHECKING:
    from ivy.skills.engine import SkillEngine


class SkillManagerTool(BaseTool):
    """Create, update, inspect, list, or delete reusable skills via SkillEngine."""

    name = "skill_manage"
    description = (
        "Create, update, inspect, list, or delete reusable procedural skills (SKILL.md) "
        "in the SkillEngine."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "inspect", "create", "update", "delete"],
                "description": "Operation to perform.",
            },
            "name": {"type": "string", "description": "Skill identifier."},
            "description": {"type": "string", "description": "Short summary for skill index."},
            "instructions": {"type": "string", "description": "Full markdown instructions."},
        },
        "required": ["action"],
    }

    def __init__(self, skill_engine: SkillEngine) -> None:
        self.skill_engine = skill_engine

    async def execute(
        self,
        action: str = "list",
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        act = (action or "list").strip().lower()

        if act == "list":
            items = self.skill_engine.list_skills()
            if not items:
                return ToolResult(tool_name=self.name, status="success", output="No skills loaded.")
            text = "\n".join(f"- {s['name']} ({s.get('status', 'active')}): {s.get('description', '')}" for s in items)
            return ToolResult(tool_name=self.name, status="success", output=text)

        if not name or not name.strip():
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"'name' is required for action '{act}'.",
            )
        clean_name = name.strip()

        if act == "inspect":
            sk = self.skill_engine.inspect_skill(clean_name)
            if sk is None:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"Skill '{clean_name}' not found.",
                )
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"# {sk.name}\n{sk.description}\n\n{sk.instructions}",
            )

        if act in ("create", "update"):
            existing = self.skill_engine.inspect_skill(clean_name)
            if act == "update" and existing is None:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"Cannot update missing skill '{clean_name}'. Use 'create' instead.",
                )
            desc = " ".join((description or (existing.description if existing else "")).splitlines()).strip()
            instr = (instructions or (existing.instructions if existing else "")).strip()
            target_status = existing.status if existing else "active"
            if not desc or not instr:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error="Both 'description' and 'instructions' are required.",
                )
            scan = scan_skill_content(clean_name, desc, instr, status=target_status)
            if not scan.safe:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=format_scan_rejection(scan),
                )
            try:
                path = self.skill_engine.save_skill(
                    SkillDefinition(
                        name=clean_name,
                        description=desc,
                        instructions=instr,
                        status=target_status,
                    )
                )
            except Exception as exc:
                return ToolResult(tool_name=self.name, status="error", error=str(exc))
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Saved skill '{clean_name}' at {path}",
            )

        if act == "delete":
            removed = self.skill_engine.remove_skill(clean_name)
            if not removed:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"Skill '{clean_name}' not found.",
                )
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Deleted skill '{clean_name}'.",
            )

        return ToolResult(
            tool_name=self.name,
            status="error",
            error=f"Unknown action '{action}'.",
        )
