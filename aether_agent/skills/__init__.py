"""Closed learning loop and agentskills.io compliant skills hub."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether_agent.skills.engine import SkillEngine
    from aether_agent.skills.format import (
        SkillDefinition,
        parse_skill_markdown,
        serialize_skill_markdown,
    )
    from aether_agent.skills.generator import SkillGenerator
    from aether_agent.skills.tool import ReadSkillTool

__all__ = [
    "SkillDefinition",
    "SkillEngine",
    "SkillGenerator",
    "ReadSkillTool",
    "parse_skill_markdown",
    "serialize_skill_markdown",
]

_EXPORT_MAP = {
    "SkillDefinition": ("aether_agent.skills.format", "SkillDefinition"),
    "parse_skill_markdown": ("aether_agent.skills.format", "parse_skill_markdown"),
    "serialize_skill_markdown": ("aether_agent.skills.format", "serialize_skill_markdown"),
    "SkillEngine": ("aether_agent.skills.engine", "SkillEngine"),
    "SkillGenerator": ("aether_agent.skills.generator", "SkillGenerator"),
    "ReadSkillTool": ("aether_agent.skills.tool", "ReadSkillTool"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        mod_path, attr = _EXPORT_MAP[name]
        module = __import__(mod_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
