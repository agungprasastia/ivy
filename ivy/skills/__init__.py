"""Closed learning loop and agentskills.io compliant skills hub."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ivy.skills.engine import SkillEngine
    from ivy.skills.format import (
        SkillDefinition,
        parse_skill_markdown,
        serialize_skill_markdown,
    )
    from ivy.skills.generator import SkillGenerator
    from ivy.skills.tool import ReadSkillTool

__all__ = [
    "SkillDefinition",
    "SkillEngine",
    "SkillGenerator",
    "ReadSkillTool",
    "parse_skill_markdown",
    "serialize_skill_markdown",
]

_EXPORT_MAP = {
    "SkillDefinition": ("ivy.skills.format", "SkillDefinition"),
    "parse_skill_markdown": ("ivy.skills.format", "parse_skill_markdown"),
    "serialize_skill_markdown": ("ivy.skills.format", "serialize_skill_markdown"),
    "SkillEngine": ("ivy.skills.engine", "SkillEngine"),
    "SkillGenerator": ("ivy.skills.generator", "SkillGenerator"),
    "ReadSkillTool": ("ivy.skills.tool", "ReadSkillTool"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        mod_path, attr = _EXPORT_MAP[name]
        module = __import__(mod_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
