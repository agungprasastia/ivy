"""Parser and serializer for agentskills.io SKILL.md format."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class SkillDefinition(BaseModel):
    """Structured representation of an agentskills.io skill."""

    name: str
    description: str
    status: str = "active"  # "active" or "draft"
    instructions: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None


def _unquote(val: str) -> str:
    val = val.strip()
    if len(val) >= 2 and (
        (val.startswith('"') and val.endswith('"'))
        or (val.startswith("'") and val.endswith("'"))
    ):
        return val[1:-1]
    return val


def parse_skill_markdown(text: str) -> SkillDefinition:
    """Parse SKILL.md content with YAML frontmatter into a SkillDefinition."""
    normalized = text.replace("\r\n", "\n").lstrip("\ufeff")
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", normalized, re.DOTALL)
    if not match:
        raise ValueError("Missing or malformed YAML frontmatter delimited by '---'")

    frontmatter_raw, body = match.group(1), match.group(2).strip()
    fields: dict[str, Any] = {}

    for line in frontmatter_raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML frontmatter line: {line!r}")
        key, raw_val = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty frontmatter key in line: {line!r}")
        fields[key] = _unquote(raw_val)

    name = str(fields.pop("name", "")).strip()
    description = str(fields.pop("description", "")).strip()
    if not name or not description:
        raise ValueError("Skill frontmatter must define non-empty 'name' and 'description'")

    status = str(fields.pop("status", "active")).strip() or "active"
    return SkillDefinition(
        name=name,
        description=description,
        status=status,
        instructions=body,
        metadata=fields,
    )


def _collapse_scalar(value: Any) -> str:
    return " ".join(str(value).splitlines()).strip()


def serialize_skill_markdown(skill: SkillDefinition) -> str:
    """Serialize a SkillDefinition into agentskills.io SKILL.md text."""
    lines = [
        "---",
        f"name: {_collapse_scalar(skill.name)}",
        f"description: {_collapse_scalar(skill.description)}",
        f"status: {_collapse_scalar(skill.status)}",
    ]
    for key, val in skill.metadata.items():
        lines.append(f"{_collapse_scalar(key)}: {_collapse_scalar(val)}")
    lines.append("---")
    if skill.instructions:
        lines.append(skill.instructions.strip())
    return "\n".join(lines) + "\n"
