"""Core data models for messages, tools, and session state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Role(str, Enum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Specification of a tool call initiated by the model."""

    model_config = ConfigDict(extra="ignore")

    id: str
    function_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except Exception:
                return {"raw_input": v}
        if isinstance(v, dict):
            return v
        return {}


class ToolResult(BaseModel):
    """Outcome of executing a tool call."""

    model_config = ConfigDict(extra="ignore")

    tool_call_id: str | None = None
    tool_name: str | None = None
    status: str = "success"  # "success", "rejected", "error"
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Standard message representation across LLMs, tools, and interfaces."""

    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str | None = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role", mode="before")
    @classmethod
    def parse_role(cls, v: Any) -> Role:
        if isinstance(v, Role):
            return v
        if isinstance(v, str):
            return Role(v.lower())
        raise ValueError(f"Invalid role: {v}")

    def to_llm_dict(self) -> dict[str, Any]:
        """Convert ChatMessage into OpenAI-compatible dict."""
        payload: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content or "",
        }

        if self.name:
            payload["name"] = self.name

        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]

        return payload


class SessionState(BaseModel):
    """Session continuity and conversation state."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
