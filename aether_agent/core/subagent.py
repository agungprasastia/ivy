"""Isolated subagent delegation runner."""

from __future__ import annotations

from uuid import uuid4

from aether_agent.core.agent import AetherAgent
from aether_agent.core.llm import LLMClient
from aether_agent.memory.store import MemoryStore
from aether_agent.tools.registry import ToolRegistry


class SubagentRunner:
    """Runs isolated child agents with ephemeral sessions and restricted toolsets."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        memory_store: MemoryStore | None = None,
        max_iterations: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.max_iterations = max_iterations

    async def run(
        self,
        task_prompt: str,
        parent_session_id: str = "main",
        allowed_tools: list[str] | None = None,
    ) -> str:
        """Execute a delegated task in an isolated ephemeral child session."""
        child_session_id = f"{parent_session_id}::sub::{uuid4().hex[:8]}"
        scoped_registry = (
            self.tool_registry.subset(allowed_tools)
            if allowed_tools is not None
            else self.tool_registry
        )
        child_agent = AetherAgent(
            llm_client=self.llm_client,
            tool_registry=scoped_registry,
            memory_store=self.memory_store,
            max_iterations=self.max_iterations,
            exclude_from_fts=True,
            extra_system_instructions=(
                "You are an isolated subagent executing a specific delegated subtask. "
                "Return a concise, factual summary of your findings."
            ),
        )
        result = await child_agent.run_turn(
            prompt=task_prompt,
            session_id=child_session_id,
            platform="subagent",
        )
        return result.content or ""
