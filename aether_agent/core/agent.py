"""ReAct autonomous agent loop and orchestrator."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable

from aether_agent.config import AetherConfig
from aether_agent.core.llm import LLMClient
from aether_agent.core.prompt import build_system_prompt, format_tool_result
from aether_agent.core.types import ChatMessage, Role
from aether_agent.memory.context import ContextWindowManager
from aether_agent.memory.store import MemoryStore
from aether_agent.skills.engine import SkillEngine
from aether_agent.tools.registry import ToolRegistry


class AetherAgent:
    """Autonomous ReAct agent coordinating LLM reasoning, tools, memory, and skills."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        memory_store: MemoryStore | None = None,
        skill_engine: SkillEngine | None = None,
        config: AetherConfig | None = None,
        max_iterations: int | None = None,
        exclude_from_fts: bool = False,
        extra_system_instructions: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.skill_engine = skill_engine
        self.config = config
        self.max_iterations = max_iterations or (
            config.max_iterations if config else 15
        )
        self.exclude_from_fts = exclude_from_fts
        self.extra_system_instructions = extra_system_instructions
        self._in_memory_sessions: dict[str, list[ChatMessage]] = defaultdict(list)
        self._fallback_context_mgr = ContextWindowManager()

    async def _load_history(self, session_id: str) -> list[ChatMessage]:
        if self.memory_store is not None:
            return await self.memory_store.get_messages(session_id)
        return list(self._in_memory_sessions[session_id])

    async def _persist_message(
        self, session_id: str, message: ChatMessage, platform: str = "cli"
    ) -> None:
        if self.memory_store is not None:
            await self.memory_store.save_message(
                session_id,
                message,
                platform=platform,
                exclude_from_fts=self.exclude_from_fts,
            )
        else:
            self._in_memory_sessions[session_id].append(message)

    async def _emit(
        self, callback: Callable[[Any], Any] | None, payload: Any
    ) -> None:
        if callback is None:
            return
        res = callback(payload)
        if inspect.isawaitable(res):
            await res

    async def run_turn(
        self,
        prompt: str,
        session_id: str = "default",
        platform: str = "cli",
        stream_callback: Callable[[str], Any | Awaitable[Any]] | None = None,
        tool_callback: Callable[[dict[str, Any]], Any | Awaitable[Any]] | None = None,
    ) -> ChatMessage:
        """Execute one full user turn through the ReAct loop."""
        user_profile: dict[str, str] | None = None
        if self.memory_store is not None:
            user_profile = await self.memory_store.get_user_profile()

        system_prompt = build_system_prompt(
            skill_engine=self.skill_engine,
            user_profile=user_profile,
            extra_instructions=self.extra_system_instructions,
        )

        history = await self._load_history(session_id)
        user_msg = ChatMessage(role=Role.USER, content=prompt)
        await self._persist_message(session_id, user_msg, platform=platform)

        messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=system_prompt),
            *history,
            user_msg,
        ]
        if self.memory_store is not None:
            messages = self.memory_store.compress_context(messages)
        else:
            messages = self._fallback_context_mgr.compress(messages)

        schemas = self.tool_registry.get_schemas() or None

        for _ in range(self.max_iterations):
            reply = await self.llm_client.chat_completion(
                messages=messages,
                tools=schemas,
                stream=False,
            )
            if not isinstance(reply, ChatMessage):
                text = "".join([chunk async for chunk in reply])
                reply = ChatMessage(role=Role.ASSISTANT, content=text)

            await self._persist_message(session_id, reply, platform=platform)
            messages.append(reply)

            if not reply.tool_calls:
                if reply.content:
                    await self._emit(stream_callback, reply.content)
                return reply

            for tc in reply.tool_calls:
                await self._emit(
                    tool_callback,
                    {"phase": "start", "name": tc.function_name, "arguments": tc.arguments},
                )
                result = await self.tool_registry.execute(
                    tool_name=tc.function_name,
                    arguments=tc.arguments,
                    tool_call_id=tc.id,
                )
                await self._emit(
                    tool_callback,
                    {
                        "phase": "end",
                        "name": tc.function_name,
                        "status": result.status,
                        "output": result.output,
                        "error": result.error,
                    },
                )

                tool_msg = ChatMessage(
                    role=Role.TOOL,
                    name=tc.function_name,
                    tool_call_id=tc.id,
                    content=format_tool_result(result),
                )
                await self._persist_message(session_id, tool_msg, platform=platform)
                messages.append(tool_msg)

        exhausted = ChatMessage(
            role=Role.ASSISTANT,
            content="Stopped after reaching maximum reasoning iterations without a final response.",
        )
        await self._persist_message(session_id, exhausted, platform=platform)
        await self._emit(stream_callback, exhausted.content)
        return exhausted
