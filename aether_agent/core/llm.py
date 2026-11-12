"""Universal LLM client using LiteLLM for multi-provider support."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import litellm
from aether_agent.config import AetherConfig, load_config
from aether_agent.core.types import ChatMessage, Role, ToolCall


class LLMClient:
    """Multi-provider asynchronous LLM client wrapping LiteLLM."""

    def __init__(
        self,
        config: AetherConfig | None = None,
        model: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.model = model or self.config.default_model

        # Configure LiteLLM global settings
        litellm.telemetry = False
        litellm.suppress_debug_info = not self.config.verbose

        if self.config.litellm_api_base:
            litellm.api_base = self.config.litellm_api_base

        # Propagate configured API keys into environment if set
        if self.config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        if self.config.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.config.anthropic_api_key
        if self.config.openrouter_api_key:
            os.environ["OPENROUTER_API_KEY"] = self.config.openrouter_api_key
        if self.config.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.config.gemini_api_key

    def _format_messages(
        self, messages: list[ChatMessage | dict[str, Any]]
    ) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted.append(msg.to_llm_dict())
            elif isinstance(msg, dict):
                formatted.append(msg)
            else:
                formatted.append({"role": "user", "content": str(msg)})
        return formatted

    async def chat_completion(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ChatMessage | AsyncIterator[str]:
        """Execute a chat completion request across providers via LiteLLM."""
        formatted_messages = self._format_messages(messages)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": stream,
            **kwargs,
        }

        if temperature is not None:
            params["temperature"] = temperature

        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        if stream:
            return self._stream_response(params)

        response = await litellm.acompletion(**params)
        return self._parse_completion_response(response)

    async def _stream_response(self, params: dict[str, Any]) -> AsyncIterator[str]:
        """Stream chunks from LiteLLM."""
        response = await litellm.acompletion(**params)
        async for chunk in response:
            delta = None
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta"):
                    delta = getattr(choice.delta, "content", None)
                elif isinstance(choice, dict):
                    delta = choice.get("delta", {}).get("content")
            elif isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}).get("content")

            if delta:
                yield delta

    def _parse_completion_response(self, response: Any) -> ChatMessage:
        """Parse LiteLLM response into a standard ChatMessage."""
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])

        if not choices:
            return ChatMessage(role=Role.ASSISTANT, content="")

        first_choice = choices[0]
        msg_obj = getattr(first_choice, "message", None)
        if msg_obj is None and isinstance(first_choice, dict):
            msg_obj = first_choice.get("message", {})

        content = getattr(msg_obj, "content", None)
        if content is None and isinstance(msg_obj, dict):
            content = msg_obj.get("content", "")

        raw_tool_calls = getattr(msg_obj, "tool_calls", None)
        if raw_tool_calls is None and isinstance(msg_obj, dict):
            raw_tool_calls = msg_obj.get("tool_calls", None)

        parsed_tool_calls: list[ToolCall] | None = None
        if raw_tool_calls:
            parsed_tool_calls = []
            for tc in raw_tool_calls:
                tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "")
                fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else {})
                fn_name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "")
                fn_args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else {})

                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {"raw_input": fn_args}

                parsed_tool_calls.append(
                    ToolCall(
                        id=str(tc_id),
                        function_name=str(fn_name),
                        arguments=fn_args if isinstance(fn_args, dict) else {},
                    )
                )

        return ChatMessage(
            role=Role.ASSISTANT,
            content=content or "",
            tool_calls=parsed_tool_calls if parsed_tool_calls else None,
        )
