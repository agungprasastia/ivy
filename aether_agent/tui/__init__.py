"""Rich Terminal TUI, prompt-toolkit input, and CLI entrypoint for Aether Agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether_agent.tui.app import AetherCLIApp, build_parser, main

__all__ = ["AetherCLIApp", "build_parser", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from aether_agent.tui import app
        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
