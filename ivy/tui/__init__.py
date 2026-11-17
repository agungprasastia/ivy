"""Rich Terminal TUI, prompt-toolkit input, and CLI entrypoint for Ivy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ivy.tui.app import IvyCLIApp, build_parser, main

__all__ = ["IvyCLIApp", "build_parser", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from ivy.tui import app
        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
