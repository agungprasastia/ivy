"""Rich Terminal TUI, prompt-toolkit input, and CLI entrypoint for Aether Agent."""

from aether_agent.tui.app import AetherCLIApp, build_parser, main

__all__ = ["AetherCLIApp", "build_parser", "main"]
