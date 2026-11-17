"""Asynchronous multi-platform messaging gateway, turn leasing, and session routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ivy.gateway.server import GatewayServer
    from ivy.gateway.session_hub import SessionHub
    from ivy.gateway.turn_lease import TurnLeaseManager

__all__ = ["GatewayServer", "SessionHub", "TurnLeaseManager"]


def __getattr__(name: str) -> Any:
    if name == "GatewayServer":
        from ivy.gateway.server import GatewayServer
        return GatewayServer
    if name == "SessionHub":
        from ivy.gateway.session_hub import SessionHub
        return SessionHub
    if name == "TurnLeaseManager":
        from ivy.gateway.turn_lease import TurnLeaseManager
        return TurnLeaseManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
