"""Cross-platform session routing and continuity hub."""

from __future__ import annotations


class SessionHub:
    """Maps platform user identities to unified Ivy session IDs."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str], str] = {}

    def bind_user_session(self, platform: str, user_id: str, session_id: str) -> None:
        """Bind a platform user to an existing or shared session ID."""
        self._bindings[(platform.lower(), str(user_id))] = session_id

    def resolve_session(
        self,
        platform: str,
        user_id: str,
        explicit_session: str | None = None,
    ) -> str:
        """Resolve the effective session ID for an incoming message."""
        if explicit_session:
            return explicit_session
        key = (platform.lower(), str(user_id))
        if key in self._bindings:
            return self._bindings[key]
        return f"{platform.lower()}:{user_id}"
