"""Safety guard and cross-channel human-in-the-loop approval policy."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import re
import sys
from typing import Any, AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)

_ApprovalCallback = Callable[[str], Any | Awaitable[Any]]
_scoped_callback_var: contextvars.ContextVar[_ApprovalCallback | None] = (
    contextvars.ContextVar("ivy_scoped_approval_callback", default=None)
)
_scoped_unattended_var: contextvars.ContextVar[bool | None] = (
    contextvars.ContextVar("ivy_scoped_unattended", default=None)
)


class SafetyApprovalPolicy:
    """Policy engine to intercept and approve or reject potentially destructive actions."""

    DEFAULT_DANGEROUS_PATTERNS = [
        # POSIX destructive patterns
        r"\brm\b.*\s+(?:-[a-zA-Z]*[rR]|--recursive)\b",
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b",
        r"\bmkfs(\.[a-zA-Z0-9]+)?\b",
        r"\bdd\s+if=",
        r">\s*/dev/sd[a-z]",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\};:",  # fork bomb
        # Windows / DOS destructive patterns
        r"\b(?:del|erase)\b(?:\s+/[a-zA-Z]+)*\s+/[a-zA-Z]*[sS]\b",
        r"\b(?:del|erase)\b.*\s+/[qQ]\b.*\s+/[fF]\b",
        r"\b(?:del|erase)\b.*\s+/[fF]\b.*\s+/[qQ]\b",
        r"\b(?:rmdir|rd)\b(?:\s+/[a-zA-Z]+)*\s+/[sS]\b",
        r"\bformat\s+[a-zA-Z]:",
        # PowerShell Remove-Item and aliases (rm, del, rd, rmdir, ri, erase)
        r"\b(?:Remove-Item|rm|del|rd|rmdir|ri|erase)\b.*\s+-(?:Recurse|Force|r|fo)\b",
        r"\bClear-Disk\b",
        r"\bFormat-Volume\b",
    ]

    def __init__(
        self,
        auto_approve: bool = False,
        interactive: bool = False,
        unattended: bool = False,
        approval_timeout: float = 60.0,
        approval_callback: _ApprovalCallback | None = None,
        custom_dangerous_patterns: list[str] | None = None,
    ) -> None:
        self.auto_approve = auto_approve
        self.interactive = interactive
        self.unattended = unattended
        self.approval_timeout = approval_timeout
        self.approval_callback = approval_callback
        self.session_approved_commands: set[str] = set()

        patterns = list(self.DEFAULT_DANGEROUS_PATTERNS)
        if custom_dangerous_patterns:
            patterns.extend(custom_dangerous_patterns)
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    @contextlib.asynccontextmanager
    async def scoped_context(
        self,
        approval_callback: _ApprovalCallback | None = None,
        unattended: bool | None = None,
    ) -> AsyncIterator[None]:
        """Temporarily override approval callback or unattended mode for current async task."""
        cb_tok = _scoped_callback_var.set(approval_callback)
        un_tok = _scoped_unattended_var.set(unattended)
        try:
            yield
        finally:
            _scoped_callback_var.reset(cb_tok)
            _scoped_unattended_var.reset(un_tok)

    def is_dangerous(self, command: str) -> bool:
        """Check whether a command matches any dangerous destructive patterns."""
        normalized = command.strip()
        return any(pattern.search(normalized) for pattern in self.compiled_patterns)

    def _interpret_callback_result(self, command: str, raw: Any) -> tuple[bool, str | None]:
        """Interpret bool, scope string, tuple(decision, feedback), or dict decision."""
        norm = command.strip()
        if isinstance(raw, bool):
            return raw, None
        if isinstance(raw, str):
            val = raw.strip().lower()
            if val == "session":
                self.session_approved_commands.add(norm)
                return True, None
            if val in ("always", "yolo"):
                self.auto_approve = True
                return True, None
            if val in ("once", "yes", "y", "true"):
                return True, None
            if val in ("no", "n", "deny", "false", ""):
                return False, None
            return False, raw.strip()
        if isinstance(raw, tuple) and len(raw) >= 2:
            approved, reason = raw[0], raw[1]
            if isinstance(approved, str):
                ok, _ = self._interpret_callback_result(command, approved)
                return ok, (str(reason).strip() if reason else None)
            return bool(approved), (str(reason).strip() if reason else None)
        if isinstance(raw, dict):
            approved = bool(raw.get("approved", False))
            scope = str(raw.get("scope", "once")).lower()
            if approved:
                if scope == "session":
                    self.session_approved_commands.add(norm)
                elif scope in ("always", "yolo"):
                    self.auto_approve = True
                return True, None
            fb = raw.get("feedback")
            return False, (str(fb).strip() if fb else None)
        return bool(raw), None

    async def evaluate_approval(self, command: str) -> tuple[bool, str | None]:
        """Evaluate command safety approval, returning (approved, optional_user_feedback)."""
        if not self.is_dangerous(command):
            return True, None

        if self.auto_approve or command.strip() in self.session_approved_commands:
            return True, None

        scoped_unattended = _scoped_unattended_var.get()
        is_unattended = self.unattended if scoped_unattended is None else scoped_unattended
        if is_unattended:
            logger.warning(
                "Unattended safety policy blocked dangerous command: %s", command
            )
            return False, None

        callback = _scoped_callback_var.get() or self.approval_callback
        if callback is not None:
            try:
                res = callback(command)
                if inspect.isawaitable(res):
                    res = await asyncio.wait_for(res, timeout=self.approval_timeout)
                return self._interpret_callback_result(command, res)
            except asyncio.TimeoutError:
                logger.warning("Approval request timed out for command: %s", command)
                return False, None
            except Exception as exc:
                logger.warning("Approval callback failed safely (rejecting): %s", exc)
                return False, None

        if self.interactive and sys.stdin.isatty():
            print(
                f"\n[SAFETY GUARD] Potentially destructive command detected:\n  {command}"
            )
            try:
                response = await asyncio.to_thread(
                    lambda: input("Do you approve this execution? [y/N]: ").strip().lower()
                )
                return (response in ("y", "yes")), None
            except (EOFError, KeyboardInterrupt):
                return False, None

        return False, None

    async def check_approval(self, command: str) -> bool:
        """Evaluate approval for command execution across CLI, bot, and cron modes."""
        approved, _ = await self.evaluate_approval(command)
        return approved
