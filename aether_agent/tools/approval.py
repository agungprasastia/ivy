"""Safety guard and cross-channel human-in-the-loop approval policy."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import sys
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class SafetyApprovalPolicy:
    """Policy engine to intercept and approve or reject potentially destructive actions."""

    DEFAULT_DANGEROUS_PATTERNS = [
        # POSIX destructive patterns
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b",
        r"\brm\b.*\s+-[a-zA-Z]*r\b.*\s+-[a-zA-Z]*f\b",
        r"\brm\b.*\s+-[a-zA-Z]*f\b.*\s+-[a-zA-Z]*r\b",
        r"\bmkfs(\.[a-zA-Z0-9]+)?\b",
        r"\bdd\s+if=",
        r">\s*/dev/sd[a-z]",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\};:",  # fork bomb
        # Windows / DOS destructive patterns
        r"\bdel\b(?:\s+/[a-zA-Z]+)*\s+/[a-zA-Z]*[sS]\b",
        r"\b(?:rmdir|rd)\b(?:\s+/[a-zA-Z]+)*\s+/[sS]\b",
        r"\bformat\s+[a-zA-Z]:",
        # PowerShell destructive patterns
        r"\bRemove-Item\b.*(-Recurse|-Force)",
        r"\bClear-Disk\b",
        r"\bFormat-Volume\b",
    ]

    def __init__(
        self,
        auto_approve: bool = False,
        interactive: bool = False,
        unattended: bool = False,
        approval_timeout: float = 60.0,
        approval_callback: Callable[[str], bool | Awaitable[bool]] | None = None,
        custom_dangerous_patterns: list[str] | None = None,
    ) -> None:
        self.auto_approve = auto_approve
        self.interactive = interactive
        self.unattended = unattended
        self.approval_timeout = approval_timeout
        self.approval_callback = approval_callback

        patterns = list(self.DEFAULT_DANGEROUS_PATTERNS)
        if custom_dangerous_patterns:
            patterns.extend(custom_dangerous_patterns)
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def is_dangerous(self, command: str) -> bool:
        """Check whether a command matches any dangerous destructive patterns."""
        normalized = command.strip()
        return any(pattern.search(normalized) for pattern in self.compiled_patterns)

    async def check_approval(self, command: str) -> bool:
        """Evaluate approval for command execution across CLI, bot, and cron modes."""
        if not self.is_dangerous(command):
            return True

        if self.auto_approve:
            return True

        if self.unattended:
            logger.warning(
                "Unattended safety policy blocked dangerous command: %s", command
            )
            return False

        if self.approval_callback is not None:
            try:
                res = self.approval_callback(command)
                if inspect.isawaitable(res):
                    return bool(await asyncio.wait_for(res, timeout=self.approval_timeout))
                return bool(res)
            except asyncio.TimeoutError:
                logger.warning("Approval request timed out for command: %s", command)
                return False

        if self.interactive and sys.stdin.isatty():
            print(
                f"\n[SAFETY GUARD] Potentially destructive command detected:\n  {command}"
            )
            try:
                response = await asyncio.to_thread(
                    lambda: input("Do you approve this execution? [y/N]: ").strip().lower()
                )
                return response in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                return False

        return False
