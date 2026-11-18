"""Zero-dependency static security scanner for SKILL.md definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_INVISIBLE_UNICODE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff\U000e0000-\U000e007f]")
_SENSITIVE_TARGET_RE = re.compile(
    r"(?:~/\.ssh|/\.ssh/|\.aws/credentials|/etc/shadow|\.git-credentials|\.netrc|"
    r"(?<![A-Za-z0-9_])\.env(?!\.(?:example|template|sample)(?![A-Za-z0-9_-]))(?:\.[A-Za-z0-9_-]+)?(?![A-Za-z0-9_-]))",
    re.IGNORECASE,
)
_NETWORK_CMD_RE = re.compile(
    r"\b(?:curl|wget|nc|ncat|scp|Invoke-WebRequest|Invoke-RestMethod|iwr|irm|requests\.post|urllib\.request)\b",
    re.IGNORECASE,
)

_PROMPT_INJECTION_RE = re.compile(
    r"\b(?:ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|"
    r"disregard\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
    r"bypass\s+(?:safety|security|approval)|"
    r"do\s+not\s+tell\s+the\s+user)\b",
    re.IGNORECASE,
)

_DANGER_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "remote_pipe_exec",
        re.compile(
            r"\b(?:curl|wget|iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^\n|]*\|\s*(?:sh|bash|zsh|pwsh|powershell|python\d?|iex|Invoke-Expression)\b",
            re.IGNORECASE,
        ),
        "Remote download piped directly to a shell/interpreter",
    ),
    (
        "powershell_iex",
        re.compile(r"\b(?:Invoke-Expression|iex)\s*\(?.*\b(?:iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b", re.IGNORECASE),
        "PowerShell remote download executed via Invoke-Expression",
    ),
    (
        "reverse_shell",
        re.compile(r"(?:/dev/tcp/\S+/\d+|\b(?:nc|ncat)\b[^\n]*\s-e\s|\bbash\s+-i\s*>&)", re.IGNORECASE),
        "Reverse shell / interactive socket redirection pattern",
    ),
    (
        "destructive_disk_cmd",
        re.compile(r"(?:\brm\s+-[rRf]{1,3}\s+(?:/|~|\$HOME)\b|\bmkfs\.[a-z0-9]+\b|\bdd\s+if=/dev/[^\s]+\s+of=/dev/[sh]d)", re.IGNORECASE),
        "Destructive system wipe or raw disk overwrite command",
    ),
    (
        "prompt_injection_override",
        _PROMPT_INJECTION_RE,
        "Prompt-injection or user-deception instruction",
    ),
    (
        "hardcoded_secret",
        re.compile(
            r"(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
            r"-----BEGIN\s+(?:RSA\s+|OPENSSH\s+|EC\s+)?PRIVATE\s+KEY-----)"
        ),
        "Hardcoded API key or private cryptographic key material",
    ),
]


@dataclass(frozen=True)
class SkillFinding:
    """Single security issue detected inside a skill definition."""

    severity: str
    rule_id: str
    reason: str
    snippet: str = ""


@dataclass(frozen=True)
class SkillScanResult:
    """Result of scanning a candidate skill definition."""

    safe: bool
    findings: list[SkillFinding] = field(default_factory=list)


def check_prompt_safety(text: str) -> str | None:
    """Return a security error string if text hides invisible Unicode or prompt injection."""
    if not text:
        return None
    if _INVISIBLE_UNICODE_RE.search(text):
        return "Blocked by security policy: hidden invisible or bidi Unicode character detected."
    if _PROMPT_INJECTION_RE.search(text):
        return "Blocked by security policy: prompt-injection or override phrase detected."
    return None


def scan_skill_content(
    name: str,
    description: str,
    instructions: str,
    metadata: dict[str, Any] | None = None,
) -> SkillScanResult:
    """Statically scan skill fields and metadata for RCE, exfiltration, leaked secrets, and injection."""
    meta_parts = (
        [f"{k}={v}" for k, v in metadata.items()]
        if isinstance(metadata, dict)
        else []
    )
    combined = "\n".join([name, description, instructions, *meta_parts])
    findings: list[SkillFinding] = []

    uni_match = _INVISIBLE_UNICODE_RE.search(combined)
    if uni_match:
        findings.append(
            SkillFinding(
                severity="danger",
                rule_id="invisible_unicode",
                reason="Hidden zero-width or bidirectional override Unicode character detected",
                snippet=repr(uni_match.group(0)),
            )
        )

    for rule_id, pattern, reason in _DANGER_PATTERNS:
        m = pattern.search(combined)
        if m:
            findings.append(
                SkillFinding(
                    severity="danger",
                    rule_id=rule_id,
                    reason=reason,
                    snippet=m.group(0)[:80],
                )
            )

    # Flag credential path access combined with outbound network command
    sens = _SENSITIVE_TARGET_RE.search(combined)
    net = _NETWORK_CMD_RE.search(combined)
    if sens and net:
        findings.append(
            SkillFinding(
                severity="danger",
                rule_id="credential_exfiltration",
                reason="References sensitive credential files alongside outbound network commands",
                snippet=f"{net.group(0)} ... {sens.group(0)}",
            )
        )

    safe = not any(f.severity == "danger" for f in findings)
    return SkillScanResult(safe=safe, findings=findings)


def format_scan_rejection(result: SkillScanResult) -> str:
    """Format blocking scan findings into a concise error string."""
    reasons = "; ".join(f"[{f.rule_id}] {f.reason}" for f in result.findings)
    return f"Skill rejected by security scanner: {reasons}"
