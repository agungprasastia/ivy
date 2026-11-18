"""Pure UX helpers for command resolution, @mentions, clipboard, git branch, and export."""

from __future__ import annotations

import base64
import difflib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ivy.core.prompt import redact_secrets
from ivy.core.types import ChatMessage

COMMAND_ALIASES: dict[str, str] = {
    "/compress": "/compact",
    "/branch": "/fork",
    "/resume": "/session",
    "/sessions": "/session",
    "/quit": "/exit",
    "/reset": "/new",
    "/compose": "/prompt",
    "/skin": "/theme",
    "/ctx": "/context",
}

_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    "_ref_hermes",
}


def resolve_slash_command(
    raw_cmd: str, known_commands: Sequence[str]
) -> tuple[str | None, list[str], str]:
    """Resolve exact command, alias, unique prefix, or fuzzy typo suggestions.

    Returns:
        (resolved_cmd_or_None, suggestions_or_ambiguities, mode)
        where mode is one of: 'exact', 'alias', 'prefix', 'ambiguous', 'unknown'.
    """
    cmd = raw_cmd.strip().lower()
    if not cmd.startswith("/"):
        cmd = f"/{cmd}"

    known = list(dict.fromkeys(c.lower() for c in known_commands))
    if cmd in known:
        return cmd, [], "exact"

    if cmd in COMMAND_ALIASES:
        target = COMMAND_ALIASES[cmd]
        if target in known:
            return target, [], "alias"

    prefix_matches = [c for c in known if c.startswith(cmd)]
    if len(prefix_matches) == 1:
        return prefix_matches[0], [], "prefix"
    if len(prefix_matches) > 1:
        return None, prefix_matches, "ambiguous"

    close = difflib.get_close_matches(cmd, known + list(COMMAND_ALIASES), n=3, cutoff=0.55)
    return None, close, "unknown"


def get_git_branch(cwd: str | None = None) -> str | None:
    """Fast zero-subprocess lookup of current git branch from .git/HEAD."""
    cur = Path(cwd or os.getcwd()).resolve()
    for parent in (cur, *cur.parents):
        head_file = parent / ".git" / "HEAD"
        if head_file.is_file():
            try:
                raw = head_file.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                return None
            if raw.startswith("ref: "):
                ref = raw[5:].strip()
                if ref.startswith("refs/heads/"):
                    return ref[len("refs/heads/"):]
                return ref.rsplit("/", 1)[-1]
            return raw[:7] if raw else None
    return None


def list_workspace_files(cwd: str | None = None, limit: int = 250) -> list[str]:
    """Fast bounded workspace file listing for @file autocomplete."""
    root = Path(cwd or os.getcwd()).resolve()
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        rel_dir = Path(dirpath).relative_to(root)
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            rel = str(rel_dir / fn if rel_dir != Path(".") else Path(fn)).replace("\\", "/")
            found.append(rel)
            if len(found) >= limit:
                return found
    return found


def run_git_diff(staged: bool = False, stat: bool = False, cwd: str | None = None) -> str:
    """Run git diff cleanly inside cwd, returning output or helpful error text."""
    git_bin = shutil.which("git")
    if not git_bin:
        return "git executable not found on PATH."
    args = [git_bin, "diff"]
    if staged:
        args.append("--cached")
    if stat:
        args.append("--stat")
    try:
        proc = subprocess.run(
            args,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        out = proc.stdout.strip()
        if not out and proc.stderr.strip():
            return proc.stderr.strip()
        return out
    except Exception as exc:
        return f"git diff failed: {exc}"


_MENTION_RE = re.compile(r"(?<!\S)@((?:url:)?https?://[^\s]+|[A-Za-z0-9_./\\-]+)")


def expand_at_mentions(
    prompt_text: str,
    cwd: str | None = None,
    max_file_bytes: int = 12_000,
) -> tuple[str, list[str]]:
    """Expand @diff, @staged, @url:https://..., and @path mentions into attached context."""
    root = Path(cwd or os.getcwd()).resolve()
    attached_labels: list[str] = []
    blocks: list[str] = []
    seen: set[str] = set()

    for match in _MENTION_RE.finditer(prompt_text):
        token = match.group(1)
        if token in seen:
            continue
        seen.add(token)

        if token.startswith(("url:http://", "url:https://", "http://", "https://")):
            url = token[4:] if token.startswith("url:") else token
            blocks.append(
                f'<url_reference href="{url}">Use web_fetch on {url} if full page contents are needed.</url_reference>'
            )
            attached_labels.append(f"@{url}")
            continue

        if token == "diff":
            diff = run_git_diff(staged=False, cwd=str(root))
            if diff:
                blocks.append(f"<git_diff>\n{diff[:max_file_bytes]}\n</git_diff>")
                attached_labels.append("@diff")
            continue

        if token == "staged":
            diff = run_git_diff(staged=True, cwd=str(root))
            if diff:
                blocks.append(f"<git_staged_diff>\n{diff[:max_file_bytes]}\n</git_staged_diff>")
                attached_labels.append("@staged")
            continue

        candidate = (root / token).resolve()
        # Keep mentions within workspace root unless explicitly readable relative file
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            truncated = ""
            if len(content) > max_file_bytes:
                content = content[:max_file_bytes]
                truncated = "\n... [truncated]"
            rel_label = token.replace("\\", "/")
            line_count = content.count("\n") + 1
            blocks.append(f'<file path="{rel_label}">\n{content}{truncated}\n</file>')
            attached_labels.append(f"@{rel_label} ({line_count}L)")

    if not blocks:
        return prompt_text, []
    augmented = f"{prompt_text}\n\n[Attached Context]\n" + "\n\n".join(blocks)
    return augmented, attached_labels


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    """Copy text to the OS clipboard cleanly, with OSC 52 terminal fallback."""
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
            ps_cmd = (
                "$t = [System.Text.Encoding]::UTF8.GetString("
                f"[System.Convert]::FromBase64String('{b64}')); "
                "Set-Clipboard -Value $t"
            )
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
                check=True,
                capture_output=True,
                timeout=5,
            )
            return True, "Windows clipboard"
        if sys_name == "Darwin" and shutil.which("pbcopy"):
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True, timeout=5)
            return True, "pbcopy"
        for tool, args in (
            ("wl-copy", ["wl-copy"]),
            ("xclip", ["xclip", "-selection", "clipboard"]),
            ("xsel", ["xsel", "--clipboard", "--input"]),
        ):
            if shutil.which(tool):
                subprocess.run(args, input=text.encode("utf-8"), check=True, timeout=5)
                return True, tool
    except Exception:
        pass

    # Fallback: OSC 52 escape sequence (works over SSH / modern terminals)
    try:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        sys.stdout.write(f"\x1b]52;c;{encoded}\x07")
        sys.stdout.flush()
        return True, "OSC 52"
    except Exception as exc:
        return False, str(exc)


def format_token_count(count: int) -> str:
    """Format token counts compactly (e.g. 950 -> '950', 12400 -> '12.4k')."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def format_relative_time(iso_ts: str | None) -> str:
    """Convert ISO-8601 UTC timestamp to friendly relative label (e.g. '5m ago')."""
    if not iso_ts:
        return ""
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta_s = int((datetime.now(timezone.utc) - dt).total_seconds())
    except ValueError:
        return str(iso_ts)[:16].replace("T", " ")

    if delta_s < 45:
        return "just now"
    if delta_s < 3600:
        return f"{delta_s // 60}m ago"
    if delta_s < 86_400:
        return f"{delta_s // 3600}h ago"
    days = delta_s // 86_400
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days}d ago"
    return dt.astimezone().strftime("%b %d")


def export_session_markdown(
    session_id: str,
    title: str | None,
    messages: Iterable[ChatMessage],
) -> str:
    """Render session transcript to portable Markdown."""
    lines = [f"# Ivy Session: {title or session_id}", f"- Session ID: `{session_id}`", ""]
    for msg in messages:
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        role_label = msg.role.value.capitalize()
        if msg.name:
            role_label = f"{role_label} ({msg.name})"
        lines.append(f"## {role_label} · {ts}")
        lines.append("")
        lines.append(redact_secrets(msg.content or ""))
        lines.append("")
    return "\n".join(lines)


def export_session_json(
    session_id: str,
    title: str | None,
    messages: Iterable[ChatMessage],
) -> str:
    """Render session transcript to formatted JSON with secrets redacted."""
    scrubbed = [
        msg.model_copy(update={"content": redact_secrets(msg.content or "")})
        for msg in messages
    ]
    payload = {
        "session_id": session_id,
        "title": title,
        "messages": [msg.model_dump(mode="json") for msg in scrubbed],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


_CLOSED_THINK_RE = re.compile(
    r"<(think|thinking|reasoning_scratchpad)\b[^>]*>(.*?)</\1>\s*",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_THINK_RE = re.compile(
    r"<(think|thinking|reasoning_scratchpad)\b[^>]*>(.*)$",
    re.DOTALL | re.IGNORECASE,
)


def extract_thinking_blocks(text: str) -> tuple[str, str]:
    """Strip completed and in-progress <think>/<thinking> blocks cleanly.

    Returns:
        (visible_text, collected_thinking_text)
    """
    if not text:
        return "", ""

    thoughts: list[str] = []

    def _collect_closed(match: re.Match[str]) -> str:
        inner = match.group(2).strip()
        if inner:
            thoughts.append(inner)
        return ""

    cleaned = _CLOSED_THINK_RE.sub(_collect_closed, text)

    open_match = _OPEN_THINK_RE.search(cleaned)
    if open_match:
        trailing = open_match.group(2).strip()
        if trailing:
            thoughts.append(trailing)
        cleaned = cleaned[: open_match.start()]

    return cleaned.strip(), "\n\n".join(thoughts).strip()

