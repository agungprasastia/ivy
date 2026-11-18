"""Web search and page extraction tools using httpx."""

from __future__ import annotations

import asyncio
import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus, urljoin

try:
    import httpx
except ImportError:  # openai>=3 / anthropic>=2 may bundle httpx2
    import httpx2 as httpx  # type: ignore[no-redef]

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool
from ivy.tools.url_safety import validate_url_target


class _TextExtractor(HTMLParser):
    """Lightweight HTML-to-text extractor skipping script/style tags."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style", "noscript"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            cleaned = data.strip()
            if cleaned:
                self._chunks.append(cleaned)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def _extract_visible_text(raw_html: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw_html)
    parser.close()
    text = parser.get_text()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class WebFetchTool(BaseTool):
    """Fetch a web URL and extract readable text content."""

    name = "web_fetch"
    concurrent_safe = True
    description = "Fetch an HTTP/HTTPS URL and extract its readable text content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Target HTTP or HTTPS URL."},
            "max_chars": {"type": "integer", "description": "Maximum characters to return.", "default": 8000},
        },
        "required": ["url"],
    }

    def __init__(
        self,
        timeout: float = 15.0,
        allow_private_urls: bool = False,
        max_redirects: int = 5,
    ) -> None:
        self.timeout = timeout
        self.allow_private_urls = allow_private_urls
        self.max_redirects = max(0, max_redirects)

    async def execute(self, url: str, max_chars: int = 8000, **kwargs: Any) -> ToolResult:
        clean_url = (url or "").strip()
        ok, err = await asyncio.to_thread(
            validate_url_target, clean_url, allow_private=self.allow_private_urls
        )
        if not ok:
            return ToolResult(tool_name=self.name, status="error", error=err)

        try:
            current_url = clean_url
            resp: Any = None
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                for hop in range(self.max_redirects + 1):
                    resp = await client.get(current_url, headers={"User-Agent": "IvyAgent/0.1"})
                    status_code = getattr(resp, "status_code", 200)
                    headers = getattr(resp, "headers", {}) or {}
                    location = headers.get("Location") or headers.get("location")
                    if status_code in (301, 302, 303, 307, 308) and location:
                        if hop >= self.max_redirects:
                            return ToolResult(
                                tool_name=self.name,
                                status="error",
                                error=f"Too many redirects (max {self.max_redirects})",
                            )
                        next_url = urljoin(current_url, str(location))
                        redir_ok, redir_err = await asyncio.to_thread(
                            validate_url_target,
                            next_url,
                            allow_private=self.allow_private_urls,
                        )
                        if not redir_ok:
                            return ToolResult(tool_name=self.name, status="error", error=redir_err)
                        current_url = next_url
                        continue
                    break

                assert resp is not None
                resp.raise_for_status()
            text = _extract_visible_text(resp.text)
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... [truncated]"
            return ToolResult(tool_name=self.name, status="success", output=text)
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Failed to fetch URL: {exc}",
            )


class WebSearchTool(BaseTool):
    """Perform web search via DuckDuckGo HTML endpoint without requiring API keys."""

    name = "web_search"
    concurrent_safe = True
    description = "Search the web for a query and return top result titles and URLs."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string."},
            "max_results": {"type": "integer", "description": "Maximum results to return.", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    async def execute(self, query: str, max_results: int = 5, **kwargs: Any) -> ToolResult:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "IvyAgent/0.1"})
                resp.raise_for_status()

            links = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                resp.text,
                re.IGNORECASE | re.DOTALL,
            )
            results: list[str] = []
            for href, raw_title in links[:max_results]:
                title = html.unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
                results.append(f"- {title}: {href}")

            output = "\n".join(results) if results else "No search results found."
            return ToolResult(tool_name=self.name, status="success", output=output)
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Web search failed: {exc}",
            )
