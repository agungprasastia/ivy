"""Web search and page extraction tools using httpx."""

from __future__ import annotations

import asyncio
import concurrent.futures
import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus, urljoin

try:
    import httpx
except ImportError:  # openai>=3 / anthropic>=2 may bundle httpx2
    import httpx2 as httpx  # type: ignore[no-redef]

try:
    import httpcore
except ImportError:  # openai>=3 / anthropic>=2 bundle httpcore2
    import httpcore2 as httpcore  # type: ignore[no-redef]

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool
from ivy.tools.url_safety import resolve_and_validate_url

_DNS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="ivy-dns",
)


class _PinnedIPBackend(httpcore.AsyncNetworkBackend):
    """Connect TCP only to pre-validated IPs while keeping original Host and TLS SNI."""

    def __init__(self, pinned_hosts: dict[str, list[str]]) -> None:
        self._inner = httpcore.AnyIOBackend()
        self._pinned_hosts = pinned_hosts

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        key = host.strip().rstrip(".").lower()
        ips = self._pinned_hosts.get(key)
        if not ips:
            raise ValueError(f"Blocked unpinned connection to host: {host}")
        last_exc: Exception | None = None
        for ip_addr in ips:
            try:
                return await self._inner.connect_tcp(
                    ip_addr,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (OSError, httpcore.ConnectError) as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> Any:
        raise RuntimeError("Unix domain sockets are disabled for web_fetch.")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class _PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):  # type: ignore[misc]
    """httpx transport using _PinnedIPBackend to prevent DNS-rebinding TOCTOU."""

    def __init__(self, pinned_hosts: dict[str, list[str]]) -> None:
        super().__init__(verify=True, trust_env=False)
        ssl_ctx = getattr(self._pool, "_ssl_context", None)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_ctx,
            network_backend=_PinnedIPBackend(pinned_hosts),
            http1=True,
            http2=False,
        )


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

    async def _validate_hop(
        self, target_url: str, dns_timeout: float
    ) -> tuple[bool, str, str | None, list[str]]:
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    _DNS_EXECUTOR,
                    lambda: resolve_and_validate_url(
                        target_url, allow_private=self.allow_private_urls
                    ),
                ),
                timeout=dns_timeout,
            )
        except asyncio.TimeoutError:
            return False, f"DNS validation timed out after {dns_timeout:.1f}s.", None, []

    async def execute(self, url: str, max_chars: int = 8000, **kwargs: Any) -> ToolResult:
        clean_url = (url or "").strip()
        dns_timeout = min(self.timeout, 5.0)
        ok, err, hop_host, approved_ips = await self._validate_hop(clean_url, dns_timeout)
        if not ok or hop_host is None or not approved_ips:
            return ToolResult(tool_name=self.name, status="error", error=err or "Blocked URL target.")

        pinned_hosts: dict[str, list[str]] = {hop_host: approved_ips}
        transport = _PinnedAsyncHTTPTransport(pinned_hosts)

        try:
            current_url = clean_url
            resp: Any = None
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,
                trust_env=False,
                transport=transport,
            ) as client:
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
                        redir_ok, redir_err, next_host, next_ips = await self._validate_hop(
                            next_url, dns_timeout
                        )
                        if not redir_ok or next_host is None or not next_ips:
                            return ToolResult(
                                tool_name=self.name,
                                status="error",
                                error=redir_err or "Blocked redirect target.",
                            )
                        pinned_hosts[next_host] = next_ips
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
