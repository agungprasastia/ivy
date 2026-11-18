"""SSRF protection helpers for validating HTTP/HTTPS target URLs and redirects."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any, Callable
from urllib.parse import urlsplit

_METADATA_HOSTS = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "169.254.169.254",
        "169.254.170.2",
        "169.254.169.253",
        "100.100.100.200",
        "fd00:ec2::254",
    }
)
_METADATA_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("169.254.170.2"),
        ipaddress.ip_address("169.254.169.253"),
        ipaddress.ip_address("100.100.100.200"),
        ipaddress.ip_address("fd00:ec2::254"),
    }
)
_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")


def _unwrap_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        return ip_obj.ipv4_mapped
    return ip_obj


def _check_ip_address(
    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_private: bool = False,
) -> str | None:
    ip_obj = _unwrap_ip(ip_obj)
    if ip_obj in _METADATA_IPS or ip_obj.is_link_local:
        return f"Blocked access to metadata/link-local IP address: {ip_obj}"
    if ip_obj.is_multicast or ip_obj.is_unspecified:
        return f"Blocked access to non-routable IP address: {ip_obj}"
    if not allow_private:
        in_cgnat = isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj in _CGNAT_NET
        if ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_reserved or in_cgnat:
            return f"Blocked access to private or loopback IP address: {ip_obj}"
    return None


def resolve_and_validate_url(
    url: str,
    *,
    allow_private: bool = False,
    resolver: Callable[..., Any] | None = None,
) -> tuple[bool, str, str | None, list[str]]:
    """Validate URL scheme, port, hostname, and resolved IPs, returning approved IPs."""
    try:
        parsed = urlsplit(url.strip())
        port_val = parsed.port
    except ValueError as exc:
        return False, f"Invalid URL '{url}': {exc}", None, []

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return False, "URL must start with http:// or https://", None, []

    raw_host = (parsed.hostname or "").strip().rstrip(".").lower()
    if not raw_host:
        return False, f"Invalid URL '{url}': missing hostname.", None, []

    if raw_host in _METADATA_HOSTS:
        return False, f"Blocked access to cloud metadata host: {raw_host}", None, []

    if not allow_private and (
        raw_host == "localhost"
        or raw_host.endswith(".localhost")
        or raw_host.endswith(".local")
        or raw_host.endswith(".internal")
    ):
        return False, f"Blocked access to private hostname: {raw_host}", None, []

    # 1. Direct IP literal check
    try:
        ip_literal = ipaddress.ip_address(raw_host)
    except ValueError:
        ip_literal = None

    if ip_literal is not None:
        reason = _check_ip_address(ip_literal, allow_private=allow_private)
        if reason is not None:
            return False, reason, None, []
        return True, "", raw_host, [str(_unwrap_ip(ip_literal))]

    # 2. DNS resolution check for domain hostnames
    port = port_val or (443 if scheme == "https" else 80)
    resolve_fn = resolver or socket.getaddrinfo
    try:
        records = resolve_fn(raw_host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return False, f"DNS resolution failed for '{raw_host}': {exc}", None, []

    resolved_ips: list[str] = []
    for rec in records:
        sockaddr = rec[4] if len(rec) >= 5 else None
        if isinstance(sockaddr, tuple) and sockaddr:
            resolved_ips.append(str(sockaddr[0]).split("%", 1)[0])

    if not resolved_ips:
        return False, f"DNS resolution returned no addresses for '{raw_host}'.", None, []

    approved_ips: list[str] = []
    for ip_str in dict.fromkeys(resolved_ips):
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        reason = _check_ip_address(ip_obj, allow_private=allow_private)
        if reason is not None:
            return False, f"{reason} (resolved from {raw_host})", None, []
        approved_ips.append(str(_unwrap_ip(ip_obj)))

    if not approved_ips:
        return False, f"DNS resolution returned no valid IP addresses for '{raw_host}'.", None, []

    return True, "", raw_host, approved_ips


def validate_url_target(
    url: str,
    *,
    allow_private: bool = False,
    resolver: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Validate URL scheme, hostname, and resolved IP addresses against SSRF targets."""
    ok, reason, _host, _ips = resolve_and_validate_url(
        url, allow_private=allow_private, resolver=resolver
    )
    return ok, reason
