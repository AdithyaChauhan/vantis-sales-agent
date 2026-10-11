"""API protection: the real client IP, the /api/chat rate limit and the sales API key (PLAN.md part 3.11).

Client IP on Render: traffic arrives through Cloudflare and Render's load balancer, so the connection's
address is a proxy. Cloudflare writes CF-Connecting-IP on every request and overwrites any value the
caller sent; it only appends to X-Forwarded-For, whose first entry is whatever the caller claims. So we
read the one header named in CLIENT_IP_HEADER and never X-Forwarded-For. IPs are never logged; the
request log only says where the IP came from ("header" or "fallback").
"""
import hmac
import ipaddress
import logging
import math

from fastapi import Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app.core.config import get_settings
from app.core.logging import request_id_var
from app.core.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

MIN_KEY_CHARS = 16  # a shorter SALES_API_KEY counts as not set (fail closed)
KEY_FAILURE_LIMIT = (10, 15 * 60)  # wrong keys per client IP per 15 minutes, then 429 until the window ends
INVALID_KEY = "Missing or invalid API key. Send it in the X-API-Key header."
NOT_CONFIGURED = "The sales API is not configured on this server."
TOO_MANY_KEYS = "Too many wrong API keys from your network. Please try again in {minutes} minutes."
TOO_FAST = "You're sending messages a little fast. Please wait {seconds} seconds and try again."
DAY_LIMIT = "You've reached today's message limit for this chat. Please try again in about {wait}."

_settings = get_settings()
CHAT_LIMITER = RateLimiter([(_settings.chat_limit_per_minute, 60), (_settings.chat_limit_per_day, 86_400)])
KEY_FAILURES = RateLimiter([KEY_FAILURE_LIMIT])
_warned_no_header = False


class ApiError(Exception):
    """An error answered as JSON {"detail", **extra, "request_id"} (handler in app/main.py)."""

    def __init__(self, status_code: int, detail: str, **extra):
        super().__init__(detail)
        self.status_code, self.detail, self.extra = status_code, detail, extra


def error_response(status_code: int, detail: str, **extra) -> JSONResponse:
    """The JSON shape of every error. A `retry_after` value is also sent as the Retry-After header."""
    headers = {"Retry-After": str(extra["retry_after"])} if "retry_after" in extra else None
    content = {"detail": detail, **extra, "request_id": request_id_var.get()}
    return JSONResponse(status_code=status_code, content=content, headers=headers)


def client_ip(request: Request) -> tuple[str, str]:
    """(ip, source). source is "header" when CLIENT_IP_HEADER is set and holds one valid IP, otherwise
    "fallback": the connection's address (on Render that is a proxy, so everyone would share one limit:
    stricter, never looser)."""
    global _warned_no_header
    header = get_settings().client_ip_header
    if header:
        value = request.headers.get(header, "").strip()
        try:
            return str(ipaddress.ip_address(value)), "header"
        except ValueError:
            if not _warned_no_header:  # once per process; the request log shows every fallback
                _warned_no_header = True
                logger.warning("client IP header %s missing or invalid; using the connection address", header)
    return (request.client.host if request.client else "unknown"), "fallback"


def limit_key(ip: str) -> str:
    """What the limits count by: an IPv4 address, or an IPv6 /64 network (one user usually has a whole /64)."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return ip  # not an IP (the test client's "testclient")
    if address.version == 6 and address.ipv4_mapped:
        return str(address.ipv4_mapped)
    if address.version == 6:
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def _client_key(request: Request) -> str:
    ip, source = client_ip(request)
    request.state.ip_source = source  # the request log line shows it (app/main.py)
    return limit_key(ip)


def _wait_words(seconds: int) -> str:
    if seconds < 3600:
        minutes = math.ceil(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = round(seconds / 3600)
    return f"{hours} hour{'s' if hours != 1 else ''}"


def chat_rate_limit(request: Request) -> None:
    """/api/chat dependency: CHAT_LIMIT_PER_MINUTE and CHAT_LIMIT_PER_DAY per client IP.

    Every request counts, including one rejected as invalid (this runs before the body is validated).
    Over a limit: ApiError 429 with Retry-After and a message the chat page can show.
    """
    wait = CHAT_LIMITER.hit(_client_key(request))
    if wait is None:
        return
    seconds = max(1, math.ceil(wait))
    if seconds <= 60:
        limit, detail = "minute", TOO_FAST.format(seconds=seconds)
    else:
        limit, detail = "day", DAY_LIMIT.format(wait=_wait_words(seconds))
    logger.warning("chat rate limit reached", extra={"limit": limit})
    raise ApiError(429, detail, retry_after=seconds)


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="The sales API key (SALES_API_KEY). In /docs: click Authorize and paste it.",
)


def require_sales_key(request: Request, key: str | None = Security(api_key_header)) -> None:
    """Sales endpoints dependency. Fails closed: 503 when SALES_API_KEY is not set (or too short).

    Only wrong keys count toward the lockout (10 per 15 minutes per client IP); a request without a key
    gets 401 and is not counted, so opening a sales URL can never lock anyone out. While locked, even the
    right key gets 429. Keys are compared in constant time and never logged.
    """
    expected = get_settings().sales_api_key
    if len(expected) < MIN_KEY_CHARS:
        logger.warning("sales API refused: SALES_API_KEY is not set or shorter than %d characters", MIN_KEY_CHARS)
        raise ApiError(503, NOT_CONFIGURED)
    client = _client_key(request)
    wait = KEY_FAILURES.blocked(client)
    if wait is not None:
        seconds = max(1, math.ceil(wait))
        raise ApiError(429, TOO_MANY_KEYS.format(minutes=math.ceil(seconds / 60)), retry_after=seconds)
    if not key:
        raise ApiError(401, INVALID_KEY)
    if not hmac.compare_digest(key.encode(), expected.encode()):
        KEY_FAILURES.hit(client)
        logger.warning("sales API: wrong API key")
        raise ApiError(401, INVALID_KEY)
