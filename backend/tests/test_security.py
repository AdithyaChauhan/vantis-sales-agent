"""Client IP, sales API key, failed-key lockout, OpenAPI and CORS (no database needed)."""
import logging

import pytest
from starlette.requests import Request

from app.api import sales, security
from app.api.security import client_ip, limit_key
from app.core.config import get_settings

KEY = "test-sales-key-0123456789"
SALES_ROUTES = [
    ("/api/leads", "get"), ("/api/leads", "post"), ("/api/leads/{lead_id}", "get"),
    ("/api/callback", "post"), ("/api/support-requests", "get"),
]


def make_request(headers: list[tuple[str, str]], host: str = "10.0.0.1") -> Request:
    raw = [(name.lower().encode(), value.encode()) for name, value in headers]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw, "client": (host, 5000)})


@pytest.fixture
def key_set(monkeypatch):
    monkeypatch.setattr(get_settings(), "sales_api_key", KEY)


@pytest.fixture
def ip_header(monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", "CF-Connecting-IP")


# ---------- client IP ----------

def test_without_a_configured_header_the_connection_address_is_used(monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", "")
    spoofed = [("X-Forwarded-For", "1.2.3.4"), ("CF-Connecting-IP", "5.6.7.8"), ("X-Real-IP", "9.9.9.9")]
    assert client_ip(make_request(spoofed)) == ("10.0.0.1", "fallback")


def test_configured_header_with_one_valid_ip_is_used(ip_header):
    request = make_request([("CF-Connecting-IP", " 203.0.113.7 "), ("X-Forwarded-For", "1.2.3.4")])
    assert client_ip(request) == ("203.0.113.7", "header")


@pytest.mark.parametrize("value", ["", "1.2.3.4, 5.6.7.8", "not-an-ip", "203.0.113.7:443", "999.1.1.1"])
def test_a_missing_or_invalid_header_falls_back_to_the_connection(ip_header, value):
    headers = [("CF-Connecting-IP", value)] if value else []
    assert client_ip(make_request(headers + [("X-Forwarded-For", "1.2.3.4")])) == ("10.0.0.1", "fallback")


def test_x_forwarded_for_is_never_read(monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", "X-Forwarded-For")  # even if misconfigured
    # Cloudflare appends to X-Forwarded-For, so it holds a list; a list is never accepted as one IP
    request = make_request([("X-Forwarded-For", "1.2.3.4, 203.0.113.7")])
    assert client_ip(request) == ("10.0.0.1", "fallback")


def test_limit_key_groups_ipv6_by_64_and_keeps_ipv4():
    assert limit_key("203.0.113.7") == "203.0.113.7"
    assert limit_key("2001:db8:1:2:aaaa::1") == limit_key("2001:db8:1:2:bbbb::9") == "2001:db8:1:2::/64"
    assert limit_key("2001:db8:1:3::1") != limit_key("2001:db8:1:2::1")
    assert limit_key("::ffff:203.0.113.7") == "203.0.113.7"
    assert limit_key("testclient") == "testclient"


# ---------- the sales key ----------

def get_leads(client, key=None, **headers):
    if key is not None:
        headers["X-API-Key"] = key
    return client.get("/api/leads", headers=headers)


def test_no_key_is_401(client, key_set, use_engine):
    use_engine(None)
    response = get_leads(client)
    assert response.status_code == 401
    assert set(response.json()) == {"detail", "request_id"}


def test_wrong_key_is_401(client, key_set, use_engine):
    use_engine(None)
    assert get_leads(client, "wrong-key-0123456789").status_code == 401
    assert get_leads(client, KEY + "x").status_code == 401


def test_right_key_passes(client, key_set, use_engine):
    use_engine(None)  # past the key check, the missing database is the next answer
    response = get_leads(client, KEY)
    assert response.status_code == 503
    assert response.json()["detail"] == sales.DB_NOT_CONFIGURED


@pytest.mark.parametrize("configured", ["", "short-key"])
def test_key_not_configured_refuses_everything(client, monkeypatch, use_engine, configured):
    monkeypatch.setattr(get_settings(), "sales_api_key", configured)
    use_engine(None)
    for key in (None, "", configured, KEY):
        response = get_leads(client, key)
        assert response.status_code == 503
        assert response.json()["detail"] == security.NOT_CONFIGURED


def test_ten_wrong_keys_lock_the_client_out(client, key_set, use_engine):
    use_engine(None)
    assert [get_leads(client, f"wrong-{i}").status_code for i in range(10)] == [401] * 10
    locked = get_leads(client, "wrong-10")
    assert locked.status_code == 429
    assert 1 <= int(locked.headers["Retry-After"]) <= 900
    assert locked.json()["retry_after"] == int(locked.headers["Retry-After"])
    assert get_leads(client, KEY).status_code == 429  # even the right key while locked


def test_spoofed_ip_headers_do_not_reset_the_lockout(client, key_set, use_engine, monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", "")
    use_engine(None)
    for i in range(10):
        get_leads(client, f"wrong-{i}", **{"X-Forwarded-For": f"10.1.0.{i}", "CF-Connecting-IP": f"10.2.0.{i}"})
    response = get_leads(client, "wrong-x", **{"X-Forwarded-For": "10.1.0.99", "CF-Connecting-IP": "10.2.0.99"})
    assert response.status_code == 429


def test_requests_without_a_key_never_lock_anyone_out(client, key_set, use_engine):
    use_engine(None)
    assert [get_leads(client).status_code for _ in range(15)] == [401] * 15
    assert get_leads(client, "wrong-key").status_code == 401  # only now the first counted failure
    assert get_leads(client, KEY).status_code == 503  # and the right key still gets through


def test_with_the_header_configured_each_real_client_has_its_own_lockout(client, key_set, use_engine, ip_header):
    use_engine(None)
    for i in range(10):
        get_leads(client, f"wrong-{i}", **{"CF-Connecting-IP": "203.0.113.7"})
    assert get_leads(client, "x", **{"CF-Connecting-IP": "203.0.113.7"}).status_code == 429
    assert get_leads(client, KEY, **{"CF-Connecting-IP": "198.51.100.4"}).status_code == 503


def test_the_key_is_never_logged(client, key_set, use_engine, caplog):
    use_engine(None)
    with caplog.at_level(logging.DEBUG):
        get_leads(client, KEY)
        get_leads(client, "wrong-key-guess-123")
    logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert KEY not in logged and "wrong-key-guess-123" not in logged
    assert "testclient" not in logged  # nor the client IP


# ---------- /docs and CORS ----------

def test_docs_list_every_endpoint_and_offer_the_key(client):
    schema = client.get("/openapi.json").json()
    expected = {"/api/health", "/api/health/live", "/api/welcome", "/api/chat", "/api/leads",
                "/api/leads/{lead_id}", "/api/callback", "/api/support-requests"}
    assert set(schema["paths"]) == expected
    [(name, scheme)] = schema["components"]["securitySchemes"].items()
    assert (scheme["type"], scheme["in"], scheme["name"]) == ("apiKey", "header", "X-API-Key")
    for path, method in SALES_ROUTES:
        assert schema["paths"][path][method]["security"] == [{name: []}], path
    assert "security" not in schema["paths"]["/api/chat"]["post"]
    assert "session_id" not in schema["components"]["schemas"]["LeadDetail"]["properties"]
    assert client.get("/docs").status_code == 200


def test_cors_allows_the_api_key_header_for_the_frontend(client):
    origin = get_settings().cors_origin_list[0]
    preflight = {"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "x-api-key"}
    response = client.options("/api/leads", headers={"Origin": origin, **preflight})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "x-api-key" in response.headers["access-control-allow-headers"].lower()
    other = client.options("/api/leads", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in other.headers


def test_cors_exposes_retry_after(client, use_engine):
    use_engine(None)
    response = client.get("/api/health", headers={"Origin": get_settings().cors_origin_list[0]})
    assert "retry-after" in response.headers["access-control-expose-headers"].lower()

