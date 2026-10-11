import logging

from app.core.db import init_db


def test_health_ok_with_reachable_db(client, use_engine, test_engine):
    use_engine(test_engine)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["db"]) == ("ok", "ok")
    # kb_chunks is a row count, or null if the test database has no kb_chunks table yet
    assert body["kb_chunks"] is None or isinstance(body["kb_chunks"], int)


def test_health_degraded_when_db_unreachable(client, use_engine, unreachable_engine):
    use_engine(unreachable_engine)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "db": "error", "kb_chunks": None}


def test_health_degraded_when_db_not_configured(client, use_engine):
    use_engine(None)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "db": "error", "kb_chunks": None}


def test_init_db_survives_unreachable_db(unreachable_engine):
    init_db(unreachable_engine)  # must not raise


def test_request_id_generated_and_echoed(client, use_engine):
    use_engine(None)
    assert client.get("/api/health").headers["X-Request-ID"]
    response = client.get("/api/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"


def test_live_check_needs_no_database(client, use_engine, unreachable_engine):
    use_engine(unreachable_engine)  # a DB query would fail or be slow; the live check makes none
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_live_check_is_not_logged(client, use_engine, caplog):
    use_engine(None)
    with caplog.at_level(logging.INFO, logger="app.request"):
        client.get("/api/health/live")
        client.get("/api/health")
    logged = [r.path for r in caplog.records if r.getMessage() == "request"]
    assert logged == ["/api/health"]
