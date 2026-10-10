from app.core.db import init_db


def test_health_ok_with_reachable_db(client, use_engine, test_engine):
    use_engine(test_engine)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok"}


def test_health_degraded_when_db_unreachable(client, use_engine, unreachable_engine):
    use_engine(unreachable_engine)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "db": "error"}


def test_health_degraded_when_db_not_configured(client, use_engine):
    use_engine(None)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "db": "error"}


def test_init_db_survives_unreachable_db(unreachable_engine):
    init_db(unreachable_engine)  # must not raise


def test_request_id_generated_and_echoed(client, use_engine):
    use_engine(None)
    assert client.get("/api/health").headers["X-Request-ID"]
    response = client.get("/api/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
