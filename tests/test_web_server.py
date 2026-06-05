"""Tests for the FastAPI dashboard (synthetic mode, no ML deps needed)."""

from fastapi.testclient import TestClient

from web.server import build_app


def test_http_endpoints():
    app = build_app()  # synthetic source
    with TestClient(app) as client:  # 'with' triggers lifespan -> engine.start()
        page = client.get("/")
        assert page.status_code == 200
        assert "FlowCount" in page.text

        stats = client.get("/api/stats").json()
        for key in ("mode", "fps", "tracks", "detections", "counts", "zones", "events"):
            assert key in stats
        assert stats["mode"] == "synthetic"
        assert "in" in stats["counts"] and "out" in stats["counts"]

        frame = client.get("/frame.jpg")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.content[:2] == b"\xff\xd8"  # JPEG magic number

        heat = client.get("/heatmap.jpg")
        assert heat.status_code in (200, 503)


def test_websocket_stats():
    app = build_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stats") as ws:
            data = ws.receive_json()
            assert "fps" in data and "counts" in data
