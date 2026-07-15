"""Tests for the FastAPI dashboard (synthetic mode, no ML deps needed)."""

import time

from fastapi.testclient import TestClient

from flowcount.web.server import build_app


def _wait_for_frame(client, timeout=5.0):
    """The engine warms up asynchronously; poll until the first frame lands."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get("/frame.jpg")
        if resp.status_code == 200:
            return resp
        time.sleep(0.02)
    raise AssertionError("engine produced no frame within %.1fs" % timeout)


def test_http_endpoints():
    app = build_app()  # synthetic source
    with TestClient(app) as client:  # 'with' triggers lifespan -> engine.start()
        page = client.get("/")
        assert page.status_code == 200
        assert "FlowCount" in page.text

        frame = _wait_for_frame(client)
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.content[:2] == b"\xff\xd8"  # JPEG magic number

        stats = client.get("/api/stats").json()
        for key in ("mode", "fps", "tracks", "detections", "counts", "zones",
                    "events", "status", "detect_every"):
            assert key in stats
        assert stats["mode"] == "synthetic"
        assert stats["status"] == "ok"
        assert "in" in stats["counts"] and "out" in stats["counts"]

        heat = client.get("/heatmap.jpg")
        assert heat.status_code in (200, 503)


def test_stats_before_first_frame_report_warming():
    app = build_app()
    # No lifespan: the engine never starts, mimicking the pre-first-frame gap.
    client = TestClient(app)
    stats = client.get("/api/stats").json()
    assert stats["status"] == "warming"
    assert stats["counts"] == {"in": 0, "out": 0, "in_by_class": {}, "out_by_class": {}}
    assert client.get("/frame.jpg").status_code == 503


def test_healthz_reports_engine_state():
    app = build_app()
    with TestClient(app) as client:
        _wait_for_frame(client)
        health = client.get("/healthz")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["engine_alive"] is True
        assert body["last_tick_age"] is not None


def test_websocket_stats():
    app = build_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stats") as ws:
            data = ws.receive_json()
            assert "fps" in data and "counts" in data


def test_events_accumulate_across_ticks():
    """Events survive between stats reads instead of being overwritten per tick."""
    app = build_app()
    with TestClient(app) as client:
        _wait_for_frame(client)
        engine = app.state.engine
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if engine.snapshot()[1].get("events"):
                break
            time.sleep(0.05)
        events_then = engine.snapshot()[1]["events"]
        assert events_then, "synthetic scene produced no crossing events in 10s"
        # Many ticks later, previously seen events are still reported.
        time.sleep(0.3)
        events_now = engine.snapshot()[1]["events"]
        assert events_then[0] in events_now or len(events_now) == 12
