"""Tests for the safety alert-delivery layer.

Covers the two shipped sinks, the severity filter, non-blocking submit, and the
close()-drains-the-backlog guarantee. No network or new dependencies: the
webhook is exercised by monkeypatching urllib.
"""

from __future__ import annotations

import json
import logging

from flowcount.analytics.base import Event
from flowcount.safety.alerts import (
    AlertDispatcher,
    LogSink,
    WebhookSink,
    event_to_payload,
)


def _event(kind="stalled", severity="critical", track_id=7):
    return Event(
        kind=kind,
        track_id=track_id,
        class_label="car",
        frame_index=42,
        timestamp=1.5,
        data={"severity": severity, "zone": "crossing"},
    )


class RecordingSink:
    """Captures delivered payloads so tests can assert on them."""

    def __init__(self):
        self.payloads = []
        self.closed = False

    def deliver(self, payload):
        self.payloads.append(payload)

    def close(self):
        self.closed = True


def test_event_to_payload_flattens_severity_and_summary():
    payload = event_to_payload(_event())
    assert payload["kind"] == "stalled"
    assert payload["severity"] == "critical"
    assert payload["track_id"] == 7
    assert "stalled" in payload["summary"]
    # Must be JSON-serializable for the webhook path.
    json.dumps(payload)


def test_event_without_severity_defaults_to_info():
    e = Event(kind="line_cross", track_id=1, class_label="car", frame_index=1, timestamp=0.0)
    assert event_to_payload(e)["severity"] == "info"


def test_dispatcher_delivers_and_close_drains_backlog():
    sink = RecordingSink()
    dispatcher = AlertDispatcher([sink], min_severity="warning")
    dispatcher.submit([_event(severity="critical")])
    dispatcher.submit([_event(kind="intrusion", severity="warning")])
    # close() must block until the worker has drained everything already accepted.
    dispatcher.close()
    kinds = {p["kind"] for p in sink.payloads}
    assert kinds == {"stalled", "intrusion"}
    assert sink.closed is True


def test_severity_filter_drops_below_threshold():
    sink = RecordingSink()
    dispatcher = AlertDispatcher([sink], min_severity="warning")
    dispatcher.submit(
        [
            _event(kind="congestion", severity="info"),  # filtered out
            _event(kind="stalled", severity="critical"),  # kept
        ]
    )
    dispatcher.close()
    assert [p["kind"] for p in sink.payloads] == ["stalled"]


def test_submit_is_nonblocking_and_survives_a_failing_sink():
    class ExplodingSink:
        def deliver(self, payload):
            raise RuntimeError("sink is down")

        def close(self):
            pass

    good = RecordingSink()
    dispatcher = AlertDispatcher([ExplodingSink(), good], min_severity="warning")
    # submit must return immediately and must not raise even though a sink will.
    dispatcher.submit([_event()])
    dispatcher.close()
    # The healthy sink still received the alert despite its neighbour throwing.
    assert len(good.payloads) == 1


def test_log_sink_emits_record(caplog):
    sink = LogSink()
    with caplog.at_level(logging.WARNING, logger="flowcount.safety.alerts"):
        sink.deliver(event_to_payload(_event()))
    assert any("ALERT" in r.message or "ALERT" in r.getMessage() for r in caplog.records)


def test_webhook_sink_posts_json(monkeypatch):
    captured = {}

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["content_type"] = req.headers.get("Content-type")
        return FakeResp()

    monkeypatch.setattr("flowcount.safety.alerts.urllib.request.urlopen", fake_urlopen)

    WebhookSink("https://example.test/alert").deliver(event_to_payload(_event()))
    assert captured["url"] == "https://example.test/alert"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    assert captured["body"]["kind"] == "stalled"


def test_webhook_sink_swallows_network_errors(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("flowcount.safety.alerts.urllib.request.urlopen", boom)
    # A dead endpoint must never propagate into the pipeline.
    WebhookSink("https://down.test/alert").deliver(event_to_payload(_event()))
