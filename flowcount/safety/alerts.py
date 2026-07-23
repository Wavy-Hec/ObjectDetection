"""Alert delivery for safety incidents.

The incident detectors decide *that* something is wrong; this module decides
*who hears about it*. It is the sink behind
:attr:`~flowcount.analytics.manager.AnalyticsManager.alert_dispatcher`: the
manager calls ``submit(events, ctx)`` on the frame thread whenever events fire,
and ``close()`` on shutdown.

The one hard rule: **submit() must never block the frame thread.** A webhook to a
paging service can take seconds or hang, and doing that inline would freeze
detection and back the camera up. So ``submit`` does only cheap work — filter by
severity, flatten to a plain dict, enqueue — and a single background worker
delivers to every sink. If delivery falls behind, alerts are dropped with a
warning rather than growing memory without bound: a safety monitor must keep
watching the road even when its alert channel is down.

Two sinks ship here, both dependency-free (stdlib only): :class:`LogSink`
(always safe, great default) and :class:`WebhookSink` (POST JSON). MQTT — the
native language of the Home Assistant / Frigate ecosystem — is a natural third
sink; it is left out only to avoid a hard ``paho-mqtt`` dependency, and a sink
is ~15 lines: implement ``deliver(payload)`` and hand it to the dispatcher.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Ordered so alerts can be filtered with a single ``min_severity`` threshold.
_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def _severity_rank(name: str | None) -> int:
    return _SEVERITY_ORDER.get(name or "info", 0)


def event_to_payload(event: Any, ctx: Any = None) -> dict:
    """Flatten an analytics ``Event`` (+ optional frame ctx) into a JSON-able dict.

    Called on the frame thread, so it must be cheap and must NOT retain the
    frame pixels — only small scalars are copied out.
    """
    data = dict(getattr(event, "data", {}) or {})
    payload = {
        "kind": getattr(event, "kind", "event"),
        "severity": data.get("severity", "info"),
        "class_label": getattr(event, "class_label", None),
        "track_id": getattr(event, "track_id", None),
        "frame_index": getattr(event, "frame_index", None),
        "timestamp": getattr(event, "timestamp", None),
        "summary": event.summary() if hasattr(event, "summary") else str(event),
        "data": data,
    }
    if ctx is not None:
        payload["fps"] = round(float(getattr(ctx, "fps", 0.0) or 0.0), 1)
    return payload


@runtime_checkable
class AlertSink(Protocol):
    """A delivery target. ``deliver`` runs on the dispatcher's worker thread."""

    def deliver(self, payload: dict) -> None: ...

    def close(self) -> None: ...


class LogSink:
    """Emit each alert through the logging system. Always-on, dependency-free.

    The sensible default: it never fails, needs no configuration, and shows up
    wherever the rest of the app already logs (console, journald, a file).
    """

    def __init__(self, level: int = logging.WARNING):
        self.level = level

    def deliver(self, payload: dict) -> None:
        logger.log(
            self.level,
            "ALERT %s [%s] %s",
            payload["kind"],
            payload["severity"],
            payload["summary"],
        )

    def close(self) -> None:
        return None


class WebhookSink:
    """POST each alert as JSON to a URL (stdlib ``urllib``; no new dependency).

    A failed or slow POST is logged and swallowed — an unreachable webhook must
    never take down the monitor. Runs on the worker thread, so a multi-second
    timeout here does not touch the frame path.
    """

    def __init__(self, url: str, *, timeout: float = 5.0, headers: dict | None = None):
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def deliver(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                status = getattr(resp, "status", 200)
                if status >= 400:
                    logger.warning("webhook %s returned HTTP %s", self.url, status)
        except Exception as exc:  # network is best-effort; never propagate
            logger.warning("webhook delivery to %s failed: %s", self.url, exc)

    def close(self) -> None:
        return None


class AlertDispatcher:
    """Fan alerts out to sinks on a background thread, never blocking submit().

    Drop-in for ``AnalyticsManager(alert_dispatcher=...)``.
    """

    def __init__(
        self,
        sinks: list,
        *,
        min_severity: str = "warning",
        max_queue: int = 256,
    ):
        self.sinks = list(sinks)
        self.min_rank = _severity_rank(min_severity)
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._dropped = 0
        self._worker = threading.Thread(
            target=self._run, name="alert-dispatcher", daemon=True
        )
        self._worker.start()

    def submit(self, events, ctx=None) -> None:
        """Enqueue alertable events. Cheap and non-blocking; runs on the frame thread."""
        for event in events:
            payload = event_to_payload(event, ctx)
            if _severity_rank(payload["severity"]) < self.min_rank:
                continue
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                # Alerts are low-volume by design; a full queue means a sink is
                # wedged. Drop rather than block the road-watching thread.
                self._dropped += 1
                if self._dropped % 50 == 1:
                    logger.warning("alert queue full; dropped %d alert(s) so far", self._dropped)

    def _run(self) -> None:
        # Keep draining until asked to stop AND the backlog is delivered, so
        # close() loses nothing already accepted.
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            for sink in self.sinks:
                try:
                    sink.deliver(payload)
                except Exception:
                    logger.exception("alert sink %s failed", type(sink).__name__)
            self._queue.task_done()

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=5.0)
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                logger.exception("closing alert sink %s failed", type(sink).__name__)
