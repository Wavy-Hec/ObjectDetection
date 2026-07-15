"""FlowCount web dashboard backend (FastAPI + WebSocket).

A background worker drives the same reusable :class:`~flowcount.pipeline.Pipeline`
used by the CLI; the web layer is pure transport:

  GET  /             -> the dashboard HTML page
  GET  /video        -> MJPEG stream of the annotated frames
  GET  /frame.jpg    -> the latest annotated frame (single JPEG)
  GET  /heatmap.jpg  -> the accumulated activity heatmap (single JPEG)
  GET  /api/stats    -> latest stats as JSON
  GET  /healthz      -> engine liveness (for Docker / hosted deploys)
  WS   /ws/stats     -> stats pushed continuously

By default it runs the *synthetic* traffic scene, so the dashboard works with
no model, camera, or GPU (great for a hosted demo). Pass ``source=`` to run on
a real webcam / video / RTSP stream (requires the ``ultralytics`` dependency).

Run:  uvicorn flowcount.web.server:app        # synthetic demo
      python -m flowcount.web.server --input traffic.mp4
      python -m flowcount.web.server --input 0 --live           # webcam
      python -m flowcount.web.server --input rtsp://cam/stream  # IP camera
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from flowcount.analytics import (
    AnalyticsManager,
    FrameContext,
    HeatmapAccumulator,
    LineCrossingCounter,
    Zone,
    ZoneManager,
)
from flowcount.logging_config import setup_logging
from flowcount.pipeline import Pipeline
from flowcount.synthetic import HEIGHT, WIDTH, SyntheticTrafficDetector, render_road
from flowcount.tracker import Tracker

logger = logging.getLogger("flowcount.web")

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"

# Keep this many recent analytics events for the dashboard panel. Events are
# accumulated across ticks (the worker runs ~4x faster than the WS push rate,
# so per-tick snapshots would silently drop most of them).
EVENT_HISTORY = 50


class DashboardEngine:
    """Runs the pipeline on a background thread and holds the latest frame/stats."""

    def __init__(
        self,
        pipeline: Pipeline,
        frame_provider: Callable,
        *,
        line_counters: list[LineCrossingCounter],
        zone_managers: list[ZoneManager],
        heatmap: HeatmapAccumulator | None = None,
        mode: str = "synthetic",
        target_fps: float = 20.0,
        cleanup: Callable | None = None,
    ):
        self.pipeline = pipeline
        self.frame_provider = frame_provider
        self.line_counters = line_counters
        self.zone_managers = zone_managers
        self.heatmap = heatmap
        self.mode = mode
        self.period = 1.0 / target_fps if target_fps > 0 else 0.0
        self._cleanup = cleanup  # e.g. release the video source on stop

        self._lock = threading.Lock()
        # Separate lock for the heatmap: its per-track accumulation is the
        # heaviest per-tick work, and holding the main lock through it would
        # stall every endpoint (they acquire _lock on the event loop).
        self._heatmap_lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stats: dict = {}
        self._seq = 0  # bumps once per fresh frame
        self._events: deque = deque(maxlen=EVENT_HISTORY)
        self._last_tick: float | None = None
        self._started_at: float | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    # ---- worker -------------------------------------------------------------
    def tick(self) -> None:
        frame = self.frame_provider()
        if frame is None:
            return
        result = self.pipeline.process_frame(frame)
        if self.heatmap is not None:
            # Guards against heatmap_jpeg() rendering mid-mutation without
            # blocking the main stats/frame lock.
            with self._heatmap_lock:
                self.heatmap.update(
                    FrameContext(
                        tracks=result.tracks,
                        frame_index=result.stats.frame_index,
                        timestamp=time.time(),
                        frame=frame,
                    )
                )
        ok, buf = cv2.imencode(".jpg", result.frame)
        with self._lock:
            self._events.extend(e.summary() for e in result.events)
            stats = self._build_stats(result)
            if ok:
                self._jpeg = buf.tobytes()
                self._seq += 1
            self._stats = stats
            self._last_tick = time.time()

    def _loop(self) -> None:
        while self._running:
            start = time.time()
            try:
                self.tick()
            except Exception:  # keep the dashboard alive on a bad frame
                logger.exception("pipeline tick failed")
            elapsed = time.time() - start
            if self.period > elapsed:
                time.sleep(self.period - elapsed)

    def start(self) -> None:
        # Warm up on the worker thread: with a real model the first tick can
        # take tens of seconds (lazy ultralytics import + weight load inside
        # ObjectDetector.detect), and blocking here would stall server
        # startup. Endpoints serve a "warming" state until the first frame.
        self._running = True
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cleanup is not None:
            try:
                self._cleanup()
            except Exception:
                logger.exception("engine cleanup failed")

    # ---- accessors ----------------------------------------------------------
    def snapshot(self):
        """Latest (jpeg, stats); stats fall back to a zeroed warming skeleton."""
        with self._lock:
            stats = dict(self._stats) if self._stats else self._warming_stats()
            return self._jpeg, stats

    def latest(self):
        """Latest (seq, jpeg) for consumers that must not resend stale frames."""
        with self._lock:
            return self._seq, self._jpeg

    # A real model can take this long to import + download + load; past it,
    # an engine that has never completed a tick is failing, not warming.
    WARMUP_GRACE_S = 120.0

    def health(self) -> dict:
        with self._lock:
            last_tick = self._last_tick
        alive = self._thread is not None and self._thread.is_alive()
        if last_tick is None:
            warming = (
                alive
                and self._started_at is not None
                and time.time() - self._started_at < self.WARMUP_GRACE_S
            )
            status = "warming" if warming else "dead"
        else:
            age = time.time() - last_tick
            status = "ok" if age < 10.0 else "stale"
        return {
            "status": status,
            "mode": self.mode,
            "engine_alive": alive,
            "last_tick_age": None if last_tick is None else round(time.time() - last_tick, 2),
        }

    def heatmap_jpeg(self) -> bytes | None:
        if self.heatmap is None:
            return None
        with self._heatmap_lock:  # don't render while the worker mutates it
            img = self.heatmap.render()
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else None

    def _warming_stats(self) -> dict:
        return {
            "mode": self.mode,
            "status": "warming",
            "fps": 0,
            "frame": 0,
            "detections": 0,
            "tracks": 0,
            "detect_every": getattr(self.pipeline, "detect_every", 1),
            "counts": {"in": 0, "out": 0, "in_by_class": {}, "out_by_class": {}},
            "zones": {},
            "events": [],
        }

    def _build_stats(self, result) -> dict:
        total_in = total_out = 0
        in_by_class: dict = {}
        out_by_class: dict = {}
        for lc in self.line_counters:
            total_in += lc.total_in
            total_out += lc.total_out
            for k, v in lc.counts["in"].items():
                in_by_class[k] = in_by_class.get(k, 0) + v
            for k, v in lc.counts["out"].items():
                out_by_class[k] = out_by_class.get(k, 0) + v

        zones: dict = {}
        for zm in self.zone_managers:
            for z in zm.zones:
                zones[z.name] = zm.occupancy(z.name)

        return {
            "mode": self.mode,
            "status": "ok",
            "fps": round(result.stats.fps, 1),
            "frame": result.stats.frame_index,
            "detections": result.stats.num_detections,
            "tracks": result.stats.num_tracks,
            "detect_every": getattr(self.pipeline, "detect_every", 1),
            "counts": {
                "in": total_in,
                "out": total_out,
                "in_by_class": in_by_class,
                "out_by_class": out_by_class,
            },
            "zones": zones,
            # Chronological; callers show newest first. Accumulated across
            # ticks so slower consumers (5 Hz WS push) never miss crossings.
            "events": list(self._events)[-12:],
        }


def _build_synthetic_engine() -> DashboardEngine:
    detector = SyntheticTrafficDetector(loop=True)
    tracker = Tracker(min_hits=1, iou_threshold=0.2, max_age=10)
    # Left->right travel is "in" for this line; the synthetic scene includes a
    # wrong-way driver, so expected_direction="in" makes WRONG WAY alerts fire.
    line = LineCrossingCounter(
        (WIDTH // 2, HEIGHT - 20), (WIDTH // 2, 55), name="count", expected_direction="in"
    )
    zones = ZoneManager([Zone("zoneA", [(60, 150), (240, 150), (240, 210), (60, 210)])])
    manager = AnalyticsManager([line, zones])
    pipeline = Pipeline(
        detector,
        tracker,
        show_speed=True,
        show_trajectory=True,
        draw_masks=False,
        analytics_manager=manager,
    )

    def provider():
        road = render_road()
        detector.advance()
        return road

    return DashboardEngine(
        pipeline,
        provider,
        line_counters=[line],
        zone_managers=[zones],
        heatmap=HeatmapAccumulator(radius=16),
        mode="synthetic",
    )


def _build_video_engine(
    source_spec: str, model: str, classes, detect_every: int = 1, live: bool = False
) -> DashboardEngine:
    from flowcount.detector import ObjectDetector
    from flowcount.video_source import LatestFrameGrabber, create_video_source

    detector = ObjectDetector(model_name=model, conf_threshold=0.25, target_classes=classes)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)

    state = {"source": create_video_source(source_spec)}
    props = state["source"].get_properties()
    w, h = props["width"], props["height"]
    is_unbounded = props["total_frames"] == -1  # webcam or network stream

    if live and is_unbounded:
        # Capture on a dedicated thread and drop stale frames so inference
        # latency never accumulates behind the camera.
        state["source"] = LatestFrameGrabber(state["source"])

    line = LineCrossingCounter((w // 2, 0), (w // 2, h), name="count")
    manager = AnalyticsManager([line])
    pipeline = Pipeline(
        detector,
        tracker,
        show_speed=True,
        show_trajectory=True,
        analytics_manager=manager,
        detect_every=detect_every,
    )

    def provider():
        ok, frame = state["source"].read()
        if not ok:
            if is_unbounded:
                # Live sources retry internally (StreamSource reconnects, the
                # grabber keeps polling through outages); no frame this tick
                # just skips the tick rather than tearing anything down.
                return None
            # Loop video files so the dashboard runs forever.
            state["source"].release()
            state["source"] = create_video_source(source_spec)
            ok, frame = state["source"].read()
        return frame if ok else None

    target_fps = min(float(props["fps"] or 20), 30.0)
    return DashboardEngine(
        pipeline,
        provider,
        line_counters=[line],
        zone_managers=[],
        heatmap=HeatmapAccumulator(radius=max(12, w // 60)),
        mode="live" if is_unbounded else "video",
        target_fps=target_fps,
        cleanup=lambda: state["source"].release(),
    )


def build_app(
    source: str | None = None,
    model: str = "yolo11n.pt",
    classes=None,
    detect_every: int = 1,
    live: bool = False,
    input: str | None = None,
) -> FastAPI:
    """Create the dashboard app. With no ``source``, runs the synthetic scene.

    ``input`` is a deprecated alias for ``source``.
    """
    source = source if source is not None else input
    engine = (
        _build_video_engine(source, model, classes, detect_every, live)
        if source
        else _build_synthetic_engine()
    )

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        logger.info("Dashboard engine started (mode=%s)", engine.mode)
        try:
            yield
        finally:
            engine.stop()
            logger.info("Dashboard engine stopped")

    app = FastAPI(title="FlowCount Dashboard", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML.read_text()

    @app.get("/api/stats")
    def api_stats():
        _, stats = engine.snapshot()
        return stats

    @app.get("/healthz")
    def healthz():
        health = engine.health()
        code = 200 if health["status"] in ("ok", "warming") else 503
        return Response(content=json.dumps(health), status_code=code, media_type="application/json")

    @app.get("/frame.jpg")
    def frame_jpg():
        jpeg, _ = engine.snapshot()
        if not jpeg:
            return Response(status_code=503)
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/heatmap.jpg")
    def heatmap_jpg():
        jpeg = engine.heatmap_jpeg()
        if not jpeg:
            return Response(status_code=503)
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/video")
    def video():
        async def gen():
            # Sleep roughly half the engine period between polls, and only
            # send when a NEW frame exists — never duplicate bytes on the wire.
            poll = max(0.01, engine.period / 2 if engine.period else 0.02)
            last_seq = -1
            while True:
                seq, jpeg = engine.latest()
                if jpeg and seq != last_seq:
                    last_seq = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(poll)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/stats")
    async def ws_stats(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                _, stats = engine.snapshot()
                await ws.send_json(stats)
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            pass
        except Exception:  # abrupt client drops raise transport errors, not WSD
            logger.debug("websocket client dropped", exc_info=True)

    return app


# Default app for `uvicorn flowcount.web.server:app` (synthetic demo, runs anywhere).
# Constructing the engine is cheap and starts no threads; work begins only
# when the ASGI lifespan runs.
app = build_app()


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the FlowCount dashboard")
    parser.add_argument(
        "--input",
        default=None,
        help="Webcam index, video path, or stream URL (rtsp://, http://). Default: synthetic scene",
    )
    parser.add_argument(
        "--model", default="yolo11n.pt", help="YOLO model (yolo11n.pt ... yolo11x.pt, yolov8*)"
    )
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument(
        "--detect-every",
        type=int,
        default=None,
        metavar="N",
        help="Detect on every Nth frame; tracks coast between (defaults to 3 with --live, else 1)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Low-latency mode for webcam/stream input: background capture, stale frames dropped",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    setup_logging("INFO")
    classes = set(args.classes) if args.classes else None
    detect_every = args.detect_every if args.detect_every is not None else (3 if args.live else 1)
    uvicorn.run(
        build_app(
            source=args.input,
            model=args.model,
            classes=classes,
            detect_every=detect_every,
            live=args.live,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
