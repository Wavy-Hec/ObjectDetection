"""FlowCount web dashboard backend (FastAPI + WebSocket).

A background worker drives the same reusable :class:`~src.pipeline.Pipeline`
used by the CLI; the web layer is pure transport:

  GET  /             -> the dashboard HTML page
  GET  /video        -> MJPEG stream of the annotated frames
  GET  /frame.jpg    -> the latest annotated frame (single JPEG)
  GET  /heatmap.jpg  -> the accumulated activity heatmap (single JPEG)
  GET  /api/stats    -> latest stats as JSON
  WS   /ws/stats     -> stats pushed continuously

By default it runs the *synthetic* traffic scene, so the dashboard works with
no model, camera, or GPU (great for a hosted demo). Pass ``input=`` to run on a
real webcam/video (requires the ``ultralytics`` dependency).

Run:  uvicorn web.server:app        # synthetic demo
      python -m web.server --input traffic.mp4 --preset traffic
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, List, Optional

import cv2
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from src.analytics import (
    AnalyticsManager,
    FrameContext,
    HeatmapAccumulator,
    LineCrossingCounter,
    Zone,
    ZoneManager,
)
from src.logging_config import setup_logging
from src.pipeline import Pipeline
from src.synthetic import HEIGHT, WIDTH, SyntheticTrafficDetector, render_road
from src.tracker import Tracker

logger = logging.getLogger("flowcount.web")

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"


class DashboardEngine:
    """Runs the pipeline on a background thread and holds the latest frame/stats."""

    def __init__(self, pipeline: Pipeline, frame_provider: Callable, *,
                 line_counters: List[LineCrossingCounter],
                 zone_managers: List[ZoneManager],
                 heatmap: Optional[HeatmapAccumulator] = None,
                 mode: str = "synthetic", target_fps: float = 20.0):
        self.pipeline = pipeline
        self.frame_provider = frame_provider
        self.line_counters = line_counters
        self.zone_managers = zone_managers
        self.heatmap = heatmap
        self.mode = mode
        self.period = 1.0 / target_fps if target_fps > 0 else 0.0

        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._stats: dict = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---- worker -------------------------------------------------------------
    def tick(self) -> None:
        frame = self.frame_provider()
        if frame is None:
            return
        result = self.pipeline.process_frame(frame)
        if self.heatmap is not None:
            self.heatmap.update(FrameContext(
                tracks=result.tracks, frame_index=result.stats.frame_index,
                timestamp=time.time(), frame=frame))
        ok, buf = cv2.imencode(".jpg", result.frame)
        stats = self._build_stats(result)
        with self._lock:
            if ok:
                self._jpeg = buf.tobytes()
            self._stats = stats

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
        self.tick()  # produce one frame synchronously so endpoints have data
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---- accessors ----------------------------------------------------------
    def snapshot(self):
        with self._lock:
            return self._jpeg, dict(self._stats)

    def heatmap_jpeg(self) -> Optional[bytes]:
        if self.heatmap is None:
            return None
        img = self.heatmap.render()
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else None

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
            "fps": round(result.stats.fps, 1),
            "frame": result.stats.frame_index,
            "detections": result.stats.num_detections,
            "tracks": result.stats.num_tracks,
            "counts": {
                "in": total_in, "out": total_out,
                "in_by_class": in_by_class, "out_by_class": out_by_class,
            },
            "zones": zones,
            "events": [e.summary() for e in result.events][-8:],
        }


def _build_synthetic_engine() -> DashboardEngine:
    detector = SyntheticTrafficDetector(loop=True)
    tracker = Tracker(min_hits=1, iou_threshold=0.2, max_age=10)
    line = LineCrossingCounter((WIDTH // 2, HEIGHT - 20), (WIDTH // 2, 55), name="count")
    zones = ZoneManager([Zone("zoneA", [(60, 150), (240, 150), (240, 210), (60, 210)])])
    manager = AnalyticsManager([line, zones])
    pipeline = Pipeline(detector, tracker, show_speed=True, show_trajectory=True,
                        draw_masks=False, analytics_manager=manager)

    def provider():
        road = render_road()
        detector.advance()
        return road

    return DashboardEngine(pipeline, provider, line_counters=[line],
                           zone_managers=[zones], heatmap=HeatmapAccumulator(radius=16),
                           mode="synthetic")


def _build_video_engine(input_path: str, model: str, classes) -> DashboardEngine:
    from src.detector import ObjectDetector
    from src.video_source import create_video_source

    detector = ObjectDetector(model_name=model, conf_threshold=0.25, target_classes=classes)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)

    state = {"source": create_video_source(input_path)}
    props = state["source"].get_properties()
    w, h = props["width"], props["height"]
    line = LineCrossingCounter((w // 2, 0), (w // 2, h), name="count")
    manager = AnalyticsManager([line])
    pipeline = Pipeline(detector, tracker, show_speed=True, show_trajectory=True,
                        analytics_manager=manager)

    def provider():
        ok, frame = state["source"].read()
        if not ok:  # loop files so the dashboard runs forever
            state["source"].release()
            state["source"] = create_video_source(input_path)
            ok, frame = state["source"].read()
        return frame if ok else None

    return DashboardEngine(pipeline, provider, line_counters=[line],
                           zone_managers=[], heatmap=HeatmapAccumulator(radius=max(12, w // 60)),
                           mode="video")


def build_app(input: Optional[str] = None, model: str = "yolov8l.pt",
              classes=None) -> FastAPI:
    engine = (_build_video_engine(input, model, classes) if input
              else _build_synthetic_engine())

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
            while True:
                jpeg, _ = engine.snapshot()
                if jpeg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpeg + b"\r\n")
                await asyncio.sleep(0.05)
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

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

    return app


# Default app for `uvicorn web.server:app` (synthetic demo, runs anywhere).
app = build_app()


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the FlowCount dashboard")
    parser.add_argument("--input", default=None, help="Webcam index or video path (default: synthetic)")
    parser.add_argument("--model", default="yolov8l.pt")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    setup_logging("INFO")
    classes = set(args.classes) if args.classes else None
    uvicorn.run(build_app(input=args.input, model=args.model, classes=classes),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
