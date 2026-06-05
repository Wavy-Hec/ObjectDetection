#!/usr/bin/env python3
"""Generate FlowCount demo assets (annotated clip + GIF + heatmap).

Two modes:

  Synthetic (default) — renders a small traffic scene and runs it through the
  REAL Pipeline + analytics. Requires no model, camera, or GPU, so anyone can
  produce the demo with `python scripts/demo.py`.

  Real video — `python scripts/demo.py --input traffic.mp4` runs YOLOv8 +
  tracking + analytics on actual footage (needs the `ultralytics` dependency).

Outputs (into --out-dir, default ./assets):
  demo.gif, demo.mp4, heatmap.jpg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.analytics import (
    AnalyticsManager,
    FrameContext,
    HeatmapAccumulator,
    LineCrossingCounter,
    Zone,
    ZoneManager,
)
from src.detector import Detection
from src.logging_config import setup_logging
from src.pipeline import Pipeline
from src.tracker import Tracker

logger = logging.getLogger("flowcount.demo")

WIDTH, HEIGHT = 640, 360
LANE_Y = (70, 180, 290)

# (start_cx, lane_y, speed_px, class_label, w, h)
VEHICLES = [
    (-40, LANE_Y[0], 6.0, "car", 60, 34),
    (-120, LANE_Y[1], 5.0, "truck", 92, 44),
    (-80, LANE_Y[2], 8.0, "car", 60, 34),
    (-200, LANE_Y[0], 7.0, "motorcycle", 34, 24),
    (-160, LANE_Y[1], 4.5, "bus", 112, 50),
    (-240, LANE_Y[2], 6.0, "car", 60, 34),
    (-300, LANE_Y[0], 9.0, "car", 58, 32),
    (-280, LANE_Y[2], 7.0, "motorcycle", 32, 22),
]


def render_road() -> np.ndarray:
    """Draw a fresh asphalt background with dashed lane separators."""
    frame = np.full((HEIGHT, WIDTH, 3), 55, dtype=np.uint8)
    for y in (125, 235):  # lane separators
        for x in range(0, WIDTH, 40):
            cv2.line(frame, (x, y), (x + 22, y), (200, 200, 200), 2)
    cv2.rectangle(frame, (0, 18), (WIDTH, 40), (35, 35, 35), -1)
    cv2.putText(frame, "FlowCount - Traffic Analytics (synthetic demo)",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return frame


class SyntheticTrafficDetector:
    """Returns Detection boxes for vehicles that advance every frame."""

    def __init__(self):
        self.vehicles = [list(v) for v in VEHICLES]  # mutable copies

    def detect(self, frame) -> List[Detection]:
        dets = []
        for cx, cy, speed, cls, w, h in self.vehicles:
            if -w < cx < WIDTH + w:
                dets.append(Detection(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], cls, 0.9))
        return dets

    def advance(self):
        for v in self.vehicles:
            v[0] += v[2]  # cx += speed


def to_gif(frames_bgr: List[np.ndarray], path: Path, fps: int = 12, scale: float = 0.75):
    import imageio.v2 as imageio
    rgb = []
    for f in frames_bgr:
        small = cv2.resize(f, (int(WIDTH * scale), int(HEIGHT * scale)))
        rgb.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, rgb, fps=fps, loop=0)


def run_synthetic(out_dir: Path, n_frames: int = 90, make_gif: bool = True):
    detector = SyntheticTrafficDetector()
    tracker = Tracker(min_hits=1, iou_threshold=0.2, max_age=10)

    # Endpoints ordered so left->right travel counts as "in".
    line = LineCrossingCounter((WIDTH // 2, HEIGHT - 20), (WIDTH // 2, 55), name="count")
    zone = ZoneManager([Zone("zoneA", [(60, 150), (240, 150), (240, 210), (60, 210)])])
    manager = AnalyticsManager([line, zone])
    heatmap = HeatmapAccumulator(radius=16)

    pipeline = Pipeline(detector, tracker, show_speed=True, show_trajectory=True,
                        draw_masks=False, analytics_manager=manager)

    mp4_path = out_dir / "demo.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             15, (WIDTH, HEIGHT))
    frames: List[np.ndarray] = []

    for i in range(n_frames):
        road = render_road()
        detector.advance()
        result = pipeline.process_frame(road)
        heatmap.update(FrameContext(tracks=result.tracks, frame_index=i,
                                    timestamp=0.0, frame=road))
        writer.write(result.frame)
        frames.append(result.frame)

    writer.release()
    logger.info("Wrote %s (%d frames)", mp4_path, len(frames))
    logger.info("Line counts -> in:%d out:%d", line.total_in, line.total_out)

    hm_path = out_dir / "heatmap.jpg"
    heatmap.save(str(out_dir / "heatmap"))  # writes <prefix>_heatmap.jpg
    legacy = out_dir / "heatmap_heatmap.jpg"
    if legacy.exists():
        legacy.replace(hm_path)
    logger.info("Wrote %s", hm_path)

    if make_gif:
        gif_path = out_dir / "demo.gif"
        to_gif(frames, gif_path)
        logger.info("Wrote %s", gif_path)


def run_real(input_path: str, out_dir: Path, n_frames: int, make_gif: bool):
    from src.detector import ObjectDetector
    from src.video_source import create_video_source

    classes = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
    detector = ObjectDetector(model_name="yolov8l.pt", conf_threshold=0.25,
                              target_classes=classes)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    source = create_video_source(input_path)
    props = source.get_properties()
    w, h = props["width"], props["height"]

    line = LineCrossingCounter((w // 2, 0), (w // 2, h), name="count")
    manager = AnalyticsManager([line])
    heatmap = HeatmapAccumulator(radius=max(12, w // 60))
    pipeline = Pipeline(detector, tracker, show_speed=True, show_trajectory=True,
                        analytics_manager=manager)

    writer = cv2.VideoWriter(str(out_dir / "demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), props["fps"], (w, h))
    frames = []
    i = 0
    while source.is_opened() and i < n_frames:
        ok, frame = source.read()
        if not ok:
            break
        result = pipeline.process_frame(frame)
        heatmap.update(FrameContext(tracks=result.tracks, frame_index=i,
                                    timestamp=0.0, frame=frame))
        writer.write(result.frame)
        frames.append(result.frame)
        i += 1
    writer.release()
    source.release()
    heatmap.save(str(out_dir / "heatmap"))
    if make_gif and frames:
        import imageio.v2 as imageio
        rgb = [cv2.cvtColor(cv2.resize(f, (640, int(640 * h / w))), cv2.COLOR_BGR2RGB)
               for f in frames[::2]]
        imageio.mimsave(out_dir / "demo.gif", rgb, fps=12, loop=0)
    logger.info("Real demo written to %s (line in:%d out:%d)",
                out_dir, line.total_in, line.total_out)


def main():
    parser = argparse.ArgumentParser(description="Generate FlowCount demo assets")
    parser.add_argument("--input", default=None,
                        help="Real video path (omit for the synthetic demo)")
    parser.add_argument("--out-dir", default="assets", help="Output directory")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF output")
    args = parser.parse_args()

    setup_logging("INFO")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        run_real(args.input, out_dir, args.frames, not args.no_gif)
    else:
        run_synthetic(out_dir, args.frames, not args.no_gif)


if __name__ == "__main__":
    main()
