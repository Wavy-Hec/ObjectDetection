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

import cv2
import numpy as np

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

logger = logging.getLogger("flowcount.demo")


def to_gif(frames_bgr: list[np.ndarray], path: Path, fps: int = 12, scale: float = 0.75):
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
    line = LineCrossingCounter(
        (WIDTH // 2, HEIGHT - 20), (WIDTH // 2, 55), name="count", expected_direction="in"
    )
    zone = ZoneManager([Zone("zoneA", [(60, 150), (240, 150), (240, 210), (60, 210)])])
    manager = AnalyticsManager([line, zone])
    heatmap = HeatmapAccumulator(radius=16)

    pipeline = Pipeline(
        detector,
        tracker,
        show_speed=True,
        show_trajectory=True,
        draw_masks=False,
        analytics_manager=manager,
    )

    mp4_path = out_dir / "demo.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (WIDTH, HEIGHT))
    frames: list[np.ndarray] = []

    for i in range(n_frames):
        road = render_road()
        detector.advance()
        result = pipeline.process_frame(road)
        heatmap.update(FrameContext(tracks=result.tracks, frame_index=i, timestamp=0.0, frame=road))
        writer.write(result.frame)
        frames.append(result.frame)

    writer.release()
    logger.info("Wrote %s (%d frames)", mp4_path, len(frames))
    logger.info("Line counts -> in:%d out:%d", line.total_in, line.total_out)

    hm_path = out_dir / "heatmap.jpg"
    heatmap.save(str(hm_path))
    logger.info("Wrote %s", hm_path)

    if make_gif:
        gif_path = out_dir / "demo.gif"
        to_gif(frames, gif_path)
        logger.info("Wrote %s", gif_path)


def run_real(input_path: str, out_dir: Path, n_frames: int, make_gif: bool):
    from flowcount.detector import ObjectDetector
    from flowcount.video_source import create_video_source

    classes = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
    detector = ObjectDetector(model_name="yolov8l.pt", conf_threshold=0.25, target_classes=classes)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    source = create_video_source(input_path)
    props = source.get_properties()
    w, h = props["width"], props["height"]

    line = LineCrossingCounter((w // 2, 0), (w // 2, h), name="count")
    manager = AnalyticsManager([line])
    heatmap = HeatmapAccumulator(radius=max(12, w // 60))
    pipeline = Pipeline(
        detector, tracker, show_speed=True, show_trajectory=True, analytics_manager=manager
    )

    writer = cv2.VideoWriter(
        str(out_dir / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), props["fps"], (w, h)
    )
    frames = []
    i = 0
    while source.is_opened() and i < n_frames:
        ok, frame = source.read()
        if not ok:
            break
        result = pipeline.process_frame(frame)
        heatmap.update(
            FrameContext(tracks=result.tracks, frame_index=i, timestamp=0.0, frame=frame)
        )
        writer.write(result.frame)
        frames.append(result.frame)
        i += 1
    writer.release()
    source.release()
    heatmap.save(str(out_dir / "heatmap.jpg"))
    if make_gif and frames:
        import imageio.v2 as imageio

        rgb = [
            cv2.cvtColor(cv2.resize(f, (640, int(640 * h / w))), cv2.COLOR_BGR2RGB)
            for f in frames[::2]
        ]
        imageio.mimsave(out_dir / "demo.gif", rgb, fps=12, loop=0)
    logger.info(
        "Real demo written to %s (line in:%d out:%d)", out_dir, line.total_in, line.total_out
    )


def main():
    parser = argparse.ArgumentParser(description="Generate FlowCount demo assets")
    parser.add_argument(
        "--input", default=None, help="Real video path (omit for the synthetic demo)"
    )
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
