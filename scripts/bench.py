#!/usr/bin/env python3
"""FlowCount benchmark harness.

Mode A (default, zero ML deps): pushes the synthetic traffic scene through the
real Pipeline and measures end-to-end FPS for several configurations. The
synthetic "detector" is nearly free, so mode A quantifies the tracking +
analytics + annotation overhead — the floor everything else sits on.

Mode B (--model / --models): benchmarks real YOLO inference (requires the
[yolo] extra). Sweeps model x image size x detect-every and reports mean
detector latency plus end-to-end pipeline FPS. Run it on your inference
machine:

    python scripts/bench.py --models yolo11n.pt yolo11s.pt \
        --imgsz 640 416 --detect-every 1 2 4 --input traffic.mp4

Results print as a Markdown table; --out also writes them to a file.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowcount.analytics import AnalyticsManager, LineCrossingCounter  # noqa: E402
from flowcount.pipeline import Pipeline  # noqa: E402
from flowcount.synthetic import HEIGHT, WIDTH, SyntheticTrafficDetector  # noqa: E402
from flowcount.tracker import Tracker  # noqa: E402


class TimedDetector:
    """Wraps any detector and accumulates time spent inside detect()."""

    def __init__(self, inner):
        self.inner = inner
        self.total = 0.0
        self.calls = 0

    def detect(self, frame):
        t0 = time.perf_counter()
        out = self.inner.detect(frame)
        self.total += time.perf_counter() - t0
        self.calls += 1
        return out

    @property
    def mean_ms(self) -> float:
        return (self.total / self.calls * 1000.0) if self.calls else 0.0


def hardware_line() -> str:
    cpu = platform.processor() or platform.machine()
    line = f"{cpu}, Python {platform.python_version()}"
    try:
        import torch

        if torch.cuda.is_available():
            line += f", GPU: {torch.cuda.get_device_name(0)}"
        else:
            line += ", CPU only"
    except ImportError:
        line += ", CPU only"
    return line


def bench_config(
    detector, frame_iter, n_frames: int, *, detect_every: int, annotate: bool, analytics: bool
) -> dict:
    timed = TimedDetector(detector)
    manager = None
    if analytics:
        line = LineCrossingCounter((WIDTH // 2, HEIGHT - 20), (WIDTH // 2, 55), name="bench")
        manager = AnalyticsManager([line])
    tracker = Tracker(min_hits=1, iou_threshold=0.2, max_age=10)
    pipeline = Pipeline(
        timed,
        tracker,
        show_speed=True,
        show_trajectory=True,
        draw_masks=False,
        analytics_manager=manager,
        detect_every=detect_every,
    )

    t0 = time.perf_counter()
    for _ in range(n_frames):
        pipeline.process_frame(next(frame_iter), annotate=annotate)
    elapsed = time.perf_counter() - t0
    return {
        "fps": n_frames / elapsed,
        "detect_ms": timed.mean_ms,
        "detect_calls": timed.calls,
    }


def synthetic_frames():
    det = SyntheticTrafficDetector(loop=True)

    def gen():
        while True:
            frame = det.render_frame()
            det.advance()
            yield frame

    return det, gen()


def file_frames(path: str, max_cache: int = 120):
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_cache:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"could not read frames from {path}")

    def gen():
        i = 0
        while True:
            yield frames[i % len(frames)]
            i += 1

    return gen()


def mode_a(n_frames: int) -> list[str]:
    rows = ["| Configuration | FPS |", "|---|---|"]
    full = {"detect_every": 1, "annotate": True, "analytics": True}
    configs = [
        ("full pipeline (annotate + analytics)", full),
        ("detect every 2nd frame", {**full, "detect_every": 2}),
        ("detect every 4th frame", {**full, "detect_every": 4}),
        ("no annotation", {**full, "annotate": False}),
        ("no annotation, no analytics", {**full, "annotate": False, "analytics": False}),
    ]
    for label, kw in configs:
        det, frames = synthetic_frames()
        r = bench_config(det, frames, n_frames, **kw)
        rows.append(f"| {label} | {r['fps']:.0f} |")
    return rows


def mode_b(
    models: list[str], sizes: list[int], intervals: list[int], input_path: str | None, n_frames: int
) -> list[str]:
    from flowcount.detector import ObjectDetector

    rows = ["| Model | imgsz | detect every | detect ms | pipeline FPS |", "|---|---|---|---|---|"]
    for model in models:
        for imgsz in sizes:
            detector = ObjectDetector(
                model_name=model,
                conf_threshold=0.25,
                img_size=imgsz,
                target_classes={"car", "truck", "bus", "motorcycle", "bicycle", "person"},
            )
            for every in intervals:
                if input_path:
                    frames = file_frames(input_path)
                else:
                    _, frames = synthetic_frames()
                r = bench_config(
                    detector, frames, n_frames, detect_every=every, annotate=True, analytics=True
                )
                rows.append(
                    f"| {model} | {imgsz} | 1/{every} | {r['detect_ms']:.1f} | {r['fps']:.1f} |"
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark the FlowCount pipeline")
    parser.add_argument("--frames", type=int, default=300, help="Frames per configuration")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="YOLO models to sweep (enables mode B; needs ultralytics)",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--detect-every", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--input", default=None, help="Video file for mode B frames")
    parser.add_argument("--out", default=None, help="Also write the table to this file")
    args = parser.parse_args()

    lines = [f"Hardware: {hardware_line()}", ""]
    if args.models:
        lines += mode_b(args.models, args.imgsz, args.detect_every, args.input, args.frames)
    else:
        lines += mode_a(args.frames)

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
