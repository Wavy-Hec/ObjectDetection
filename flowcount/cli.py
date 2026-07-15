#!/usr/bin/env python3
"""
FlowCount CLI — real-time tracking on webcam, video files, or streams.

Usage (installed as the `flowcount` console script; `python main.py` also works):
    # Live webcam, low latency
    flowcount --input 0 --live

    # Video file
    flowcount --input video.mp4

    # IP camera, with recording
    flowcount --input rtsp://camera/stream --live --output tracked.mp4

Defaults come from config.yaml (override with --config); any CLI flag you pass
takes precedence over the config file.

Controls:
    q / ESC - Quit
    p       - Pause/Resume
    s       - Save current frame
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from flowcount.analytics import (
    AnalyticsManager,
    CSVExporter,
    EventClipRecorder,
    HeatmapAccumulator,
    LineCrossingCounter,
    SQLiteExporter,
    Zone,
    ZoneManager,
)
from flowcount.config import load_config
from flowcount.detector import ObjectDetector
from flowcount.logging_config import setup_logging
from flowcount.pipeline import Pipeline
from flowcount.tracker import Tracker
from flowcount.video_source import LatestFrameGrabber, VideoSourceError, create_video_source

logger = logging.getLogger("objectdetection.main")


def add_overlays(
    frame: np.ndarray, fps: float, num_detections: int, num_tracks: int, is_paused: bool = False
):
    """
    Add FPS and statistics overlays to frame.

    Args:
        frame: Frame to overlay on
        fps: Current FPS
        num_detections: Number of detections
        num_tracks: Number of tracks
        is_paused: Whether system is paused
    """
    # FPS counter (top-left, green)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Detection count
    det_text = f"Detections: {num_detections}"
    cv2.putText(frame, det_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Track count
    track_text = f"Tracks: {num_tracks}"
    cv2.putText(frame, track_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Paused indicator
    if is_paused:
        pause_text = "PAUSED"
        cv2.putText(frame, pause_text, (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


def save_frame(frame: np.ndarray, prefix: str = "frame") -> str:
    """
    Save frame to file.

    Args:
        frame: Frame to save
        prefix: Filename prefix

    Returns:
        Saved filename
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename


def print_controls():
    """Print keyboard controls."""
    print("\n" + "=" * 60)
    print("  KEYBOARD CONTROLS")
    print("=" * 60)
    print("  q / ESC  - Quit")
    print("  p        - Pause/Resume")
    print("  s        - Save current frame")
    print("=" * 60 + "\n")


# Predefined class presets for common use-cases.
# Mixes COCO classes (detected by standard YOLOv8) and custom classes
# (detected via YOLO-World open-vocabulary).
CLASS_PRESETS = {
    "lab": [
        # People & safety
        "person",
        "tie",
        "glasses",
        "glove",
        "hat",
        "badge",
        "lanyard",
        "watch",
        "shoe",
        "wallet",
        # Hand tools
        "screwdriver",
        "hammer",
        "wrench",
        "pliers",
        "drill",
        "saw",
        "tape",
        "ruler",
        "key",
        # Stationery
        "pen",
        "pencil",
        "marker",
        "scissors",
        "stapler",
        "eraser",
        "notebook",
        "book",
        # Electronics
        "laptop",
        "cell phone",
        "keyboard",
        "mouse",
        "remote",
        "tv",
        "headphones",
        "cable",
        "wire",
        # Lab / desk items
        "bottle",
        "cup",
        "mug",
        "bowl",
        "vase",
        "clock",
        "backpack",
        "handbag",
        "umbrella",
        "chair",
        # Vehicles (if window/outdoor)
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        # Animals
        "dog",
        "cat",
        "bird",
    ],
    "office": [
        # People & accessories
        "person",
        "tie",
        "glasses",
        "watch",
        "hat",
        "badge",
        "lanyard",
        # Stationery
        "pen",
        "pencil",
        "marker",
        "scissors",
        "stapler",
        "ruler",
        "eraser",
        # Electronics
        "laptop",
        "cell phone",
        "keyboard",
        "mouse",
        "remote",
        "tv",
        "headphones",
        "cable",
        # Desk items
        "book",
        "notebook",
        "cup",
        "mug",
        "bottle",
        "backpack",
        "handbag",
        "chair",
        "clock",
    ],
    "tools": [
        "person",
        "screwdriver",
        "hammer",
        "wrench",
        "pliers",
        "drill",
        "saw",
        "tape",
        "ruler",
        "knife",
        "scissors",
        "key",
        "cable",
        "wire",
        "glove",
        "glasses",
        "backpack",
    ],
    "traffic": [
        # COCO-only vehicle/road classes -> standard YOLOv8 (no YOLO-World needed)
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "person",
    ],
    "general": [
        # People & accessories
        "person",
        "tie",
        "glasses",
        "watch",
        "hat",
        "glove",
        "shoe",
        "badge",
        "lanyard",
        "wallet",
        # Stationery & tools
        "pen",
        "pencil",
        "marker",
        "scissors",
        "stapler",
        "ruler",
        "screwdriver",
        "hammer",
        "wrench",
        "pliers",
        "tape",
        # Electronics
        "laptop",
        "cell phone",
        "keyboard",
        "mouse",
        "remote",
        "tv",
        "headphones",
        "cable",
        # Everyday items
        "book",
        "notebook",
        "cup",
        "mug",
        "bottle",
        "backpack",
        "handbag",
        "umbrella",
        "chair",
        "clock",
        "vase",
        "key",
        # Vehicles (if visible)
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        # Animals
        "dog",
        "cat",
        "bird",
    ],
}


def parse_arguments():
    """Parse command-line arguments.

    Config-backed options default to ``None`` so we can tell whether the user
    supplied them; when omitted, the value falls back to config.yaml.
    """
    parser = argparse.ArgumentParser(
        description="Real-time object tracking with webcam or video files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input source: 0 for webcam, a video file path, or a stream URL (rtsp://, http://)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output video file path (optional)")
    parser.add_argument(
        "--config", default=None, help="Path to YAML config (default: ./config.yaml)"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Detection (None -> fall back to config.yaml)
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="YOLO model to use (e.g. yolo11n.pt, yolo11s.pt, "
        "yolov8n.pt ... yolov8x.pt, or "
        "yolov8l-worldv2.pt for open-vocabulary)",
    )
    parser.add_argument(
        "--confidence", type=float, default=None, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size (lower = faster; default from config)",
    )
    parser.add_argument(
        "--device", default=None, help="Inference device: auto, cuda, cuda:N, or cpu"
    )
    parser.add_argument(
        "--segmentation", action="store_true", help="Enable segmentation masks (off by default)"
    )

    # Live / performance
    parser.add_argument(
        "--detect-every",
        type=int,
        default=None,
        metavar="N",
        help="Run detection on every Nth frame only; tracks coast "
        "on Kalman prediction in between (big FPS lever on CPU)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Low-latency live mode for webcam/stream sources: "
        "capture on a background thread and drop stale "
        "frames; defaults --detect-every to 3",
    )

    # Tracking (None -> fall back to config.yaml)
    parser.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="Maximum frames to keep track alive without detection",
    )
    parser.add_argument("--min-hits", type=int, default=None, help="Minimum hits to confirm track")
    parser.add_argument(
        "--iou-threshold", type=float, default=None, help="IOU threshold for tracking"
    )

    # Features
    parser.add_argument(
        "--trajectories",
        action="store_true",
        help="Enable trajectory visualization (off by default)",
    )
    parser.add_argument("--no-speed", action="store_true", help="Disable speed overlay")

    # Display
    parser.add_argument(
        "--no-display", action="store_true", help="Disable display window (headless mode)"
    )

    # Analytics & events (Phase 2)
    parser.add_argument(
        "--count-line",
        action="append",
        metavar="x1,y1,x2,y2",
        help="Add a line-crossing counter (repeatable). Example: --count-line 0,300,640,300",
    )
    parser.add_argument(
        "--expect-direction",
        choices=["in", "out"],
        default=None,
        help="Expected crossing direction for all count lines; crossings the "
        "other way raise a WRONG WAY alert (red flash + wrong_way event)",
    )
    parser.add_argument(
        "--zone",
        action="append",
        metavar="x1,y1,x2,y2,...",
        help="Add a polygon zone for occupancy/dwell (repeatable, "
        ">=3 points). Example: --zone 100,100,300,100,300,300,100,300",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Emit a dwell event when an object stays in a zone this long",
    )
    parser.add_argument(
        "--heatmap", action="store_true", help="Accumulate an activity heatmap (saved on exit)"
    )
    parser.add_argument(
        "--export-csv", default=None, metavar="PATH", help="Export per-frame tracks + events to CSV"
    )
    parser.add_argument(
        "--export-db",
        default=None,
        metavar="PATH",
        help="Export tracks + events to a SQLite database",
    )
    parser.add_argument(
        "--record-events",
        default=None,
        metavar="DIR",
        help="Record event-triggered clips (pre/post roll) into DIR",
    )

    # Class filter
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Only detect these classes (e.g. --classes person "
        "tie pen screwdriver). If any class is not in COCO, "
        "YOLO-World is used automatically. Default: all classes",
    )
    parser.add_argument(
        "--preset",
        choices=list(CLASS_PRESETS.keys()) + ["none"],
        default=None,
        help="Use a predefined class set: "
        "traffic (vehicles + people, COCO-only - the default when neither "
        "--classes nor config target_classes is set), "
        "lab (broad lab/workshop detection), "
        "office (desk/stationery/electronics), "
        "tools (hand tools/hardware), "
        "general (broad everyday mix). "
        "Use --preset none to detect all COCO classes. "
        "Overridden by --classes if both are given.",
    )

    return parser.parse_args()


def resolve_target_classes(args, cfg):
    """Resolve which classes to detect from CLI args then config.

    Priority: --classes > explicit --preset > config target_classes > the
    'traffic' preset. --preset none detects all COCO classes.
    """
    if args.classes:
        target = set(args.classes)
        logger.info("Filtering for classes: %s", target)
        return target
    if args.preset == "none":
        logger.info("Detecting all COCO classes (--preset none)")
        return None
    if args.preset:
        target = set(CLASS_PRESETS[args.preset])
        logger.info("Using preset '%s' (%d classes)", args.preset, len(target))
        return target
    if cfg.detector.target_classes:
        target = set(cfg.detector.target_classes)
        logger.info("Using config target_classes (%d classes)", len(target))
        return target
    target = set(CLASS_PRESETS["traffic"])
    logger.info("Using default preset 'traffic' (%d classes)", len(target))
    return target


def _parse_points(spec):
    """Parse 'x1,y1,x2,y2,...' into a list of (x, y) float tuples."""
    nums = [float(v) for v in spec.replace(";", ",").split(",") if v.strip() != ""]
    if len(nums) % 2 != 0:
        raise ValueError(f"Expected an even number of coordinates, got: {spec!r}")
    return list(zip(nums[0::2], nums[1::2], strict=True))


def build_analytics(args, props):
    """Build an AnalyticsManager from CLI flags, or None if none were given."""
    analyzers = []
    exporters = []
    recorder = None

    for i, spec in enumerate(args.count_line or [], 1):
        pts = _parse_points(spec)
        if len(pts) != 2:
            raise ValueError(f"--count-line needs exactly 2 points, got {len(pts)}")
        analyzers.append(
            LineCrossingCounter(
                pts[0], pts[1], name=f"line{i}", expected_direction=args.expect_direction
            )
        )

    zones = []
    for i, spec in enumerate(args.zone or [], 1):
        pts = _parse_points(spec)
        if len(pts) < 3:
            raise ValueError(f"--zone needs at least 3 points, got {len(pts)}")
        zones.append(Zone(name=f"zone{i}", polygon=pts, dwell_threshold_s=args.dwell))
    if zones:
        analyzers.append(ZoneManager(zones))

    if args.heatmap:
        analyzers.append(HeatmapAccumulator())
    if args.export_csv:
        exporters.append(CSVExporter(args.export_csv))
    if args.export_db:
        exporters.append(SQLiteExporter(args.export_db))
    if args.record_events:
        recorder = EventClipRecorder(args.record_events, fps=props["fps"])

    if not (analyzers or exporters or recorder):
        return None
    logger.info(
        "Analytics enabled: %d analyzers, %d exporters, recorder=%s",
        len(analyzers),
        len(exporters),
        recorder is not None,
    )
    return AnalyticsManager(analyzers, exporters, recorder)


def main():
    """Main application entry point."""

    # Parse arguments and set up logging/config
    args = parse_arguments()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    # Resolve effective settings: CLI flag overrides config, config overrides built-in.
    model = args.model or cfg.detector.model_name
    confidence = (
        args.confidence if args.confidence is not None else cfg.detector.confidence_threshold
    )
    imgsz = args.imgsz if args.imgsz is not None else cfg.detector.image_size
    device = args.device or cfg.detector.device
    detect_every = args.detect_every if args.detect_every is not None else (3 if args.live else 1)
    max_age = args.max_age if args.max_age is not None else cfg.tracker.max_age
    min_hits = args.min_hits if args.min_hits is not None else cfg.tracker.min_hits
    iou_threshold = (
        args.iou_threshold if args.iou_threshold is not None else cfg.tracker.iou_threshold
    )
    show_trajectory = args.trajectories or cfg.visualization.show_trails
    show_speed = not args.no_speed
    output_path = args.output or cfg.video.output_path
    display = (not args.no_display) and cfg.video.display

    print("\n" + "=" * 60)
    print("  REAL-TIME MULTI-OBJECT TRACKING")
    print("=" * 60)

    video_source = None
    recorder = None
    pipeline = None
    analytics = None
    frame_count = 0

    try:
        # 1. Create video source
        logger.info("[1/4] Initializing video source...")
        video_source = create_video_source(
            args.input,
            webcam_width=cfg.video.webcam_width,
            webcam_height=cfg.video.webcam_height,
            webcam_fps=cfg.video.webcam_fps,
        )
        props = video_source.get_properties()

        # Live mode: capture on a background thread and drop stale frames so
        # slow inference never builds up camera latency. Only meaningful for
        # unbounded sources (webcam/stream) — files are read frame by frame.
        is_live = args.live and props["total_frames"] == -1
        if is_live:
            video_source = LatestFrameGrabber(video_source)
            if args.output or args.record_events:
                logger.warning(
                    "--live drops frames to stay current, but recordings are "
                    "written at the camera's %s FPS — output timing will run "
                    "fast if processing is slower than the camera",
                    props["fps"],
                )
        elif args.live:
            logger.warning("--live has no effect on video files; ignoring")
            if args.detect_every is None:
                detect_every = 1  # undo --live's implied detect-every default

        # 2. Initialize detector
        logger.info("[2/4] Initializing detector...")
        target_classes = resolve_target_classes(args, cfg)
        detector = ObjectDetector(
            model_name=model,
            conf_threshold=confidence,
            img_size=imgsz,
            use_segmentation=args.segmentation,
            target_classes=target_classes,
            use_half=cfg.detector.use_half,
            device=device,
        )

        # 3. Initialize tracker
        logger.info("[3/4] Initializing tracker...")
        tracker = Tracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            class_aware=cfg.tracker.class_aware,
            track_high_thresh=cfg.tracker.track_high_thresh,
            track_low_thresh=cfg.tracker.track_low_thresh,
            output_coast=cfg.tracker.output_coast,
        )

        # Build analytics (line counters, zones, heatmap, exporters, recorder)
        analytics = build_analytics(args, props)

        # Build the reusable pipeline (detect -> track -> analytics -> annotate)
        pipeline = Pipeline(
            detector,
            tracker,
            show_speed=show_speed,
            show_trajectory=show_trajectory,
            draw_masks=args.segmentation,
            analytics_manager=analytics,
            detect_every=detect_every,
        )
        if detect_every > 1:
            logger.info("Detecting every %d frames; tracks coast in between", detect_every)

        # 4. Setup display and recording
        logger.info("[4/4] Setting up display and recording...")

        window_name = "Object Tracking"
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            recorder = cv2.VideoWriter(
                output_path, fourcc, props["fps"], (props["width"], props["height"])
            )
            if not recorder.isOpened():
                logger.warning("Could not open video writer for %s", output_path)
                recorder = None
            else:
                logger.info("Recording to: %s", output_path)

        # Print controls
        print_controls()

        # Print startup info
        print("=" * 60)
        print("  SYSTEM READY")
        print("=" * 60)
        print(f"  Source: {props['source_type']}")
        print(f"  Resolution: {props['width']}x{props['height']}")
        print(f"  FPS Target: {props['fps']}")
        print(f"  Detection: {model} (conf={confidence})")
        if target_classes:
            print(f"  Classes: {', '.join(sorted(target_classes))}")
            if detector.is_world_model:
                print("  Mode: YOLO-World open-vocabulary")
        else:
            print(f"  Classes: ALL ({len(detector.class_names)} classes)")
        print(f"  Segmentation: {'Enabled' if args.segmentation else 'Disabled'}")
        print(f"  Trajectories: {'Enabled' if show_trajectory else 'Disabled'}")
        print(f"  Speed: {'Enabled' if show_speed else 'Disabled'}")
        print(f"  Analytics: {'Enabled' if analytics else 'Disabled'}")
        print("=" * 60 + "\n")

        # Main processing loop
        is_paused = False
        is_file = props["source_type"] == "file"
        fps_limit = cfg.video.fps_limit

        print("Processing... Press 'q' to quit\n")

        while video_source.is_opened():
            loop_start = time.time()

            # Handle pause BEFORE reading: consuming frames while paused would
            # silently skip video content. The paused frame stays on screen
            # (with its PAUSED banner drawn when the pause key was pressed).
            if is_paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord("p") or key == ord("P"):
                    is_paused = False
                    logger.info("Resumed")
                elif key == ord("q") or key == ord("Q") or key == 27:
                    logger.info("Quit requested")
                    break
                continue

            # Read frame
            ret, frame = video_source.read()
            if not ret:
                logger.info("End of video stream")
                break

            frame_count += 1

            # Detect -> track -> annotate (single reusable code path). File
            # sources use media time so dwell/speed reflect video time, not
            # how fast this machine processes it.
            media_ts = frame_count / props["fps"] if is_file else None
            result = pipeline.process_frame(frame, timestamp=media_ts)
            processed_frame = result.frame

            # Add FPS / count overlays (CLI-specific presentation)
            add_overlays(
                processed_frame,
                result.stats.fps,
                result.stats.num_detections,
                result.stats.num_tracks,
                False,
            )

            # Display
            if display:
                cv2.imshow(window_name, processed_frame)

            # Record
            if recorder:
                recorder.write(processed_frame)

            # Progress (for video files; some containers report no frame count)
            if is_file and frame_count % 30 == 0 and props["total_frames"] > 0:
                progress = (frame_count / props["total_frames"]) * 100
                logger.info(
                    "Progress: %.1f%% | Frame %d/%d | FPS: %.1f | Tracks: %d",
                    progress,
                    frame_count,
                    props["total_frames"],
                    result.stats.fps,
                    result.stats.num_tracks,
                )

            # Handle keyboard (only meaningful when a window exists)
            key = (cv2.waitKey(1) & 0xFF) if display else 0xFF

            if key == ord("q") or key == ord("Q") or key == 27:
                logger.info("Quit requested")
                break
            elif key == ord("p") or key == ord("P"):
                is_paused = True
                logger.info("Paused (press 'p' to resume)")
                if display:
                    paused_view = processed_frame.copy()
                    add_overlays(
                        paused_view,
                        result.stats.fps,
                        result.stats.num_detections,
                        result.stats.num_tracks,
                        True,
                    )
                    cv2.imshow(window_name, paused_view)
            elif key == ord("s") or key == ord("S"):
                filename = save_frame(processed_frame)
                logger.info("Saved: %s", filename)

            # Optional processing-rate cap (e.g. play files at real time)
            if fps_limit:
                budget = 1.0 / fps_limit - (time.time() - loop_start)
                if budget > 0:
                    time.sleep(budget)

        # Summary
        avg_fps = pipeline.fps_tracker.average_fps if pipeline else 0.0
        print("\n" + "=" * 60)
        print("  PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Total frames: {frame_count}")
        print(f"  Average FPS: {avg_fps:.2f}")
        if recorder:
            print(f"  Output saved: {output_path}")
        print("=" * 60 + "\n")

    except VideoSourceError as e:
        logger.error("Error: %s", e)
        print("\nTroubleshooting:")
        print("  • For webcam: Make sure camera is connected and not in use")
        print("  • For video: Check file path and format (MP4, AVI, MOV)")
        return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if analytics is not None:
            analytics.save("output")  # e.g. output_heatmap.jpg
            analytics.close()
        if video_source is not None:
            video_source.release()
        if recorder:
            recorder.release()
        cv2.destroyAllWindows()
        logger.info("Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
