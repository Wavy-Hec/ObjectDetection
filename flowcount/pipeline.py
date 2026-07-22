"""Reusable detection + tracking pipeline.

Wraps the ``detect -> track -> (optionally) annotate`` flow in a single
reusable unit so the CLI (``main.py``), the web dashboard, and the benchmark
module all share one code path instead of duplicating the loop body.

Example::

    pipeline = Pipeline(detector, tracker)
    result = pipeline.process_frame(frame)
    cv2.imshow("out", result.frame)
    print(result.stats.fps, result.stats.num_tracks)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .analytics.base import Event, FrameContext
from .detector import Detection
from .tracker import Track
from .visualization import draw_segmentation_masks, draw_tracks


class FPSTracker:
    """Track FPS with a smoothed moving average over recent frames."""

    def __init__(self, buffer_size: int = 30):
        self.times: list[float] = []
        self.buffer_size = buffer_size
        self.last_time = time.time()

    def update(self) -> float:
        """Record the time since the last call and return the smoothed FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time

        self.times.append(elapsed)
        if len(self.times) > self.buffer_size:
            self.times.pop(0)

        return self.average_fps

    @property
    def average_fps(self) -> float:
        """Smoothed FPS over the current buffer (0.0 if no samples yet)."""
        if not self.times:
            return 0.0
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


@dataclass
class FrameStats:
    """Per-frame statistics produced by the pipeline."""

    frame_index: int
    fps: float
    num_detections: int
    num_tracks: int
    detection_ran: bool = True  # False on frames coasted via detect_every


@dataclass
class ProcessResult:
    """Result of processing a single frame."""

    frame: np.ndarray  # annotated frame (or the original if annotate=False)
    detections: list[Detection]
    tracks: list[Track]
    stats: FrameStats
    events: list[Event] = field(default_factory=list)  # analytics events this frame


class Pipeline:
    """Owns the detector + tracker and runs them over frames.

    The pipeline deliberately does NOT handle display, recording, keyboard
    input, or pause state — those belong to the caller (CLI / web server).
    This keeps the same code path reusable across every front end.
    """

    def __init__(
        self,
        detector,
        tracker,
        *,
        show_speed: bool = True,
        show_trajectory: bool = False,
        draw_masks: bool = True,
        analytics_manager=None,
        detect_every: int = 1,
        stabilizer=None,
        keep_prev_frame: bool = False,
    ):
        """
        Args:
            detector: Anything with a ``detect(frame) -> List[Detection]`` method.
            tracker: Anything with an ``update(detections) -> List[Track]`` method.
            show_speed: Draw per-track speed labels.
            show_trajectory: Draw per-track trajectory trails.
            draw_masks: Overlay segmentation masks when detections carry them.
            analytics_manager: Optional Phase 2 hook with ``update(tracks, ...)``
                and optional ``draw(frame)`` methods.
            detect_every: Run the detector on every Nth frame only; between
                detection frames, tracks coast on Kalman prediction via
                ``tracker.predict_only()``. 1 (the default) detects every
                frame. Values > 1 trade detection latency for a large FPS
                gain — the main lever for live CPU-only processing.
            stabilizer: Optional camera-motion estimator with
                ``estimate(frame, exclude_boxes=...) -> np.ndarray | None``.
                Its 3x3 image->reference transform is published on
                ``FrameContext.transform`` for analyzers that compensate for
                camera drift. The frame itself is never warped: downstream
                consumers (recorder, display, MJPEG stream) must keep seeing
                exactly what the camera produced.
            keep_prev_frame: Publish the previous frame on
                ``FrameContext.prev_frame``. Opt-in because it keeps one extra
                frame alive; safe without copying because every video source
                allocates a fresh array per read rather than reusing a buffer.
        """
        self.detector = detector
        self.tracker = tracker
        self.show_speed = show_speed
        self.show_trajectory = show_trajectory
        self.draw_masks = draw_masks
        self.analytics_manager = analytics_manager
        self.detect_every = max(1, int(detect_every))
        self.stabilizer = stabilizer
        self.keep_prev_frame = keep_prev_frame
        # output_coast is measured in raw frames, but a "missed detection"
        # under detect-every-N costs N frames: a track missed at one detection
        # frame reaches time_since_update = 2N-1 before its next chance to
        # re-match. Scale the tracker's window so output_coast keeps meaning
        # "missed detection frames absorbed" regardless of N.
        #
        # The unscaled value is remembered on the tracker so that building a
        # second Pipeline over the same tracker (e.g. lowering detect_every
        # when an edge device throttles) re-derives the window instead of
        # compounding the previous scaling.
        if hasattr(tracker, "output_coast"):
            base_coast = getattr(tracker, "_base_output_coast", tracker.output_coast)
            tracker._base_output_coast = base_coast
            tracker.output_coast = (
                (base_coast + 1) * self.detect_every - 1 if self.detect_every > 1 else base_coast
            )
        self._coast_window = getattr(tracker, "output_coast", self.detect_every)
        self.fps_tracker = FPSTracker()
        self.frame_index = 0
        self._prev_frame: np.ndarray | None = None

    def process_frame(
        self, frame: np.ndarray, *, annotate: bool = True, timestamp: float = None
    ) -> ProcessResult:
        """Run detection + tracking on a frame and (optionally) annotate it.

        Args:
            frame: BGR frame from OpenCV.
            annotate: When True, draw tracks/masks onto a *copy* of the frame
                (the input frame is never mutated). When False, the original
                frame is returned untouched in ``ProcessResult.frame``.
            timestamp: Media time of this frame in seconds. Pass
                ``frame_number / source_fps`` when processing a video file so
                time-based analytics (dwell, speed) reflect video time rather
                than processing speed. Defaults to wall-clock time, which is
                correct for live sources.

        Returns:
            A :class:`ProcessResult` with detections, tracks, the (possibly
            annotated) frame, and per-frame stats.
        """
        self.frame_index += 1
        if timestamp is None:
            timestamp = time.time()

        detection_ran = self.detect_every <= 1 or (self.frame_index - 1) % self.detect_every == 0
        if detection_ran:
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
        else:
            detections = []
            tracks = self.tracker.predict_only(max_coast=self._coast_window)
        fps = self.fps_tracker.update()

        # Camera-motion estimate, after tracking so the tracks themselves can
        # mask moving objects out of the estimate — without that, heavy traffic
        # drags the transform toward the vehicles' motion.
        transform = None
        if self.stabilizer is not None:
            transform = self.stabilizer.estimate(frame, exclude_boxes=[t.bbox for t in tracks])

        # Phase 2 analytics hook (line counters, zones, heatmaps, ...).
        events: list[Any] = []
        if self.analytics_manager is not None:
            ctx = FrameContext(
                tracks=tracks,
                frame_index=self.frame_index,
                timestamp=timestamp,
                fps=fps,
                frame=frame,  # raw (un-annotated) frame for heatmap/recorder
                detections=detections,
                detection_ran=detection_ran,
                transform=transform,
                prev_frame=self._prev_frame,
            )
            events = self.analytics_manager.update(ctx)

        if self.keep_prev_frame:
            self._prev_frame = frame

        out = frame
        if annotate:
            out = frame.copy()
            if self.draw_masks and any(getattr(d, "mask", None) is not None for d in detections):
                draw_segmentation_masks(out, detections, alpha=0.3)
            draw_tracks(
                out, tracks, show_speed=self.show_speed, show_trajectory=self.show_trajectory
            )
            if self.analytics_manager is not None:
                self.analytics_manager.draw(out)

        stats = FrameStats(
            frame_index=self.frame_index,
            fps=fps,
            num_detections=len(detections),
            num_tracks=len(tracks),
            detection_ran=detection_ran,
        )
        return ProcessResult(
            frame=out, detections=detections, tracks=tracks, stats=stats, events=events
        )
