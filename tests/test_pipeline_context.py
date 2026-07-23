"""Tests for the FrameContext fields and Pipeline hooks the safety analyzers need.

Covers the plumbing added for camera stabilization, detection provenance, and
previous-frame access — all of which must stay invisible to existing callers.
"""

import numpy as np

from flowcount.analytics.base import FrameContext
from flowcount.detector import Detection
from flowcount.pipeline import Pipeline
from flowcount.tracker import Track


class StubDetector:
    def detect(self, frame):
        return [Detection([10, 10, 50, 50], "car", 0.9)]


class StubTracker:
    def update(self, detections):
        return [
            Track(
                id=1,
                bbox=[10, 10, 50, 50],
                class_label="car",
                confidence=0.9,
                age=1,
                hits=1,
                time_since_update=0,
            )
        ]

    def predict_only(self, max_coast=1):
        return self.update([])


class CapturingAnalytics:
    """Records every ctx it is handed so tests can assert on the contract."""

    def __init__(self):
        self.seen = []

    def update(self, ctx):
        self.seen.append(ctx)
        return []

    def draw(self, frame):
        pass


def _frame(fill=0):
    return np.full((100, 100, 3), fill, dtype=np.uint8)


def test_context_defaults_are_backward_compatible():
    """Analyzers written before these fields existed still construct a ctx."""
    ctx = FrameContext(tracks=[], frame_index=1, timestamp=0.0)
    assert ctx.detections == []
    assert ctx.detection_ran is True
    assert ctx.transform is None
    assert ctx.prev_frame is None


def test_detections_and_detection_ran_are_published():
    analytics = CapturingAnalytics()
    pipeline = Pipeline(StubDetector(), StubTracker(), analytics_manager=analytics, detect_every=2)

    pipeline.process_frame(_frame())
    pipeline.process_frame(_frame())

    ran, coasted = analytics.seen
    assert ran.detection_ran is True
    assert len(ran.detections) == 1
    # A coasted frame has no detections; detection_ran is what distinguishes
    # that from the detector genuinely seeing nothing.
    assert coasted.detection_ran is False
    assert coasted.detections == []


def test_prev_frame_is_opt_in():
    analytics = CapturingAnalytics()
    pipeline = Pipeline(StubDetector(), StubTracker(), analytics_manager=analytics)
    pipeline.process_frame(_frame(1))
    pipeline.process_frame(_frame(2))
    assert all(ctx.prev_frame is None for ctx in analytics.seen)


def test_prev_frame_published_when_requested():
    analytics = CapturingAnalytics()
    pipeline = Pipeline(
        StubDetector(), StubTracker(), analytics_manager=analytics, keep_prev_frame=True
    )
    first, second = _frame(1), _frame(2)
    pipeline.process_frame(first)
    pipeline.process_frame(second)

    assert analytics.seen[0].prev_frame is None  # nothing came before frame 1
    assert analytics.seen[1].prev_frame is first


def test_stabilizer_transform_is_published_and_frame_untouched():
    matrix = np.eye(3, dtype=np.float32)
    calls = []

    class StubStabilizer:
        def estimate(self, frame, exclude_boxes=None, timestamp=None):
            calls.append(exclude_boxes)
            return matrix

    analytics = CapturingAnalytics()
    pipeline = Pipeline(
        StubDetector(),
        StubTracker(),
        analytics_manager=analytics,
        stabilizer=StubStabilizer(),
    )
    frame = _frame()
    result = pipeline.process_frame(frame, annotate=False)

    assert analytics.seen[0].transform is matrix
    # Tracks are handed to the stabilizer so moving objects can be masked out
    # of the motion estimate.
    assert calls == [[[10, 10, 50, 50]]]
    # The frame itself is never warped: the recorder and the video stream must
    # keep seeing exactly what the camera produced.
    assert result.frame is frame
    assert frame.sum() == 0


def test_stabilizer_receives_frame_timestamp():
    """Regression: the stabilizer's time-based reposition detection is dead
    unless process_frame threads the frame timestamp into estimate(). Without
    it a real camera knock never emits ``camera_moved`` and never suspends the
    incident detectors — the stabilizer's whole reason for existing."""
    seen_ts = []

    class TimestampStabilizer:
        def estimate(self, frame, exclude_boxes=None, timestamp=None):
            seen_ts.append(timestamp)
            return np.eye(3)

    pipeline = Pipeline(
        StubDetector(),
        StubTracker(),
        analytics_manager=CapturingAnalytics(),
        stabilizer=TimestampStabilizer(),
    )
    # Explicit media time (video file) must reach the stabilizer verbatim.
    pipeline.process_frame(_frame(), timestamp=12.5)
    # A live source passes no timestamp; the pipeline fills in wall-clock time,
    # so estimate() still gets a real, monotonic float rather than None.
    pipeline.process_frame(_frame())

    assert seen_ts[0] == 12.5
    assert seen_ts[1] is not None and isinstance(seen_ts[1], float)


def test_no_stabilizer_means_no_transform():
    analytics = CapturingAnalytics()
    pipeline = Pipeline(StubDetector(), StubTracker(), analytics_manager=analytics)
    pipeline.process_frame(_frame())
    assert analytics.seen[0].transform is None


def test_output_coast_scaling_is_idempotent():
    """Rebuilding a Pipeline over the same tracker must not compound the scaling.

    Lowering detect_every at runtime (e.g. when an edge device throttles) means
    a second Pipeline over a live tracker; compounding would silently stretch
    the coast window every time.
    """

    class TrackerWithCoast(StubTracker):
        output_coast = 1

    tracker = TrackerWithCoast()

    Pipeline(StubDetector(), tracker, detect_every=3)
    assert tracker.output_coast == 5  # (1 + 1) * 3 - 1

    Pipeline(StubDetector(), tracker, detect_every=3)
    assert tracker.output_coast == 5  # re-derived, not re-scaled

    Pipeline(StubDetector(), tracker, detect_every=2)
    assert tracker.output_coast == 3  # (1 + 1) * 2 - 1

    Pipeline(StubDetector(), tracker, detect_every=1)
    assert tracker.output_coast == 1  # back to the original
