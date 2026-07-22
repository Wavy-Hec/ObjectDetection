"""Pytest configuration shared across the test suite.

Living at the repo root, this file makes pytest add the repo root to
``sys.path`` so ``import flowcount`` works, and provides an autouse fixture that
resets the global Kalman track-ID counter before each test so that track-ID
assertions are deterministic.
"""

from dataclasses import dataclass

import pytest


@dataclass
class FakeTrack:
    """Lightweight stand-in for tracker.Track for analytics tests.

    Mirrors the full public surface of ``flowcount.tracker.Track`` — analyzers
    legitimately read ``time_since_update`` (observed vs. Kalman-predicted) and
    ``confidence``, so a fixture missing them would fail every test that uses it
    rather than the code under test.
    """

    id: int
    bbox: list[float]
    class_label: str = "person"
    velocity: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.9
    age: int = 1
    hits: int = 1
    time_since_update: int = 0  # 0 = detection-backed this frame
    class_id: int = -1
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = [self.get_center()]

    def get_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def get_speed(self):
        return (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5


@pytest.fixture
def make_track():
    def _make(
        track_id,
        bbox,
        class_label="person",
        velocity=(0.0, 0.0),
        *,
        confidence=0.9,
        time_since_update=0,
    ):
        return FakeTrack(
            track_id,
            list(bbox),
            class_label,
            velocity,
            confidence=confidence,
            time_since_update=time_since_update,
        )

    return _make


@pytest.fixture
def make_ctx():
    from flowcount.analytics import FrameContext

    def _make(
        tracks,
        frame_index=1,
        timestamp=0.0,
        fps=0.0,
        frame=None,
        *,
        detections=None,
        detection_ran=True,
        transform=None,
        prev_frame=None,
    ):
        return FrameContext(
            tracks=tracks,
            frame_index=frame_index,
            timestamp=timestamp,
            fps=fps,
            frame=frame,
            detections=list(detections or []),
            detection_ran=detection_ran,
            transform=transform,
            prev_frame=prev_frame,
        )

    return _make


@pytest.fixture(autouse=True)
def _reset_kalman_counter():
    """Reset ``KalmanBoxTracker.count`` (a class-level global) around each
    test. Guarded so tests that don't need the tracker still run even if
    ``filterpy`` isn't installed."""
    try:
        from flowcount.tracker import KalmanBoxTracker
    except Exception:
        yield
        return

    KalmanBoxTracker.count = 0
    yield
    KalmanBoxTracker.count = 0
