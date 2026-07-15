"""Pytest configuration shared across the test suite.

Living at the repo root, this file makes pytest add the repo root to
``sys.path`` so ``import flowcount`` works, and provides an autouse fixture that
resets the global Kalman track-ID counter before each test so that track-ID
assertions are deterministic.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import pytest


@dataclass
class FakeTrack:
    """Lightweight stand-in for tracker.Track for analytics tests
    (implements just the interface analyzers rely on)."""

    id: int
    bbox: List[float]
    class_label: str = "person"
    velocity: Tuple[float, float] = (0.0, 0.0)

    def get_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def get_speed(self):
        return (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5


@pytest.fixture
def make_track():
    def _make(track_id, bbox, class_label="person", velocity=(0.0, 0.0)):
        return FakeTrack(track_id, list(bbox), class_label, velocity)
    return _make


@pytest.fixture
def make_ctx():
    from flowcount.analytics import FrameContext

    def _make(tracks, frame_index=1, timestamp=0.0, fps=0.0, frame=None):
        return FrameContext(tracks=tracks, frame_index=frame_index,
                            timestamp=timestamp, fps=fps, frame=frame)
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
