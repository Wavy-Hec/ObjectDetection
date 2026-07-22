"""Tests for debris / abandoned-object detection.

Frames are generated here rather than read from a fixture so every scenario —
an object appearing, a car stopping, a cloud front — is exactly reproducible
and the assertions can be about behaviour rather than about one video file.
"""

import cv2
import numpy as np

from flowcount.analytics.zones import Zone
from flowcount.safety.static_objects import DebrisConfig, StaticObjectMonitor

W, H = 640, 360
FPS = 10.0

# Fast enough for a test, but keeps the invariant the algorithm needs:
# tau_fast (2 s) < persist (6 s) < tau_slow / 2 (20 s).
TEST_CFG = DebrisConfig(
    rate_hz=5.0,
    fast_history=10,
    slow_history=200,
    persist_s=6.0,
    warmup_s=4.0,
    t_clear=2.0,
)


def _background(seed=0):
    """Textured asphalt — a flat colour would make every subtractor look good."""
    rng = np.random.default_rng(seed)
    return rng.integers(55, 95, (H, W, 3), dtype=np.uint8)


def _render(bg, *, car_x=None, debris=None, brightness=0):
    frame = bg.copy()
    if car_x is not None:
        cv2.rectangle(frame, (car_x, 150), (car_x + 60, 190), (205, 205, 205), -1)
        cv2.rectangle(frame, (car_x + 12, 155), (car_x + 40, 170), (90, 70, 50), -1)
    if debris is not None:
        x, y = debris
        cv2.rectangle(frame, (x, y), (x + 20, y + 16), (30, 30, 165), -1)
        cv2.rectangle(frame, (x + 3, y + 3), (x + 17, y + 13), (80, 60, 200), -1)
    if brightness:
        frame = cv2.add(frame, np.full_like(frame, brightness))
    return frame


class _Track:
    """Minimal stand-in matching the attributes the monitor reads."""

    def __init__(self, bbox):
        self.bbox = list(bbox)
        self.id = 1
        self.class_label = "car"

    def get_center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


def _drive(
    monitor,
    make_ctx,
    seconds,
    *,
    debris_from=None,
    debris_at=(300, 250),
    stopped_car=None,
    brightness_from=None,
    seed=0,
):
    """Run a scene: traffic flows, optionally something is dropped."""
    bg = _background(seed)
    events = []
    for i in range(int(FPS * seconds)):
        now = i / FPS
        car_x = int((i * 9) % (W + 120)) - 60
        tracks = [_Track([car_x, 150, car_x + 60, 190])]
        if stopped_car is not None and now >= stopped_car:
            car_x = 420
            tracks = [_Track([car_x, 150, car_x + 60, 190])]

        debris = debris_at if (debris_from is not None and now >= debris_from) else None
        brightness = 60 if (brightness_from is not None and now >= brightness_from) else 0
        frame = _render(bg, car_x=car_x, debris=debris, brightness=brightness)

        events.extend(
            monitor.update(make_ctx(tracks, frame_index=i + 1, timestamp=now, frame=frame))
        )
    return events


def test_no_debris_means_no_events(make_ctx):
    monitor = StaticObjectMonitor(config=TEST_CFG)
    assert _drive(monitor, make_ctx, seconds=40) == []


def test_dropped_object_is_reported(make_ctx):
    monitor = StaticObjectMonitor(config=TEST_CFG)
    events = _drive(monitor, make_ctx, seconds=45, debris_from=15.0)

    raised = [e for e in events if e.kind == "debris"]
    assert len(raised) == 1
    assert raised[0].data["requires_human_review"] is True
    assert raised[0].data["severity"] == "low"
    # Reported after the persistence window, not instantly and not never.
    assert 15.0 + TEST_CFG.persist_s <= raised[0].timestamp <= 15.0 + 25.0


def test_debris_is_reported_once_not_every_frame(make_ctx):
    monitor = StaticObjectMonitor(config=TEST_CFG)
    events = _drive(monitor, make_ctx, seconds=60, debris_from=15.0)
    assert len([e for e in events if e.kind == "debris"]) == 1


def test_removed_object_clears(make_ctx):
    """Present for a while, then gone: the incident must close."""
    monitor = StaticObjectMonitor(config=TEST_CFG)
    bg = _background()
    events = []
    for i in range(int(FPS * 70)):
        now = i / FPS
        car_x = int((i * 9) % (W + 120)) - 60
        debris = (300, 250) if 10.0 <= now < 40.0 else None
        frame = _render(bg, car_x=car_x, debris=debris)
        events.extend(
            monitor.update(
                make_ctx(
                    [_Track([car_x, 150, car_x + 60, 190])],
                    frame_index=i + 1,
                    timestamp=now,
                    frame=frame,
                )
            )
        )

    kinds = [e.kind for e in events]
    assert kinds == ["debris", "debris_clear"]
    assert events[1].timestamp > 40.0


def test_a_stopped_vehicle_is_not_debris(make_ctx):
    """It is a stall, handled by the incident detector — not a dropped object.

    This is what the track-overlap gate is for: without it the biggest, most
    obvious static blob in the scene would be misreported.
    """
    monitor = StaticObjectMonitor(config=TEST_CFG)
    events = _drive(monitor, make_ctx, seconds=60, stopped_car=10.0)
    assert [e for e in events if e.kind == "debris"] == []


def test_lighting_change_does_not_fire(make_ctx):
    """A cloud front flips a huge fraction of pixels; that is not an object."""
    monitor = StaticObjectMonitor(config=TEST_CFG)
    events = _drive(monitor, make_ctx, seconds=50, brightness_from=15.0)
    assert [e for e in events if e.kind == "debris"] == []


def test_zone_filter_excludes_outside_debris(make_ctx):
    zone = Zone("roadway", [(0, 0), (200, 0), (200, H), (0, H)])
    monitor = StaticObjectMonitor([zone], config=TEST_CFG)
    events = _drive(monitor, make_ctx, seconds=45, debris_from=15.0, debris_at=(500, 250))
    assert events == []


def test_zone_filter_includes_inside_debris(make_ctx):
    zone = Zone("roadway", [(0, 200), (W, 200), (W, H), (0, H)])
    monitor = StaticObjectMonitor([zone], config=TEST_CFG)
    events = _drive(monitor, make_ctx, seconds=45, debris_from=15.0, debris_at=(300, 250))
    assert [e.kind for e in events] == ["debris"]
    assert events[0].data["zone"] == "roadway"


def test_nothing_fires_during_warmup(make_ctx):
    """The background models are meaningless until they have converged."""
    monitor = StaticObjectMonitor(config=TEST_CFG)
    events = _drive(monitor, make_ctx, seconds=3, debris_from=0.0)
    assert events == []
    assert monitor.warming


def test_stats_and_draw(make_ctx):
    monitor = StaticObjectMonitor(config=TEST_CFG)
    _drive(monitor, make_ctx, seconds=45, debris_from=15.0)
    stats = monitor.stats()["debris"]
    assert stats["open"] == 1
    assert stats["warming"] is False

    frame = np.zeros((H, W, 3), dtype=np.uint8)
    monitor.draw(frame)
    assert frame.sum() > 0


def test_processing_is_rate_limited(make_ctx):
    """Pixel work runs at a few Hz, not at frame rate — the edge cost control."""
    monitor = StaticObjectMonitor(config=TEST_CFG)
    bg = _background()
    processed = []
    for i in range(60):  # 6 s at 10 fps
        now = i / FPS
        frame = _render(bg, car_x=100)
        before = monitor._last_processed_ts
        monitor.update(make_ctx([], frame_index=i + 1, timestamp=now, frame=frame))
        if monitor._last_processed_ts != before:
            processed.append(now)

    # 6 s at 5 Hz is ~30 passes, not 60.
    assert 25 <= len(processed) <= 35
