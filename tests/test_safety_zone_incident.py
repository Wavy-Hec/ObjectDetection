"""Tests for ZoneIncidentDetector — the Analyzer that drives the incident engine."""

import numpy as np

from flowcount.analytics.zones import Zone
from flowcount.safety import IncidentRule, ZoneIncidentDetector, intrusion_rule, stalled_rule

ZONE = Zone("crossing", [(100, 100), (400, 100), (400, 400), (100, 400)])


def _car(track_id, cx, cy, *, label="car", conf=0.9, tsu=0, size=40):
    from conftest import FakeTrack

    return FakeTrack(
        track_id,
        [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2],
        label,
        confidence=conf,
        time_since_update=tsu,
    )


def _drive(detector, make_ctx, positions, *, dt=0.1, start=0.0, label="car", tsu=0):
    """Feed a per-frame list of (track_id, cx, cy) and collect every event."""
    events = []
    for i, frame_tracks in enumerate(positions):
        tracks = [_car(tid, cx, cy, label=label, tsu=tsu) for tid, cx, cy in frame_tracks]
        events.extend(
            detector.update(make_ctx(tracks, frame_index=i + 1, timestamp=start + i * dt))
        )
    return events


def test_stalled_vehicle_raises_after_ten_seconds(make_ctx):
    detector = ZoneIncidentDetector([stalled_rule([ZONE])])
    # 12 s at 10 fps, parked in the middle of the zone.
    events = _drive(detector, make_ctx, [[(1, 250, 250)] for _ in range(120)])

    kinds = [e.kind for e in events]
    assert kinds == ["stalled"]
    raised = events[0]
    assert raised.data["zone"] == "crossing"
    assert raised.data["severity"] == "critical"
    assert raised.data["incident_id"]
    assert 10.0 <= raised.data["duration_s"] <= 11.0


def test_moving_vehicle_never_raises(make_ctx):
    detector = ZoneIncidentDetector([stalled_rule([ZONE])])
    events = _drive(detector, make_ctx, [[(1, 110 + i * 2, 250)] for i in range(120)])
    assert events == []


def test_vehicle_outside_the_zone_is_ignored(make_ctx):
    detector = ZoneIncidentDetector([stalled_rule([ZONE])])
    events = _drive(detector, make_ctx, [[(1, 600, 600)] for _ in range(120)])
    assert events == []


def test_class_filter_excludes_pedestrians_from_the_stalled_rule(make_ctx):
    detector = ZoneIncidentDetector([stalled_rule([ZONE])])
    events = _drive(detector, make_ctx, [[(1, 250, 250)] for _ in range(120)], label="person")
    assert events == []


def test_intrusion_fires_fast_for_a_pedestrian(make_ctx):
    """1.5 s, not 10 s: the cost asymmetry runs the other way here."""
    detector = ZoneIncidentDetector([intrusion_rule([ZONE])])
    events = _drive(detector, make_ctx, [[(1, 250, 250)] for _ in range(30)], label="person")
    assert [e.kind for e in events] == ["intrusion"]
    assert events[0].data["duration_s"] < 2.0


def test_intrusion_fires_for_a_walking_pedestrian(make_ctx):
    """Presence is the trigger, so a person who keeps walking still counts."""
    detector = ZoneIncidentDetector([intrusion_rule([ZONE])])
    events = _drive(
        detector, make_ctx, [[(1, 150 + i * 3, 250)] for i in range(30)], label="person"
    )
    assert [e.kind for e in events] == ["intrusion"]


def test_escalation_and_clear_are_emitted(make_ctx):
    rule = stalled_rule([ZONE], t_raise=2.0, t_escalate=(5.0,), t_clear=1.0)
    detector = ZoneIncidentDetector([rule])

    frames = [[(1, 250, 250)] for _ in range(80)]  # 8 s parked
    frames += [[(1, 250 + i * 12, 250)] for i in range(30)]  # then drives off
    events = _drive(detector, make_ctx, frames)

    kinds = [e.kind for e in events]
    assert kinds == ["stalled", "stalled_escalate", "stalled_clear"]
    assert events[1].data["level"] == 1
    assert events[2].data["reason"] == "moved"


def test_congestion_reported_instead_of_stalled_when_peers_are_present(make_ctx):
    """A queue is a collective anomaly; paging dispatch for rush hour is noise."""
    detector = ZoneIncidentDetector([stalled_rule([ZONE], congestion_peers=3)])
    frames = [[(1, 150, 250), (2, 250, 250), (3, 350, 250)] for _ in range(120)]
    events = _drive(detector, make_ctx, frames)

    assert {e.kind for e in events} == {"congestion"}
    assert all(e.data["severity"] == "info" for e in events)
    assert all(e.data["peers"] >= 3 for e in events)


def test_single_stall_amid_traffic_is_still_a_stall(make_ctx):
    """One stopped car with moving traffic around it must not read as congestion."""
    detector = ZoneIncidentDetector([stalled_rule([ZONE], congestion_peers=3)])
    frames = [[(1, 250, 200), (2, 110 + i * 2, 300), (3, 130 + i * 2, 350)] for i in range(120)]
    events = _drive(detector, make_ctx, frames)
    assert [e.kind for e in events] == ["stalled"]


def test_coasted_frames_do_not_manufacture_a_stall(make_ctx):
    """Kalman-predicted boxes barely move; that must not read as stopping."""
    detector = ZoneIncidentDetector([stalled_rule([ZONE])])
    events = _drive(detector, make_ctx, [[(1, 250, 250)] for _ in range(120)], tsu=2)
    assert events == []


def test_camera_sway_does_not_manufacture_or_hide_a_stall(make_ctx):
    """With a stabilizer transform, sway is cancelled; without it, it isn't."""
    frames = 140
    sway = [16.0 * np.sin(i * 0.3) for i in range(frames)]

    unstabilized = ZoneIncidentDetector([stalled_rule([ZONE])])
    events = []
    for i in range(frames):
        tracks = [_car(1, 250 + sway[i], 250)]
        events.extend(unstabilized.update(make_ctx(tracks, frame_index=i + 1, timestamp=i * 0.1)))
    assert events == []  # the parked car is missed entirely

    stabilized = ZoneIncidentDetector([stalled_rule([ZONE])])
    events = []
    for i in range(frames):
        tracks = [_car(1, 250 + sway[i], 250)]
        matrix = np.array([[1, 0, -sway[i]], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        events.extend(
            stabilized.update(
                make_ctx(tracks, frame_index=i + 1, timestamp=i * 0.1, transform=matrix)
            )
        )
    assert [e.kind for e in events] == ["stalled"]


def test_suspension_blocks_alerts(make_ctx):
    detector = ZoneIncidentDetector([stalled_rule([ZONE])])
    detector.suspend(until_ts=1000.0)
    events = _drive(detector, make_ctx, [[(1, 250, 250)] for _ in range(150)])
    assert events == []
    assert detector.suspended


def test_stats_and_draw(make_ctx):
    detector = ZoneIncidentDetector([stalled_rule([ZONE], t_raise=1.0)])
    _drive(detector, make_ctx, [[(1, 250, 250)] for _ in range(40)])

    stats = detector.stats()["incidents"]
    assert stats["counts"] == {"stalled": 1}
    assert stats["open"][0]["zone"] == "crossing"
    assert stats["open"][0]["class"] == "car"

    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    detector.draw(frame)
    assert frame.sum() > 0  # something was actually drawn


def test_multiple_rules_run_independently(make_ctx):
    detector = ZoneIncidentDetector([stalled_rule([ZONE]), intrusion_rule([ZONE])])
    frames = [[(1, 250, 250)] for _ in range(120)]
    car_events = _drive(detector, make_ctx, frames)
    assert [e.kind for e in car_events] == ["stalled"]

    detector2 = ZoneIncidentDetector([stalled_rule([ZONE]), intrusion_rule([ZONE])])
    person_events = _drive(detector2, make_ctx, frames, label="person")
    assert [e.kind for e in person_events] == ["intrusion"]


def test_rule_without_zones_watches_the_whole_frame(make_ctx):
    detector = ZoneIncidentDetector([IncidentRule(kind="stalled", t_raise=1.0)])
    events = _drive(detector, make_ctx, [[(1, 900, 900)] for _ in range(40)])
    assert [e.kind for e in events] == ["stalled"]
    assert events[0].data["zone"] is None


def test_track_windows_are_pruned(make_ctx):
    """Long runs must not accumulate one motion window per track ever seen."""
    detector = ZoneIncidentDetector([stalled_rule([ZONE])])
    for i in range(30):
        detector.update(make_ctx([_car(i, 250, 250)], frame_index=i + 1, timestamp=i * 1.0))
    assert len(detector._windows) < 30


def test_zone_membership_is_tested_in_the_stabilized_frame(make_ctx):
    """Zone polygons are fixed against the reference view.

    Testing containment in raw pixels while anchoring the incident in
    stabilized coordinates makes an object near the boundary drift in and out
    of its own zone as the camera sways, while its incident sits still.
    """
    narrow = Zone("narrow", [(240, 100), (280, 100), (280, 400), (240, 400)])
    detector = ZoneIncidentDetector([stalled_rule([narrow], t_raise=1.0)])

    # Camera drifts +40 px; the car is parked at x=260 in the world, so in raw
    # pixels it appears at 300 — outside the zone — but the transform undoes it.
    events = []
    for i in range(60):
        track = _car(1, 300, 250)
        matrix = np.array([[1, 0, -40], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        events.extend(
            detector.update(
                make_ctx([track], frame_index=i + 1, timestamp=i * 0.1, transform=matrix)
            )
        )

    assert [e.kind for e in events] == ["stalled"]
    assert events[0].data["zone"] == "narrow"


def test_clear_reports_duration_not_an_escalation_level(make_ctx):
    """`level` means escalation level. A clear event must not populate it."""
    rule = stalled_rule([ZONE], t_raise=2.0, t_escalate=(), t_clear=1.0)
    detector = ZoneIncidentDetector([rule])

    frames = [[(1, 250, 250)] for _ in range(60)]  # 6 s parked
    frames += [[(1, 250 + i * 12, 250)] for i in range(30)]  # then leaves
    events = _drive(detector, make_ctx, frames)

    clear = [e for e in events if e.kind == "stalled_clear"][0]
    assert "level" not in clear.data
    # Duration is how long it was stalled, not how long ago it started —
    # the clearing window must not inflate it.
    assert 4.0 <= clear.data["duration_s"] <= 7.0
