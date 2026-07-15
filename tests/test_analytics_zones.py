"""Tests for zone occupancy, enter/exit, and dwell-time analytics."""

from flowcount.analytics import Zone, ZoneManager

SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]


def test_enter_exit_and_occupancy(make_track, make_ctx):
    zm = ZoneManager([Zone("z", SQUARE)])

    # outside the zone
    events = zm.update(make_ctx([make_track(1, [10, 10, 30, 30])], frame_index=1, timestamp=0.0))
    assert events == []
    assert zm.occupancy("z") == 0

    # inside the zone (center 200,200)
    events = zm.update(make_ctx([make_track(1, [190, 190, 210, 210])], frame_index=2, timestamp=1.0))
    assert any(e.kind == "zone_enter" for e in events)
    assert zm.occupancy("z") == 1

    # track disappears -> exit with dwell time
    events = zm.update(make_ctx([], frame_index=3, timestamp=5.0))
    exits = [e for e in events if e.kind == "zone_exit"]
    assert len(exits) == 1
    assert exits[0].data["dwell_s"] == 4.0  # 5.0 - 1.0
    assert zm.occupancy("z") == 0


def test_dwell_event_fires_past_threshold(make_track, make_ctx):
    zm = ZoneManager([Zone("z", SQUARE, dwell_threshold_s=2.0)])
    zm.update(make_ctx([make_track(1, [190, 190, 210, 210])], frame_index=1, timestamp=0.0))

    early = zm.update(make_ctx([make_track(1, [195, 195, 215, 215])], frame_index=2, timestamp=1.0))
    assert not any(e.kind == "dwell" for e in early)

    late = zm.update(make_ctx([make_track(1, [195, 195, 215, 215])], frame_index=3, timestamp=2.5))
    dwell = [e for e in late if e.kind == "dwell"]
    assert len(dwell) == 1
    assert dwell[0].data["zone"] == "z"


def test_dwell_fires_only_once(make_track, make_ctx):
    zm = ZoneManager([Zone("z", SQUARE, dwell_threshold_s=1.0)])
    zm.update(make_ctx([make_track(1, [190, 190, 210, 210])], frame_index=1, timestamp=0.0))
    first = zm.update(make_ctx([make_track(1, [190, 190, 210, 210])], frame_index=2, timestamp=2.0))
    second = zm.update(make_ctx([make_track(1, [190, 190, 210, 210])], frame_index=3, timestamp=3.0))
    assert sum(e.kind == "dwell" for e in first) == 1
    assert sum(e.kind == "dwell" for e in second) == 0


def test_class_filter_on_zone(make_track, make_ctx):
    zm = ZoneManager([Zone("z", SQUARE, classes={"person"})])
    events = zm.update(make_ctx([make_track(1, [190, 190, 210, 210], "car")],
                                frame_index=1, timestamp=0.0))
    assert events == []
    assert zm.occupancy("z") == 0
