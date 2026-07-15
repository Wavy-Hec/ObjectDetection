"""Tests for the line-crossing counter."""

from flowcount.analytics import LineCrossingCounter


def test_counts_a_crossing(make_track, make_ctx):
    counter = LineCrossingCounter((0, 100), (200, 100), name="L")
    # frame 1: above the line (center y=50)
    counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=1))
    # frame 2: below the line (center y=150) -> crossed
    events = counter.update(make_ctx([make_track(1, [40, 140, 60, 160])], frame_index=2))
    assert len(events) == 1
    assert events[0].kind == "line_cross"
    assert counter.total_in + counter.total_out == 1


def test_no_crossing_when_moving_parallel(make_track, make_ctx):
    counter = LineCrossingCounter((0, 100), (200, 100))
    counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=1))
    events = counter.update(make_ctx([make_track(1, [80, 40, 100, 60])], frame_index=2))
    assert events == []
    assert counter.total_in == 0 and counter.total_out == 0


def test_direction_in_then_out(make_track, make_ctx):
    counter = LineCrossingCounter((0, 100), (200, 100))
    counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=1))  # above
    counter.update(make_ctx([make_track(1, [40, 140, 60, 160])], frame_index=2))  # below -> in
    assert counter.total_in == 1 and counter.total_out == 0
    counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=3))  # above -> out
    assert counter.total_out == 1


def test_class_filter(make_track, make_ctx):
    counter = LineCrossingCounter((0, 100), (200, 100), classes={"person"})
    counter.update(make_ctx([make_track(1, [40, 40, 60, 60], "car")], frame_index=1))
    events = counter.update(make_ctx([make_track(1, [40, 140, 60, 160], "car")], frame_index=2))
    assert events == []  # car is filtered out


def test_counts_per_class(make_track, make_ctx):
    counter = LineCrossingCounter((0, 100), (200, 100))
    counter.update(make_ctx([make_track(5, [40, 40, 60, 60], "person")], frame_index=1))
    counter.update(make_ctx([make_track(5, [40, 140, 60, 160], "person")], frame_index=2))
    assert counter.counts["in"]["person"] == 1


def test_wrong_way_event_on_unexpected_direction(make_track, make_ctx):
    counter = LineCrossingCounter((0, 100), (200, 100), name="L", expected_direction="in")
    # Crossing top->bottom is "in" for this line: expected, no alert.
    counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=1))
    events = counter.update(make_ctx([make_track(1, [40, 140, 60, 160])], frame_index=2))
    assert [e.kind for e in events] == ["line_cross"]

    # Crossing back (an "out") violates expected_direction -> wrong_way fires.
    events = counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=3))
    assert [e.kind for e in events] == ["line_cross", "wrong_way"]
    assert events[1].data["direction"] == "out"


def test_no_wrong_way_without_expectation(make_track, make_ctx):
    counter = LineCrossingCounter((0, 100), (200, 100))
    counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=1))
    counter.update(make_ctx([make_track(1, [40, 140, 60, 160])], frame_index=2))
    events = counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=3))
    assert all(e.kind == "line_cross" for e in events)


def test_crossing_survives_short_dropout(make_track, make_ctx):
    """A track that vanishes for a couple of frames mid-crossing still counts."""
    counter = LineCrossingCounter((0, 100), (200, 100))
    counter.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=1))
    counter.update(make_ctx([], frame_index=2))  # dropout
    counter.update(make_ctx([], frame_index=3))  # dropout
    events = counter.update(make_ctx([make_track(1, [40, 140, 60, 160])], frame_index=4))
    assert len(events) == 1  # prev center survived the grace window
