"""Tests for the tracker: SORT core (Kalman + Hungarian) with ByteTrack-style
two-stage association, confirmed-latch, and predict_only coast mode."""

from flowcount.detector import Detection
from flowcount.tracker import Track, Tracker


def make_det(bbox, label="person", conf=0.9):
    return Detection(bbox=list(bbox), class_label=label, confidence=conf)


def test_iou_identical_boxes():
    assert Tracker._iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_boxes():
    assert Tracker._iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 region -> inter=50, union=150.
    assert abs(Tracker._iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-6


def test_single_object_gets_stable_id():
    tracker = Tracker(min_hits=1, iou_threshold=0.3)

    tracks = tracker.update([make_det([100, 100, 200, 200])])
    assert len(tracks) == 1
    first_id = tracks[0].id
    assert first_id == 0  # counter reset by the autouse fixture

    moved = tracker.update([make_det([108, 106, 208, 206])])
    assert len(moved) == 1
    assert moved[0].id == first_id  # same object keeps its ID


def test_class_aware_matching_keeps_classes_separate():
    tracker = Tracker(min_hits=1)
    tracks = tracker.update(
        [
            make_det([100, 100, 200, 200], "person"),
            make_det([300, 300, 400, 400], "car"),
        ]
    )
    assert sorted(t.class_label for t in tracks) == ["car", "person"]
    assert len({t.id for t in tracks}) == 2


def test_track_expires_after_max_age():
    tracker = Tracker(min_hits=1, max_age=3)
    tracker.update([make_det([100, 100, 200, 200])])

    out = None
    for _ in range(6):
        out = tracker.update([])  # no detections -> track ages out

    assert out == []
    assert len(tracker.trackers) == 0


def test_track_center_and_speed():
    track = Track(
        id=7,
        bbox=[0, 0, 10, 20],
        class_label="person",
        confidence=0.9,
        age=1,
        hits=1,
        time_since_update=0,
        velocity=(3.0, 4.0),
    )
    assert track.get_center() == (5.0, 10.0)
    assert track.get_speed() == 5.0  # 3-4-5 triangle


# --- ByteTrack two-stage association -----------------------------------------


def test_low_confidence_detection_recovers_existing_track():
    tracker = Tracker(min_hits=1)
    first = tracker.update([make_det([100, 100, 200, 200], conf=0.9)])
    assert len(first) == 1

    # Next frame the object is half-occluded: only a low-confidence detection.
    recovered = tracker.update([make_det([105, 102, 205, 202], conf=0.3)])
    assert len(recovered) == 1
    assert recovered[0].id == first[0].id
    assert recovered[0].time_since_update == 0  # stage 2 matched it


def test_low_confidence_detection_never_spawns_a_track():
    tracker = Tracker(min_hits=1)
    out = tracker.update([make_det([100, 100, 200, 200], conf=0.3)])
    assert out == []
    assert tracker.trackers == []


def test_detection_below_low_threshold_is_ignored():
    tracker = Tracker(min_hits=1)
    tracker.update([make_det([100, 100, 200, 200], conf=0.9)])

    # conf < track_low_thresh (0.1) belongs to neither pool: the track coasts.
    out = tracker.update([make_det([100, 100, 200, 200], conf=0.05)])
    assert len(out) == 1
    assert out[0].time_since_update == 1  # coasting, not updated
    assert len(tracker.trackers) == 1  # the track itself survives


# --- Confirmed latch + output coast -------------------------------------------


def test_confirmed_track_survives_a_single_missed_frame():
    tracker = Tracker(min_hits=1)  # output_coast defaults to 1
    first = tracker.update([make_det([100, 100, 200, 200], conf=0.9)])

    # One dropped detection frame: track is emitted at its predicted position.
    coasted = tracker.update([])
    assert len(coasted) == 1
    assert coasted[0].id == first[0].id
    assert coasted[0].time_since_update == 1

    # Reappears and re-attaches to the same ID.
    back = tracker.update([make_det([102, 101, 202, 201], conf=0.9)])
    assert len(back) == 1
    assert back[0].id == first[0].id
    assert back[0].time_since_update == 0


def test_confirmation_latch_with_min_hits():
    tracker = Tracker(min_hits=3, max_age=30)

    # Burn the startup grace window (frame_count <= min_hits confirms early).
    for _ in range(3):
        tracker.update([])

    # The spawning detection is hit #1; confirmation needs min_hits detections.
    seen = []
    for i in range(4):
        out = tracker.update([make_det([100 + 5 * i, 100, 200 + 5 * i, 200])])
        seen.append(len(out))
    assert seen == [0, 0, 1, 1]  # tentative until the 3rd detection

    # Latched: after a miss the confirmed track is still emitted immediately.
    assert len(tracker.update([])) == 1
    out = tracker.update([make_det([125, 100, 225, 200])])
    assert len(out) == 1 and out[0].time_since_update == 0


# --- predict_only coast mode (detect-every-N) ---------------------------------


def test_predict_only_coasts_along_velocity_then_expires():
    tracker = Tracker(min_hits=1)
    tracker.update([make_det([100, 100, 200, 200])])
    second = tracker.update([make_det([110, 100, 210, 200])])
    x_prev = second[0].get_center()[0]

    for expected_tsu in (1, 2):
        out = tracker.predict_only(max_coast=2)
        assert len(out) == 1
        assert out[0].time_since_update == expected_tsu
        x_now = out[0].get_center()[0]
        assert x_now > x_prev  # moving forward on Kalman velocity
        x_prev = x_now

    # Past the coast window the track is withheld (but not deleted).
    assert tracker.predict_only(max_coast=2) == []
    assert len(tracker.trackers) == 1


def test_detect_every_two_keeps_id_stable():
    tracker = Tracker(min_hits=1)
    ids = set()
    for step in range(6):
        x = 100 + 10 * step
        if step % 2 == 0:
            out = tracker.update([make_det([x, 100, x + 100, 200])])
        else:
            out = tracker.predict_only(max_coast=1)
        assert len(out) == 1
        ids.add(out[0].id)
    assert len(ids) == 1  # one object, one ID across detect + coast frames
