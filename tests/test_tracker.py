"""Tests for the SORT tracker (Kalman filter + Hungarian matching)."""

from src.detector import Detection
from src.tracker import Track, Tracker


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
    tracks = tracker.update([
        make_det([100, 100, 200, 200], "person"),
        make_det([300, 300, 400, 400], "car"),
    ])
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
        id=7, bbox=[0, 0, 10, 20], class_label="person", confidence=0.9,
        age=1, hits=1, time_since_update=0, velocity=(3.0, 4.0),
    )
    assert track.get_center() == (5.0, 10.0)
    assert track.get_speed() == 5.0  # 3-4-5 triangle
