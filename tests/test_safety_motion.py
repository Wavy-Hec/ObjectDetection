"""Tests for the stillness metrics."""

import numpy as np

from flowcount.safety.motion import (
    StillnessConfig,
    TrackMotionWindow,
    anchor_of,
    evaluate_stillness,
    iou,
    size_of,
    transform_bbox,
)


def _box(cx, cy, w=40, h=40):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _fill(window, centers, *, start=0.0, dt=0.1, conf=0.9, backed=True, size=40):
    for i, (cx, cy) in enumerate(centers):
        window.add(start + i * dt, _box(cx, cy, size, size), conf, backed)
    return window


def test_anchor_is_bottom_centre_not_centroid():
    """The ground-contact point survives the box top being clipped."""
    assert anchor_of((10, 20, 50, 60)) == (30.0, 60.0)


def test_size_and_iou_helpers():
    assert size_of((0, 0, 10, 40)) == 20.0
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_stationary_track_is_still():
    w = _fill(TrackMotionWindow(3.0), [(100, 100)] * 30)
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is True


def test_jitter_still_reads_as_still():
    """Detector jitter of a few px on a 40 px object must not defeat a stall."""
    rng = np.random.default_rng(0)
    centers = [(100 + rng.normal(0, 1.2), 100 + rng.normal(0, 1.2)) for _ in range(30)]
    w = _fill(TrackMotionWindow(3.0), centers)
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is True


def test_moving_track_is_not_still():
    w = _fill(TrackMotionWindow(3.0), [(100 + i * 6, 100) for i in range(30)])
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is False


def test_slow_crawl_is_not_still():
    """Self-IoU saturates for a crawl; the normalised radius is what catches it."""
    w = _fill(TrackMotionWindow(3.0), [(100 + i * 0.9, 100) for i in range(30)])
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is False


def test_coasted_track_is_unjudgeable_not_moving():
    """Kalman-predicted positions cannot support a stillness verdict.

    Returning None (rather than False) is what lets a caller freeze a stall
    timer through an occlusion instead of resetting a nearly-mature one.
    """
    w = TrackMotionWindow(3.0)
    for i in range(30):
        w.add(i * 0.1, _box(100, 100), 0.9, detection_backed=i < 10)  # 33% support
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is None


def test_short_window_is_unjudgeable():
    w = _fill(TrackMotionWindow(3.0), [(100, 100)] * 3)
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is None


def test_hysteresis_keeps_a_still_object_still_through_a_nudge():
    """Enter needs IoU > 0.85; a nudge to ~0.75 must not flip the state back."""
    cfg = StillnessConfig()
    w = _fill(TrackMotionWindow(3.0), [(100, 100)] * 20)
    _fill(w, [(105, 100)] * 10, start=2.0)
    assert evaluate_stillness(w, cfg, currently_still=True) is True
    assert evaluate_stillness(w, cfg, currently_still=False) is False


def test_tiny_and_low_confidence_objects_are_not_judged_still():
    cfg = StillnessConfig()
    small = _fill(TrackMotionWindow(3.0), [(100, 100)] * 30, size=10)
    assert evaluate_stillness(small, cfg, currently_still=False) is None

    faint = _fill(TrackMotionWindow(3.0), [(100, 100)] * 30, conf=0.2)
    assert evaluate_stillness(faint, cfg, currently_still=False) is None


def test_normalized_radius_is_scale_invariant():
    """Same real motion at two apparent scales must give the same metric.

    This is what makes one threshold work across the whole frame without any
    camera calibration.
    """
    near = TrackMotionWindow(3.0)
    far = TrackMotionWindow(3.0)
    for i in range(20):
        near.add(i * 0.1, _box(100 + i * 4, 100, 80, 80), 0.9, True)
        far.add(i * 0.1, _box(100 + i * 1, 100, 20, 20), 0.9, True)  # 1/4 scale, 1/4 motion
    assert abs(near.normalized_radius() - far.normalized_radius()) < 0.02


def test_transform_bbox_maps_through_a_translation():
    matrix = np.array([[1, 0, 10], [0, 1, -5], [0, 0, 1]], dtype=np.float64)
    assert transform_bbox((0, 0, 10, 10), matrix) == (10.0, -5.0, 20.0, 5.0)


def test_transform_bbox_is_identity_when_no_matrix():
    assert transform_bbox((1, 2, 3, 4), None) == (1.0, 2.0, 3.0, 4.0)


def test_transform_bbox_survives_a_degenerate_matrix():
    """A bad homography must not poison every downstream metric with NaN."""
    bad = np.zeros((3, 3), dtype=np.float64)
    assert transform_bbox((1, 2, 3, 4), bad) == (1.0, 2.0, 3.0, 4.0)


def test_stabilized_coordinates_cancel_camera_sway():
    """Under sway, raw pixels look like motion; stabilized ones do not."""
    cfg = StillnessConfig()
    raw = TrackMotionWindow(3.0)
    stabilized = TrackMotionWindow(3.0)
    for i in range(30):
        sway = 14.0 * np.sin(i * 0.4)  # camera drifts; object does not
        raw.add(i * 0.1, _box(100 + sway, 100), 0.9, True)
        # A stabilizer reports image -> reference, i.e. it subtracts the sway.
        matrix = np.array([[1, 0, -sway], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        stabilized.add(i * 0.1, transform_bbox(_box(100 + sway, 100), matrix), 0.9, True)

    assert evaluate_stillness(raw, cfg, currently_still=False) is False
    assert evaluate_stillness(stabilized, cfg, currently_still=False) is True


def test_detection_support_is_measured_against_opportunities_not_frames():
    """Frame skipping must not silently disable stillness judgement.

    Under detect_every=N only 1/N of frames carry a detection at all, so a
    support ratio computed over *all* frames could never exceed 1/N — and a
    0.6 threshold would make every stall undetectable at exactly the setting an
    edge device runs. Support is therefore observed/opportunities.
    """
    w = TrackMotionWindow(3.0)
    for i in range(30):
        detector_ran = i % 3 == 0  # detect_every=3
        w.add(
            i * 0.1, _box(100, 100), 0.9, detection_backed=detector_ran, detector_ran=detector_ran
        )

    assert w.detection_support == 1.0  # every opportunity was taken
    assert len(w.observed) == 10
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is True


def test_missed_detections_do_lower_support():
    """A genuinely occluded object still becomes unjudgeable."""
    w = TrackMotionWindow(3.0)
    for i in range(30):
        detector_ran = i % 3 == 0
        # The detector ran 10 times but only found the object 3 times.
        w.add(
            i * 0.1,
            _box(100, 100),
            0.9,
            detection_backed=detector_ran and i < 9,
            detector_ran=detector_ran,
        )

    assert w.detection_support < 0.6
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is None


def test_metrics_ignore_coasted_positions_entirely():
    """A coasting Kalman box must not be able to fake motion or stillness."""
    w = TrackMotionWindow(3.0)
    for i in range(30):
        observed = i % 3 == 0
        # Observed samples all say "parked"; coasted ones drift away fast.
        cx = 100 if observed else 100 + i * 20
        w.add(i * 0.1, _box(cx, 100), 0.9, detection_backed=observed, detector_ran=observed)

    assert w.normalized_radius() < 0.01  # the coasted drift is invisible
    assert evaluate_stillness(w, StillnessConfig(), currently_still=False) is True
