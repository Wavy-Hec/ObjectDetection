"""Tests for camera-motion compensation.

Ground truth is a known affine applied to a static textured image, so accuracy
can be asserted in pixels rather than eyeballed — which is the only way to catch
a silent sign flip or a scale-back error in the downscaling.
"""

import cv2
import numpy as np

from flowcount.safety.stabilize import CameraStabilizer, StabilityMonitor, StabilizerConfig

W, H = 640, 360


def _stab(**overrides):
    """Estimator under test: every frame, unless a test says otherwise.

    The shipped default decimates to every 3rd frame for the edge budget;
    accuracy tests opt out so they measure the estimator rather than the
    scheduler.
    """
    overrides.setdefault("estimate_every", 1)
    return CameraStabilizer(StabilizerConfig(**overrides))


def _scene(seed=0):
    """A textured outdoor-ish scene. Corner detection needs real structure."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 90, (H, W, 3), dtype=np.uint8)
    for _ in range(90):
        x, y = rng.integers(0, W - 40), rng.integers(0, H - 40)
        colour = tuple(int(v) for v in rng.integers(90, 240, 3))
        cv2.rectangle(img, (x, y), (x + rng.integers(6, 34), y + rng.integers(6, 34)), colour, -1)
    for _ in range(40):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 12)), (220, 220, 220), 1)
    return img


def _shift(img, tx, ty, angle=0.0):
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REFLECT)


def _corner_error(matrix, tx, ty, angle=0.0):
    """Max reprojection error at the frame corners, in pixels.

    The stabilizer reports current -> reference, so applying it to a shifted
    frame's corners should land back on the original corners.
    """
    truth = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    truth[0, 2] += tx
    truth[1, 2] += ty
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    moved = cv2.transform(corners.reshape(-1, 1, 2), truth).reshape(-1, 2)
    recovered = cv2.transform(moved.reshape(-1, 1, 2), matrix[:2]).reshape(-1, 2)
    return float(np.abs(recovered - corners).max())


def test_identity_on_the_first_frame():
    s = _stab()
    m = s.estimate(_scene())
    assert np.allclose(m, np.eye(3))
    assert not s.state.degraded


def test_recovers_pure_translation():
    scene = _scene()
    s = _stab()
    s.estimate(scene)
    m = s.estimate(_shift(scene, 12, -7))
    assert m is not None
    assert _corner_error(m, 12, -7) < 1.0


def test_recovers_translation_and_roll():
    """A pole rolls as well as translating; a translation-only model would miss it."""
    scene = _scene()
    s = _stab()
    s.estimate(scene)
    m = s.estimate(_shift(scene, 8, 4, angle=1.5))
    assert m is not None
    assert _corner_error(m, 8, 4, angle=1.5) < 2.0


def test_keyframe_matching_does_not_drift():
    """The reason for matching a keyframe rather than chaining frames.

    Chained pairwise transforms are a random walk with no restoring force;
    over a few hundred frames of sway the error grows without bound.
    """
    scene = _scene()
    s = _stab()
    s.estimate(scene)

    errors = []
    for i in range(120):
        tx = 9.0 * np.sin(i * 0.15)
        ty = 4.0 * np.cos(i * 0.11)
        m = s.estimate(_shift(scene, tx, ty), timestamp=i / 30.0)
        if m is not None:
            errors.append(_corner_error(m, tx, ty))

    assert len(errors) > 100
    # Flat, not growing: the last third is no worse than the first third.
    assert np.mean(errors[-40:]) < np.mean(errors[:40]) + 0.5
    assert max(errors) < 2.0


def test_moving_objects_do_not_drag_the_estimate():
    """A queue of vehicles moving together is a large, perfectly consistent
    inlier set. Unmasked, RANSAC happily fits *their* motion and calls it
    camera motion — and it fails silently."""
    scene = _scene()
    s = _stab()
    s.estimate(scene)

    # Camera perfectly still; a large bright object sweeps across the frame.
    worst = 0.0
    for i in range(20):
        frame = scene.copy()
        x = 40 + i * 22
        cv2.rectangle(frame, (x, 120), (x + 190, 250), (250, 250, 250), -1)
        cv2.rectangle(frame, (x + 20, 140), (x + 90, 190), (20, 20, 20), -1)
        boxes = [[x, 120, x + 190, 250]]
        m = s.estimate(frame, exclude_boxes=boxes, timestamp=i / 30.0)
        if m is not None:
            worst = max(worst, _corner_error(m, 0, 0))

    assert worst < 1.5  # the camera did not move, and we agree


def test_degraded_estimate_returns_none_rather_than_a_guess():
    """A wrong matrix is worse than none: analytics would apply it silently."""
    scene = _scene()
    s = _stab()
    s.estimate(scene)
    m = s.estimate(np.zeros((H, W, 3), dtype=np.uint8))  # lens blocked
    assert m is None
    assert s.state.degraded


def test_downscale_translation_is_scaled_back():
    """The fit happens at reduced resolution; forgetting to rescale the
    translation is an error that stays invisible until zones drift."""
    scene = _scene()
    for downscale in (1, 2, 4):
        s = _stab(downscale=downscale)
        s.estimate(scene)
        m = s.estimate(_shift(scene, 16, -10))
        assert m is not None, downscale
        assert _corner_error(m, 16, -10) < 2.0, downscale


def test_transient_gust_does_not_rebaseline():
    scene = _scene()
    s = _stab()
    s.estimate(scene, timestamp=0.0)
    for i in range(60):
        tx = 10.0 * np.sin(i * 0.4)
        s.estimate(_shift(scene, tx, 0), timestamp=i / 30.0)
    assert s.pending_events == []


def test_sustained_reposition_is_reported():
    """A knocked camera invalidates operator-drawn geometry; say so."""
    scene = _scene()
    s = _stab(reposition_hold_s=0.5)
    s.estimate(scene, timestamp=0.0)
    for i in range(1, 60):
        s.estimate(_shift(scene, 60, 30), timestamp=i / 10.0)

    assert any(e["kind"] == "camera_moved" for e in s.pending_events)


def test_monitor_emits_events_and_suspends_detectors(make_ctx):
    class Detector:
        def __init__(self):
            self.suspended_until = None

        def suspend(self, until_ts):
            self.suspended_until = until_ts

    scene = _scene()
    s = _stab(reposition_hold_s=0.5)
    detector = Detector()
    monitor = StabilityMonitor(s, suspends=[detector])

    s.estimate(scene, timestamp=0.0)
    for i in range(1, 60):
        s.estimate(_shift(scene, 60, 30), timestamp=i / 10.0)

    events = monitor.update(make_ctx([], timestamp=6.0))
    assert [e.kind for e in events] == ["camera_moved"]
    assert events[0].data["requires_human_review"] is True
    assert detector.suspended_until is not None
    # Drained, so it is reported once rather than every frame.
    assert monitor.update(make_ctx([], timestamp=6.1)) == []


def test_monitor_stats():
    s = _stab()
    s.estimate(_scene())
    stats = StabilityMonitor(s).stats()["camera"]
    assert stats["stabilized"] is True
    assert "drift_px" in stats


def test_decimation_reuses_the_last_transform_and_stays_accurate():
    """Sway is ~1 Hz, so a 100 ms-old transform is still sub-pixel.

    This is the cheapest performance lever available (9.4 -> 3.1 ms/frame at
    720p), so it needs to be shown not to cost accuracy at realistic sway.
    """
    scene = _scene()
    s = CameraStabilizer(StabilizerConfig(estimate_every=3))
    s.estimate(scene, timestamp=0.0)

    errors = []
    matrices = []
    for i in range(1, 90):
        tx, ty = 8.0 * np.sin(i * 0.12), 3.0 * np.cos(i * 0.09)
        m = s.estimate(_shift(scene, tx, ty), timestamp=i / 30.0)
        assert m is not None
        matrices.append(m)
        errors.append(_corner_error(m, tx, ty))

    # Consecutive frames often share a matrix object: that is the reuse.
    reused = sum(1 for a, b in zip(matrices, matrices[1:], strict=False) if a is b)
    assert reused > len(matrices) // 3
    # And it costs no meaningful accuracy at realistic sway.
    assert max(errors) < 3.0
