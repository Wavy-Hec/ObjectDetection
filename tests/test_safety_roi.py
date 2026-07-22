"""Tests for automatic zone-of-interest discovery.

Traffic is simulated over a known road band so the learned polygon can be
scored against ground truth by IoU rather than inspected by eye.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from flowcount.safety.roi import ZoneLearner, ZoneLearnerConfig

W, H = 640, 360
ROAD_TOP, ROAD_BOTTOM = 140, 260  # ground-truth drivable band


def _track(track_id, cx, cy, w=60, h=34):
    from conftest import FakeTrack

    return FakeTrack(track_id, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "car")


def _frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


def _drive(learner, make_ctx, n_vehicles=60, *, lanes=(165, 205, 240), transform=None):
    """Send vehicles left-to-right along the road band."""
    ts = 0.0
    frame_index = 0
    for v in range(n_vehicles):
        lane = lanes[v % len(lanes)]
        for step in range(24):
            frame_index += 1
            ts += 1 / 30.0
            cx = -40 + step * 30
            learner.update(
                make_ctx(
                    [_track(1000 + v, cx, lane)],
                    frame_index=frame_index,
                    timestamp=ts,
                    frame=_frame(),
                    transform=transform,
                )
            )


def _road_iou(polygon):
    truth = np.zeros((H, W), np.uint8)
    truth[ROAD_TOP:ROAD_BOTTOM, :] = 255
    got = np.zeros((H, W), np.uint8)
    cv2.fillPoly(got, [np.array(polygon, np.int32)], 255)
    inter = np.count_nonzero(truth & got)
    union = np.count_nonzero(truth | got)
    return inter / union if union else 0.0


def test_learns_the_road_band(make_ctx):
    learner = ZoneLearner()
    _drive(learner, make_ctx)

    polygon = learner.proposal
    assert polygon is not None
    assert len(polygon) >= 4
    # The band is the whole width and ~1/3 of the height; a proposal that
    # merely covers the frame would score far lower than this.
    assert _road_iou(polygon) > 0.7


def test_footprints_cover_more_than_the_centre_corridor(make_ctx):
    """Why footprints rather than track centres: a vehicle's centre never
    reaches the lane edge, so centres describe a corridor narrower than the
    road and the learned zone would systematically under-cover it."""
    learner = ZoneLearner()
    _drive(learner, make_ctx, lanes=(200,))  # single lane, one vehicle height

    ys = [y for _, y in learner.proposal]
    # A 34 px-tall vehicle centred on y=200 should give a band taller than the
    # zero-height line its centres trace.
    assert max(ys) - min(ys) > 25


def test_parked_vehicles_do_not_define_the_road(make_ctx):
    """A car sitting in a lay-by is not evidence of drivable surface."""
    learner = ZoneLearner()
    for i in range(400):
        learner.update(
            make_ctx(
                [_track(1, 500, 60)],  # never moves, well off the road band
                frame_index=i + 1,
                timestamp=i / 30.0,
                frame=_frame(),
            )
        )
    assert learner.proposal is None
    assert learner.stats()["zone_learning"]["tracks_observed"] == 0


def test_flickering_detections_do_not_qualify(make_ctx):
    """An object jittering in place travels far but goes nowhere."""
    learner = ZoneLearner()
    rng = np.random.default_rng(0)
    for i in range(400):
        cx, cy = 300 + rng.normal(0, 25), 200 + rng.normal(0, 25)
        learner.update(
            make_ctx([_track(1, cx, cy)], frame_index=i + 1, timestamp=i / 30.0, frame=_frame())
        )
    # Displacement may pass, but direction consistency will not.
    assert learner.stats()["zone_learning"]["tracks_observed"] == 0


def test_proposal_is_not_offered_until_it_has_converged(make_ctx):
    learner = ZoneLearner()
    _drive(learner, make_ctx, n_vehicles=8)
    assert not learner.stable

    _drive(learner, make_ctx, n_vehicles=80)
    assert learner.stable


def test_proposal_event_is_emitted_once(make_ctx):
    learner = ZoneLearner()
    events = []
    ts = 0.0
    idx = 0
    for v in range(120):
        lane = (165, 205, 240)[v % 3]
        for step in range(24):
            idx += 1
            ts += 1 / 30.0
            events.extend(
                learner.update(
                    make_ctx(
                        [_track(2000 + v, -40 + step * 30, lane)],
                        frame_index=idx,
                        timestamp=ts,
                        frame=_frame(),
                    )
                )
            )

    proposals = [e for e in events if e.kind == "zone_proposal"]
    assert len(proposals) == 1
    assert proposals[0].data["requires_human_review"] is True
    assert len(proposals[0].data["polygon"]) >= 4


def test_camera_drift_does_not_smear_the_learned_road(make_ctx):
    """Learning happens in the stabilizer's reference frame."""
    drifting = ZoneLearner()
    stabilized = ZoneLearner()

    ts, idx = 0.0, 0
    for v in range(60):
        lane = (165, 205, 240)[v % 3]
        for step in range(24):
            idx += 1
            ts += 1 / 30.0
            # Drift ACROSS the road, not along it. Sway parallel to a
            # full-width band is unobservable — it is the perpendicular
            # component that smears the learned extent.
            sway = 30.0 * np.sin(idx * 0.01)
            matrix = np.array([[1, 0, 0], [0, 1, -sway], [0, 0, 1]], dtype=np.float64)
            track = _track(3000 + v, -40 + step * 30, lane + sway)
            drifting.update(make_ctx([track], frame_index=idx, timestamp=ts, frame=_frame()))
            stabilized.update(
                make_ctx([track], frame_index=idx, timestamp=ts, frame=_frame(), transform=matrix)
            )

    assert _road_iou(stabilized.proposal) > _road_iou(drifting.proposal)


def test_cli_argument_round_trips(make_ctx):
    learner = ZoneLearner()
    _drive(learner, make_ctx)
    spec = learner.cli_argument()
    nums = [float(v) for v in spec.split(",")]
    assert len(nums) % 2 == 0
    assert len(nums) >= 8


def test_as_zone_and_save(make_ctx, tmp_path):
    learner = ZoneLearner()
    _drive(learner, make_ctx)

    zone = learner.as_zone("crossing")
    assert zone.name == "crossing"
    assert zone.contains((320, 200))  # middle of the road band
    assert not zone.contains((320, 20))  # sky

    path = learner.save(str(tmp_path / "run"))
    payload = json.loads(Path(path).read_text())
    assert payload["stable"] is True
    assert payload["cli"] == learner.cli_argument()
    assert (tmp_path / "run_zone_density.jpg").exists()


def test_nothing_learned_yields_no_artifacts(make_ctx):
    learner = ZoneLearner(ZoneLearnerConfig())
    assert learner.proposal is None
    assert learner.cli_argument() is None
    assert learner.as_zone() is None
    assert learner.save("/tmp/unused") is None
    learner.draw(_frame())  # must not raise
