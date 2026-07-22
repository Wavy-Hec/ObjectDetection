"""End-to-end safety tests through the real Pipeline and the real Tracker.

The unit tests drive hand-built FakeTracks, which cannot exhibit the things that
actually break incident logic in the field: Kalman coasting, ID churn, tracks
dying and respawning. These tests run the genuine detect -> track -> analytics
path over the synthetic traffic scene, so the tracker really is doing Hungarian
matching over Kalman predictions.
"""

from flowcount.analytics import AnalyticsManager, Zone
from flowcount.pipeline import Pipeline
from flowcount.safety import ZoneIncidentDetector, stalled_rule
from flowcount.synthetic import HEIGHT, SyntheticTrafficDetector
from flowcount.tracker import Tracker

FPS = 30.0
# A hazard zone across the middle of the road, like a crossing.
CROSSING = Zone("crossing", [(240, 40), (400, 40), (400, HEIGHT - 20), (240, HEIGHT - 20)])


def _run(rules, seconds, *, stop_vehicle=None, detect_every=1):
    """Drive the synthetic scene through the real pipeline.

    Args:
        stop_vehicle: index into ``SyntheticTrafficDetector.vehicles``. That
            vehicle's speed is set to zero the first time it is inside the
            crossing — the test locates that moment itself rather than hardcoding
            a frame number, so it stays correct if the scene is ever retimed.
    """
    detector = SyntheticTrafficDetector(loop=True)
    incidents = ZoneIncidentDetector(rules)
    pipeline = Pipeline(
        detector,
        Tracker(min_hits=3, iou_threshold=0.2, max_age=30),
        analytics_manager=AnalyticsManager([incidents]),
        show_speed=False,
        detect_every=detect_every,
    )

    events = []
    stopped_at = None
    for i in range(int(FPS * seconds)):
        now = i / FPS
        if stop_vehicle is not None and stopped_at is None:
            vehicle = detector.vehicles[stop_vehicle]
            if CROSSING.contains((vehicle[0], vehicle[1])):
                vehicle[2] = 0.0  # speed -> 0: it stalls on the crossing
                stopped_at = now
        frame = detector.render_frame()
        detector.advance()
        events.extend(pipeline.process_frame(frame, timestamp=now).events)
    return events, incidents, stopped_at


def test_flowing_traffic_produces_no_alerts():
    """The false-alarm floor: every vehicle is moving, so nothing should fire.

    Automatic-incident-detection systems are judged on false alarms per camera
    per day, so a rule that fires on ordinary traffic is worthless regardless of
    how reliably it catches real stalls.
    """
    events, _, _ = _run([stalled_rule([CROSSING])], seconds=90)
    assert events == []


def test_a_vehicle_that_stalls_in_the_zone_raises():
    # Vehicle 2 is a car in the bottom lane; it stalls the first time it reaches
    # the crossing and never moves again.
    events, incidents, stopped_at = _run([stalled_rule([CROSSING])], seconds=45, stop_vehicle=2)

    raised = [e for e in events if e.kind == "stalled"]
    assert len(raised) == 1
    assert raised[0].data["zone"] == "crossing"
    assert raised[0].data["severity"] == "critical"
    assert raised[0].class_label == "car"
    assert incidents.stats()["incidents"]["counts"] == {"stalled": 1}

    # Time-to-detect: the 10 s threshold plus the stillness window needed to
    # judge it stopped at all. PIARC's guidance for a stopped-vehicle alarm is
    # ~10 s, so anything past ~20 s here would be a regression worth noticing.
    ttd = raised[0].timestamp - stopped_at
    assert 10.0 <= ttd <= 20.0


def test_a_stall_alert_is_raised_once_not_every_frame():
    """A level-triggered alert would open one endless clip and spam every sink."""
    events, _, _ = _run([stalled_rule([CROSSING])], seconds=45, stop_vehicle=2)
    assert len([e for e in events if e.kind == "stalled"]) == 1


def test_stall_survives_detect_every_coasting():
    """Under frame skipping most frames are Kalman predictions.

    The stall must still be found: stillness is sampled only on
    detection-backed frames, which is what keeps this honest rather than
    reading a coasting filter as a stopped car.
    """
    events, _, _ = _run([stalled_rule([CROSSING])], seconds=45, stop_vehicle=2, detect_every=3)
    assert [e.kind for e in events] == ["stalled"]
