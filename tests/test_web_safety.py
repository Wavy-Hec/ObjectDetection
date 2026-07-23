"""The synthetic dashboard scene must actually raise (and surface) a stall.

This is the end-to-end payoff of the synthetic scene: a stalled car parked in
the crossing drives the real Pipeline + ZoneIncidentDetector to a STALLED
incident, and that incident shows up in the dashboard stats — no model, camera,
or GPU. Driven with explicit timestamps so it is deterministic and fast rather
than depending on wall-clock ticks.
"""

from __future__ import annotations

from flowcount.web.server import _build_synthetic_engine


def _drive(engine, seconds=14.0, dt=0.05):
    """Run the synthetic pipeline for a simulated duration; collect fired events."""
    fired = []
    t = 0.0
    last_result = None
    steps = int(seconds / dt)
    for _ in range(steps):
        frame = engine.frame_provider()
        last_result = engine.pipeline.process_frame(frame, timestamp=t)
        fired.extend(last_result.events)
        t += dt
    return fired, last_result


def test_synthetic_scene_raises_and_surfaces_a_stalled_incident():
    engine = _build_synthetic_engine()
    fired, result = _drive(engine)

    # A stalled incident fired at some point.
    assert any(e.kind.startswith("stalled") for e in fired), (
        "expected a stalled incident from the parked car in the crossing"
    )

    # And it is still open and visible in the dashboard stats at the end (the
    # car never moves, so the incident does not clear).
    stats = engine._build_stats(result)
    assert "safety" in stats
    incidents = stats["safety"].get("incidents", {})
    assert incidents.get("counts", {}).get("stalled", 0) >= 1
    open_kinds = {o["kind"] for o in incidents.get("open", [])}
    assert "stalled" in open_kinds


def test_warming_stats_include_a_safety_slot():
    engine = _build_synthetic_engine()
    warming = engine._warming_stats()
    assert warming["safety"] == {}
