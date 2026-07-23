"""Tests that the CLI actually wires the safety subsystem into the pipeline.

The safety package was previously reachable only from its own tests; these lock
in that --stall-zone / --intrusion-zone / --debris / --learn-zone / --stabilize
build the right analyzers, a stabilizer, and an alert dispatcher.
"""

from __future__ import annotations

import argparse

from flowcount.cli import build_analytics
from flowcount.safety import (
    CameraStabilizer,
    StabilityMonitor,
    StaticObjectMonitor,
    ZoneIncidentDetector,
    ZoneLearner,
)

PROPS = {"fps": 30.0}


def _args(**overrides):
    """A Namespace with every flag build_analytics reads, defaulted to off."""
    base = dict(
        count_line=None,
        expect_direction=None,
        zone=None,
        dwell=None,
        heatmap=False,
        export_csv=None,
        export_db=None,
        record_events=None,
        stall_zone=None,
        intrusion_zone=None,
        debris=False,
        learn_zone=False,
        stabilize=False,
        alert_webhook=None,
        alert_min_severity="warning",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _types(manager):
    return [type(a) for a in manager.analyzers]


def test_no_flags_builds_nothing():
    manager, stabilizer = build_analytics(_args(), PROPS)
    assert manager is None
    assert stabilizer is None


def test_stall_zone_builds_incident_detector_and_dispatcher():
    manager, stabilizer = build_analytics(
        _args(stall_zone=["100,100,300,100,300,300,100,300"]), PROPS
    )
    try:
        assert stabilizer is None
        assert ZoneIncidentDetector in _types(manager)
        # An incident detector can raise alerts, so a dispatcher is wired.
        assert manager.alert_dispatcher is not None
        det = next(a for a in manager.analyzers if isinstance(a, ZoneIncidentDetector))
        assert det.rules[0].kind == "stalled"
    finally:
        manager.close()


def test_intrusion_zone_uses_intrusion_rule():
    manager, _ = build_analytics(_args(intrusion_zone=["0,0,50,0,50,50,0,50"]), PROPS)
    try:
        det = next(a for a in manager.analyzers if isinstance(a, ZoneIncidentDetector))
        assert det.rules[0].kind == "intrusion"
        assert det.rules[0].trigger == "present"
    finally:
        manager.close()


def test_stabilize_builds_stabilizer_and_suspends_incident_detectors():
    manager, stabilizer = build_analytics(
        _args(stabilize=True, stall_zone=["100,100,300,100,300,300,100,300"]),
        PROPS,
    )
    try:
        assert isinstance(stabilizer, CameraStabilizer)
        monitors = [a for a in manager.analyzers if isinstance(a, StabilityMonitor)]
        assert len(monitors) == 1
        # StabilityMonitor must run before the incident detector it suspends.
        assert isinstance(manager.analyzers[0], StabilityMonitor)
        det = next(a for a in manager.analyzers if isinstance(a, ZoneIncidentDetector))
        assert det in monitors[0].suspends
    finally:
        manager.close()


def test_debris_and_learn_zone_flags():
    manager, _ = build_analytics(_args(debris=True, learn_zone=True), PROPS)
    try:
        types = _types(manager)
        assert StaticObjectMonitor in types
        assert ZoneLearner in types
    finally:
        manager.close()


def test_webhook_adds_a_second_sink_and_respects_min_severity():
    manager, _ = build_analytics(
        _args(
            stall_zone=["100,100,300,100,300,300,100,300"],
            alert_webhook="https://example.test/hook",
            alert_min_severity="critical",
        ),
        PROPS,
    )
    try:
        disp = manager.alert_dispatcher
        assert len(disp.sinks) == 2  # LogSink + WebhookSink
        assert disp.min_rank == 2  # "critical"
    finally:
        manager.close()
