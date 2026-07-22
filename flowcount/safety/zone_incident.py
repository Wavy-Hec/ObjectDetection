"""Zone incident detection: stuck objects, intrusions, and congestion.

One rule-parameterised analyzer rather than three near-identical ones. Stalled
vehicles, pedestrians-in-zone and queues differ only in *what counts as a
trigger* and *how long it must persist*; the association, hysteresis, occlusion
handling and cooldown logic is identical and is the part that is easy to get
subtly wrong. Sharing it means fixing a bug once.

The alert thresholds are not invented. PIARC's automatic-incident-detection
guidance puts time-to-detect for a stopped vehicle at ~10 s with a false-alarm
budget around 0.025 alarms/camera/day — roughly one false alert per camera per
six weeks. Frigate's stationary-object default is also 10 s. That false-alarm
budget, not raw detection accuracy, is what the confirmation logic here buys:
sustained-trigger windows, a confirmation count, hysteresis in both directions,
and per-anchor cooldown.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2

from ..analytics.base import Analyzer, Event, FrameContext
from ..analytics.zones import Zone
from .incidents import IncidentTracker, Observation
from .motion import StillnessConfig, TrackMotionWindow, evaluate_stillness, transform_bbox

#: COCO classes that count as vehicles for the default stalled rule.
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
#: The default intrusion class. Deliberately just pedestrians: on an open road
#: a motorcycle in the carriageway is ordinary traffic, and alerting on it turns
#: the feed into noise. Pass ``classes=VULNERABLE_CLASSES`` for a zone where
#: two-wheelers genuinely do not belong.
PEDESTRIAN_CLASSES = frozenset({"person"})
#: The wider "vulnerable road user" set used by tunnel/AID products.
VULNERABLE_CLASSES = frozenset({"person", "bicycle", "motorcycle"})

SEVERITY_COLORS = {
    "info": (200, 200, 60),
    "warning": (0, 165, 255),
    "critical": (0, 0, 255),
}


@dataclass
class IncidentRule:
    """One safety rule: what to watch for, where, and for how long."""

    kind: str = "stalled"
    #: ``"still"`` triggers on an object that has stopped; ``"present"`` triggers
    #: on mere presence in the zone (the right semantics for an intrusion).
    trigger: str = "still"
    zones: Sequence[Zone] = ()
    classes: frozenset | set | None = None
    t_raise: float = 10.0
    t_escalate: tuple[float, ...] = (60.0, 300.0)
    t_clear: float = 3.0
    severity: str = "warning"
    occlusion_grace_s: float = 8.0
    cooldown_s: float = 30.0
    confirmations: int = 3
    stillness: StillnessConfig = field(default_factory=StillnessConfig)
    #: When at least this many objects are simultaneously triggering in the same
    #: zone, report the incident as congestion instead. A stalled vehicle in
    #: free flow is a point anomaly; a queue is a collective one, and paging a
    #: dispatcher for rush hour is the fastest way to get an alert feed ignored.
    congestion_peers: int | None = None
    congestion_kind: str = "congestion"
    congestion_severity: str = "info"

    def zone_for(self, point) -> tuple[bool, str | None]:
        """(applies, zone name) for a point. No zones configured = whole frame."""
        if not self.zones:
            return True, None
        for zone in self.zones:
            if zone.contains(point):
                return True, zone.name
        return False, None


def stalled_rule(zones: Sequence[Zone] = (), **overrides) -> IncidentRule:
    """A vehicle stopped in the hazard zone — the grade-crossing headline case."""
    params: dict[str, Any] = {
        "kind": "stalled",
        "trigger": "still",
        "zones": zones,
        "classes": VEHICLE_CLASSES,
        "t_raise": 10.0,
        "severity": "critical",
        "congestion_peers": 3,
    }
    params.update(overrides)
    return IncidentRule(**params)


def intrusion_rule(zones: Sequence[Zone] = (), **overrides) -> IncidentRule:
    """A person on foot in the hazard zone.

    Deliberately trades false-alarm rate for latency: 1.5 s rather than 10 s,
    because the cost asymmetry runs the other way from a stalled vehicle.

    The occlusion grace is short. For a stalled vehicle, vanishing usually means
    a truck drove in front of it and the hazard is still there, so the incident
    must be held. For a presence trigger, walking out of the zone *is* the exit
    condition and the normal one — holding it for the stalled rule's 8 s would
    keep a stale incident open long enough to absorb the next person to walk
    through the same spot.
    """
    params: dict[str, Any] = {
        "kind": "intrusion",
        "trigger": "present",
        "zones": zones,
        "classes": PEDESTRIAN_CLASSES,
        "t_raise": 1.5,
        "t_escalate": (15.0,),
        "t_clear": 1.0,
        "occlusion_grace_s": 1.5,
        "cooldown_s": 5.0,
        "severity": "critical",
        "confirmations": 3,
    }
    params.update(overrides)
    return IncidentRule(**params)


class ZoneIncidentDetector(Analyzer):
    """Runs a set of :class:`IncidentRule` s over the tracks of each frame."""

    #: Drop a track's motion window this long after it was last seen.
    PRUNE_AFTER_S = 10.0

    def __init__(self, rules: Sequence[IncidentRule]):
        self.rules = list(rules)
        self._trackers = [
            IncidentTracker(
                rule.kind,
                t_raise=rule.t_raise,
                t_escalate=rule.t_escalate,
                t_clear=rule.t_clear,
                occlusion_grace_s=rule.occlusion_grace_s,
                cooldown_s=rule.cooldown_s,
                confirmations=rule.confirmations,
            )
            for rule in self.rules
        ]
        self._windows: dict[int, TrackMotionWindow] = {}
        self._window_last_ts: dict[int, float] = {}
        # Per (rule index, track id) latched stillness, for the Schmitt trigger.
        self._still_state: dict[tuple[int, int], bool] = {}
        self._suspend_until_ts: float = float("-inf")
        self._now: float = 0.0

    # ---- external control ---------------------------------------------------
    def suspend(self, until_ts: float) -> None:
        """Freeze all state transitions until ``until_ts``.

        Called after a camera bump: a gust that shifts the whole image looks
        like every object moving at once, which would both manufacture alerts
        and — more dangerously — falsely clear real ones.
        """
        self._suspend_until_ts = max(self._suspend_until_ts, until_ts)

    @property
    def suspended(self) -> bool:
        return self._now < self._suspend_until_ts

    # ---- Analyzer -----------------------------------------------------------
    def update(self, ctx: FrameContext) -> list[Event]:
        now = ctx.timestamp
        self._now = now
        self._refresh_windows(ctx)

        events: list[Event] = []
        for idx, (rule, tracker) in enumerate(zip(self.rules, self._trackers, strict=True)):
            observations = self._observe_rule(idx, rule, ctx)
            triggering_per_zone: dict[str | None, int] = {}
            for obs in observations:
                if obs.still:
                    triggering_per_zone[obs.zone] = triggering_per_zone.get(obs.zone, 0) + 1

            for update in tracker.observe(now, observations, suspended=self.suspended):
                inc = update.incident
                if update.action == "raise":
                    inc.peers = triggering_per_zone.get(inc.zone, 1)
                    if (
                        rule.congestion_peers is not None
                        and inc.peers >= rule.congestion_peers
                        and rule.trigger == "still"
                    ):
                        inc.kind = rule.congestion_kind
                events.append(self._to_event(rule, update, ctx))

        return events

    def _refresh_windows(self, ctx: FrameContext) -> None:
        """Update every track's motion window in stabilized coordinates."""
        for track in ctx.tracks:
            # Metrics are computed in the stabilizer's reference frame so that
            # camera sway does not read as object motion. With no stabilizer
            # this is the identity.
            bbox = transform_bbox(track.bbox, ctx.transform)
            window = self._windows.get(track.id)
            if window is None:
                window = TrackMotionWindow(window_s=self._max_window_s())
                self._windows[track.id] = window
            window.add(
                ctx.timestamp,
                bbox,
                getattr(track, "confidence", 1.0),
                getattr(track, "time_since_update", 0) == 0,
                frame_index=ctx.frame_index,
                detector_ran=ctx.detection_ran,
            )
            self._window_last_ts[track.id] = ctx.timestamp

        stale = [
            tid
            for tid, ts in self._window_last_ts.items()
            if ctx.timestamp - ts > self.PRUNE_AFTER_S
        ]
        for tid in stale:
            self._windows.pop(tid, None)
            self._window_last_ts.pop(tid, None)
            for key in [k for k in self._still_state if k[1] == tid]:
                self._still_state.pop(key, None)

    def _max_window_s(self) -> float:
        return max((r.stillness.window_s for r in self.rules), default=3.0)

    def _observe_rule(self, idx: int, rule: IncidentRule, ctx: FrameContext) -> list[Observation]:
        observations: list[Observation] = []
        for track in ctx.tracks:
            if rule.classes and track.class_label not in rule.classes:
                continue
            applies, zone_name = rule.zone_for(track.get_center())
            if not applies:
                continue

            window = self._windows.get(track.id)
            if window is None:
                continue

            key = (idx, track.id)
            if rule.trigger == "present":
                still: bool | None = True
            else:
                previous = self._still_state.get(key, False)
                still = evaluate_stillness(window, rule.stillness, currently_still=previous)
                if still is not None:
                    self._still_state[key] = still

            observations.append(
                Observation(
                    track_id=track.id,
                    bbox=transform_bbox(track.bbox, ctx.transform),
                    class_label=track.class_label,
                    still=still,
                    zone=zone_name,
                    display_bbox=tuple(float(v) for v in track.bbox),
                )
            )
        return observations

    def _to_event(self, rule: IncidentRule, update, ctx: FrameContext) -> Event:
        inc = update.incident
        severity = rule.severity
        if inc.kind == rule.congestion_kind and rule.congestion_peers is not None:
            severity = rule.congestion_severity

        kind = inc.kind
        if update.action == "escalate":
            kind = f"{inc.kind}_escalate"
            severity = "critical"
        elif update.action == "clear":
            kind = f"{inc.kind}_clear"
            severity = "info"

        data: dict[str, Any] = {
            "incident_id": inc.id,
            "severity": severity,
            "zone": inc.zone,
            "duration_s": round(inc.duration_s(ctx.timestamp), 2),
            "bbox": [round(float(v), 1) for v in (inc.display_bbox or inc.bbox)],
            "track_ids": sorted(inc.contributing_track_ids),
        }
        if update.action == "raise":
            data["peers"] = inc.peers
        if update.action == "escalate":
            data["level"] = update.level
        if update.action == "clear":
            data["reason"] = update.reason

        return Event(
            kind=kind,
            # The originating track. Contributing IDs are all recorded in data:
            # an incident routinely outlives the track that started it.
            track_id=min(inc.contributing_track_ids, default=-1),
            class_label=inc.class_label,
            frame_index=ctx.frame_index,
            timestamp=ctx.timestamp,
            data=data,
        )

    # ---- presentation -------------------------------------------------------
    def draw(self, frame) -> None:
        for rule, tracker in zip(self.rules, self._trackers, strict=True):
            for zone in rule.zones:
                pts = _polygon_points(zone)
                cv2.polylines(frame, [pts], True, (120, 120, 120), 1)

            for inc in tracker.open_incidents:
                box = inc.display_bbox or inc.bbox
                x1, y1, x2, y2 = (int(v) for v in box)
                color = SEVERITY_COLORS.get(rule.severity, (0, 165, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{inc.kind.upper()} {inc.duration_s(self._now):.0f}s"
                cv2.putText(
                    frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )

    def stats(self) -> dict:
        open_incidents = []
        counts: dict[str, int] = {}
        for tracker in self._trackers:
            for inc in tracker.open_incidents:
                counts[inc.kind] = counts.get(inc.kind, 0) + 1
                open_incidents.append(
                    {
                        "id": inc.id,
                        "kind": inc.kind,
                        "zone": inc.zone,
                        "class": inc.class_label,
                        "duration_s": round(inc.duration_s(self._now), 1),
                    }
                )
        return {
            "incidents": {
                "open": open_incidents,
                "counts": counts,
                "suspended": self.suspended,
            }
        }


def _polygon_points(zone: Zone):
    import numpy as np

    return np.array(zone.polygon, dtype=np.int32)
