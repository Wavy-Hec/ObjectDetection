"""Incident lifecycle: raise, escalate, clear — keyed on place, not on track ID.

The central decision here is that an incident is anchored to a **location**,
not to a ``track.id``. A SORT tracker reassigns IDs after an occlusion, and
FlowCount's class-aware matching (``tracker.py``) additionally spawns a fresh ID
whenever a detection's class flips (car -> truck on a partially occluded box).
If a 9.9-second stall timer lives on the track, any of those events silently
resets it to zero and the alert never fires — the failure is invisible, which
makes it the worst kind.

So an :class:`Incident` owns ``(anchor, anchor_size, first_still_ts)`` and
tracks *associate to it*: any still track overlapping the anchor adopts the
incident's start time regardless of its own ID.

The other rule that matters: **absence is not clearance.** If a truck occludes a
stalled car, no track associates for a few seconds. Treating that as "cleared"
would drop a live hazard. The incident holds, with its timer still running, for
an occlusion grace window. Symmetrically, clearing demands both sustained motion
*and* a full-object-length displacement, because jitter or a camera bump could
otherwise silently clear a hazard that is still sitting there — a much worse
failure than a false alarm.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from .motion import anchor_of, iou, size_of

# Incident states.
CANDIDATE = "candidate"  # still, but not yet long enough to alert
ACTIVE = "active"  # alert raised
CLEARING = "clearing"  # moving again / absent, but not yet confirmed clear
CLEARED = "cleared"  # terminal


def new_incident_id(timestamp: float) -> str:
    """Sortable, human-decodable, collision-safe id, e.g. ``1753142881.968383-i6ygli``.

    Time-ordered so a lexicographic sort of ids is a chronological sort, which
    makes log files and outbox tables browsable without parsing.
    """
    return f"{timestamp:.6f}-{secrets.token_hex(3)}"


@dataclass
class Observation:
    """One track's contribution to one rule this frame.

    ``still`` is three-valued: ``None`` means "seen, but not judgeable" (usually
    an occlusion or a run of coasted frames), which holds an incident's timers
    rather than advancing or resetting them.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    class_label: str
    still: bool | None
    zone: str | None = None
    display_bbox: tuple[float, float, float, float] | None = None


@dataclass
class Incident:
    id: str
    kind: str
    zone: str | None
    anchor: tuple[float, float]
    anchor_size: float
    bbox: tuple[float, float, float, float]
    class_label: str
    state: str
    first_trigger_ts: float
    last_trigger_ts: float
    last_seen_ts: float
    confirmations: int = 0
    contributing_track_ids: set = field(default_factory=set)
    escalated_levels: set = field(default_factory=set)
    clearing_since_ts: float | None = None
    display_bbox: tuple[float, float, float, float] | None = None
    peers: int = 0

    def duration_s(self, now: float) -> float:
        return max(0.0, now - self.first_trigger_ts)


@dataclass
class IncidentUpdate:
    """A state transition worth telling the outside world about."""

    action: str  # "raise" | "escalate" | "clear"
    incident: Incident
    level: int = 0
    reason: str = ""


class IncidentTracker:
    """Runs the incident state machine for a single rule."""

    def __init__(
        self,
        kind: str,
        *,
        t_raise: float = 10.0,
        t_escalate: tuple[float, ...] = (60.0, 300.0),
        t_clear: float = 3.0,
        occlusion_grace_s: float = 8.0,
        cooldown_s: float = 30.0,
        clear_displacement: float = 1.0,
        confirmations: int = 3,
        associate_iou: float = 0.5,
        anchor_ema: float = 0.1,
    ):
        """
        Args:
            kind: Event kind prefix, e.g. ``"stalled"``.
            t_raise: Seconds of sustained trigger before an alert is raised.
            t_escalate: Additional durations at which to emit escalation events,
                each fired at most once per incident.
            t_clear: Seconds of sustained non-trigger before clearing.
            occlusion_grace_s: How long an incident survives with nothing
                associated to it, timers still running. Should exceed
                ``tracker.max_age / fps`` so a normal track death doesn't clear.
            cooldown_s: Suppression window at a cleared incident's anchor, so a
                flapping object doesn't produce a burst of alerts.
            clear_displacement: Object-size multiples the object must move
                before a clear is believed.
            confirmations: Consecutive triggering observations required before
                raising, on top of ``t_raise``.
            associate_iou: Box overlap above which a track joins an incident.
            anchor_ema: Smoothing for the anchor, so it follows an object
                settling into place without chasing per-frame jitter.
        """
        self.kind = kind
        self.t_raise = t_raise
        self.t_escalate = tuple(t_escalate)
        self.t_clear = t_clear
        self.occlusion_grace_s = occlusion_grace_s
        self.cooldown_s = cooldown_s
        self.clear_displacement = clear_displacement
        self.confirmations = confirmations
        self.associate_iou = associate_iou
        self.anchor_ema = anchor_ema

        self.incidents: list[Incident] = []
        self._cooldowns: list[tuple[tuple[float, float], float, float]] = []

    # ---- association --------------------------------------------------------
    def _match(self, obs: Observation) -> Incident | None:
        """Find the open incident this observation belongs to, if any."""
        # A track already bound to an incident stays bound. Position alone is
        # not enough: an object that drives away stops overlapping its own
        # incident, so a purely spatial rule could never *witness* it leaving
        # and the incident would only ever clear by timeout.
        for inc in self.incidents:
            if (
                inc.state != CLEARED
                and inc.zone == obs.zone
                and obs.track_id in inc.contributing_track_ids
            ):
                return inc

        obs_anchor = anchor_of(obs.bbox)
        best, best_score = None, 0.0
        for inc in self.incidents:
            if inc.state == CLEARED:
                continue
            if inc.zone != obs.zone:
                continue
            score = iou(obs.bbox, inc.bbox)
            if score < self.associate_iou:
                # Fall back to anchor proximity: boxes can shrink a lot when an
                # object is partly occluded, dropping IoU below threshold while
                # the object has plainly not moved.
                dist = (
                    (obs_anchor[0] - inc.anchor[0]) ** 2 + (obs_anchor[1] - inc.anchor[1]) ** 2
                ) ** 0.5
                if inc.anchor_size > 0 and dist < 0.5 * inc.anchor_size:
                    score = self.associate_iou
                else:
                    continue
            if score > best_score:
                best, best_score = inc, score
        return best

    def _in_cooldown(self, obs: Observation, now: float) -> bool:
        obs_anchor = anchor_of(obs.bbox)
        for anchor, size, until in self._cooldowns:
            if now > until:
                continue
            dist = ((obs_anchor[0] - anchor[0]) ** 2 + (obs_anchor[1] - anchor[1]) ** 2) ** 0.5
            if size > 0 and dist < 0.5 * size:
                return True
        return False

    # ---- main step ----------------------------------------------------------
    def observe(
        self, now: float, observations: list[Observation], *, suspended: bool = False
    ) -> list[IncidentUpdate]:
        """Advance the state machine by one frame.

        Args:
            now: Frame timestamp in seconds (media time for files, wall clock
                for live sources).
            observations: One entry per candidate track for this rule.
            suspended: While True (e.g. inside a camera-bump settle window) no
                transition fires and no incident is created, but timers keep
                running. A gust of wind must not manufacture alerts, and must
                not falsely clear real ones either.
        """
        updates: list[IncidentUpdate] = []
        self._cooldowns = [c for c in self._cooldowns if c[2] > now]

        matched: dict[str, Observation] = {}
        for obs in observations:
            inc = self._match(obs)
            if inc is not None:
                inc.contributing_track_ids.add(obs.track_id)
                # If several tracks land on one incident, the triggering one
                # decides its fate — otherwise a car passing in front of a
                # stalled one could clear it.
                existing = matched.get(inc.id)
                if existing is None or (obs.still and not existing.still):
                    matched[inc.id] = obs
            elif obs.still and not suspended and not self._in_cooldown(obs, now):
                self.incidents.append(self._new_incident(obs, now))

        for inc in self.incidents:
            if inc.state == CLEARED:
                continue
            obs = matched.get(inc.id)
            updates.extend(self._advance(inc, obs, now, suspended=suspended))

        self.incidents = [i for i in self.incidents if i.state != CLEARED]
        return updates

    def _new_incident(self, obs: Observation, now: float) -> Incident:
        return Incident(
            id=new_incident_id(now),
            kind=self.kind,
            zone=obs.zone,
            anchor=anchor_of(obs.bbox),
            anchor_size=size_of(obs.bbox),
            bbox=obs.bbox,
            class_label=obs.class_label,
            state=CANDIDATE,
            first_trigger_ts=now,
            last_trigger_ts=now,
            last_seen_ts=now,
            confirmations=1,
            contributing_track_ids={obs.track_id},
            display_bbox=obs.display_bbox or obs.bbox,
        )

    def _advance(
        self, inc: Incident, obs: Observation | None, now: float, *, suspended: bool
    ) -> list[IncidentUpdate]:
        updates: list[IncidentUpdate] = []

        if obs is not None:
            inc.last_seen_ts = now
            inc.class_label = obs.class_label
            inc.display_bbox = obs.display_bbox or obs.bbox
            if obs.still:
                if inc.confirmations > 1 and self._displacement(inc, obs) > self.clear_displacement:
                    # The bound track stopped, but somewhere else entirely: it
                    # drove off and parked. The original spot is vacant, so
                    # clear here rather than dragging the incident along with
                    # the vehicle. A fresh incident opens at the new location.
                    if inc.state in (ACTIVE, CLEARING):
                        updates.append(self._clear(inc, now, "moved"))
                    else:
                        inc.state = CLEARED
                    return updates
                inc.bbox = obs.bbox
                a = self.anchor_ema
                new_anchor = anchor_of(obs.bbox)
                inc.anchor = (
                    (1 - a) * inc.anchor[0] + a * new_anchor[0],
                    (1 - a) * inc.anchor[1] + a * new_anchor[1],
                )
                inc.anchor_size = (1 - a) * inc.anchor_size + a * size_of(obs.bbox)
                inc.last_trigger_ts = now
                inc.confirmations += 1
                if inc.state == CLEARING:
                    # Re-triggered before the clear was confirmed. The original
                    # start time is deliberately kept: an object that twitched
                    # and settled has not been safe for one second.
                    inc.state = ACTIVE
                    inc.clearing_since_ts = None
            elif obs.still is False:
                self._begin_clearing(inc, now)
            # obs.still is None -> seen but unjudgeable; hold everything.
        elif now - inc.last_seen_ts > self.occlusion_grace_s:
            self._begin_clearing(inc, now)

        if suspended:
            return updates

        # ---- raise
        if (
            inc.state == CANDIDATE
            and now - inc.first_trigger_ts >= self.t_raise
            and inc.confirmations >= self.confirmations
        ):
            inc.state = ACTIVE
            updates.append(IncidentUpdate("raise", inc))

        # ---- escalate (one-shot per level)
        if inc.state == ACTIVE:
            duration = inc.duration_s(now)
            for level, threshold in enumerate(self.t_escalate, start=1):
                if duration >= threshold and level not in inc.escalated_levels:
                    inc.escalated_levels.add(level)
                    updates.append(IncidentUpdate("escalate", inc, level=level))

        # ---- clear
        if inc.state == CLEARING and inc.clearing_since_ts is not None:
            elapsed = now - inc.clearing_since_ts
            absent = obs is None
            if absent:
                # Nothing to measure displacement against. Only conclude the
                # object left once the occlusion grace has also expired.
                if now - inc.last_seen_ts > self.occlusion_grace_s + self.t_clear:
                    updates.append(self._clear(inc, now, "absent"))
            elif elapsed >= self.t_clear:
                moved = self._displacement(inc, obs)
                if moved >= self.clear_displacement:
                    updates.append(self._clear(inc, now, "moved"))

        return updates

    def _begin_clearing(self, inc: Incident, now: float) -> None:
        if inc.state == CANDIDATE:
            # Never alerted; just drop it. No clear event for something the
            # outside world was never told about.
            inc.state = CLEARED
        elif inc.state == ACTIVE and inc.clearing_since_ts is None:
            inc.state = CLEARING
            inc.clearing_since_ts = now

    def _displacement(self, inc: Incident, obs: Observation) -> float:
        if inc.anchor_size <= 0:
            return 0.0
        ax, ay = anchor_of(obs.bbox)
        dist = ((ax - inc.anchor[0]) ** 2 + (ay - inc.anchor[1]) ** 2) ** 0.5
        return dist / inc.anchor_size

    def _clear(self, inc: Incident, now: float, reason: str) -> IncidentUpdate:
        duration = max(0.0, inc.last_trigger_ts - inc.first_trigger_ts)
        inc.state = CLEARED
        self._cooldowns.append((inc.anchor, inc.anchor_size, now + self.cooldown_s))
        return IncidentUpdate("clear", inc, reason=reason, level=int(duration))

    @property
    def open_incidents(self) -> list[Incident]:
        """Incidents that have actually alerted (candidates are not yet news)."""
        return [i for i in self.incidents if i.state in (ACTIVE, CLEARING)]
