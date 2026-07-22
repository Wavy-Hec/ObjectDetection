"""Tests for the incident state machine.

These encode the failure modes the design exists to prevent — an ID switch
resetting a mature stall timer, an occlusion reading as "cleared", a camera
bump silently clearing a live hazard.
"""

from flowcount.safety.incidents import ACTIVE, CANDIDATE, CLEARING, IncidentTracker, Observation


def _obs(track_id, cx, cy, still=True, size=40, zone=None):
    return Observation(
        track_id=track_id,
        bbox=(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
        class_label="car",
        still=still,
        zone=zone,
    )


def _run(tracker, steps, *, start=0.0, dt=1.0, suspended=False):
    """Feed a list of per-frame observation lists; return all updates."""
    updates = []
    for i, obs in enumerate(steps):
        updates.extend(tracker.observe(start + i * dt, obs, suspended=suspended))
    return updates


def test_raise_after_threshold():
    t = IncidentTracker("stalled", t_raise=10.0, confirmations=3)
    updates = _run(t, [[_obs(1, 100, 100)] for _ in range(10)])
    assert [u.action for u in updates] == []  # 9 s elapsed at the last frame

    updates = t.observe(10.0, [_obs(1, 100, 100)])
    assert [u.action for u in updates] == ["raise"]
    assert updates[0].incident.state == ACTIVE


def test_no_raise_before_threshold():
    t = IncidentTracker("stalled", t_raise=10.0)
    updates = _run(t, [[_obs(1, 100, 100)] for _ in range(8)])
    assert updates == []
    assert t.incidents[0].state == CANDIDATE
    assert t.open_incidents == []  # a candidate is not yet news


def test_confirmations_gate_the_raise():
    """A long gap between two sightings satisfies t_raise but not confirmation."""
    t = IncidentTracker("stalled", t_raise=5.0, confirmations=3)
    t.observe(0.0, [_obs(1, 100, 100)])
    updates = t.observe(20.0, [_obs(1, 100, 100)])
    assert updates == []  # only 2 confirmations


def test_id_switch_does_not_reset_the_timer():
    """The whole point of anchoring on place rather than on track.id.

    A tracker that reassigns an ID mid-stall (occlusion, class flip) must not
    silently restart a 9-second timer — the alert would simply never fire.
    """
    t = IncidentTracker("stalled", t_raise=10.0, confirmations=3)
    _run(t, [[_obs(1, 100, 100)] for _ in range(9)])
    # Same object, brand new track ID, essentially the same position.
    updates = t.observe(9.0, [_obs(77, 101, 100)])
    updates += t.observe(10.0, [_obs(77, 101, 100)])
    assert [u.action for u in updates] == ["raise"]
    assert updates[0].incident.contributing_track_ids == {1, 77}


def test_occlusion_holds_the_incident_with_the_timer_running():
    """Absence is not clearance: a truck passing in front must not clear it."""
    t = IncidentTracker("stalled", t_raise=5.0, occlusion_grace_s=8.0, confirmations=3)
    _run(t, [[_obs(1, 100, 100)] for _ in range(6)])
    assert t.incidents[0].state == ACTIVE

    # Four seconds with nothing associated at all.
    for ts in (6.0, 7.0, 8.0, 9.0):
        assert t.observe(ts, []) == []
    assert t.incidents[0].state == ACTIVE
    assert t.incidents[0].duration_s(9.0) == 9.0  # timer kept running


def test_unjudgeable_observation_holds_state():
    """still=None (coasted / low detection support) must neither raise nor clear."""
    t = IncidentTracker("stalled", t_raise=5.0, confirmations=3)
    _run(t, [[_obs(1, 100, 100)] for _ in range(6)])
    assert t.incidents[0].state == ACTIVE

    for ts in (6.0, 7.0, 8.0):
        assert t.observe(ts, [_obs(1, 100, 100, still=None)]) == []
    assert t.incidents[0].state == ACTIVE


def test_clear_requires_both_time_and_displacement():
    """Sustained motion alone is not enough — jitter would clear a live hazard."""
    t = IncidentTracker(
        "stalled", t_raise=5.0, t_clear=3.0, clear_displacement=1.0, confirmations=3
    )
    _run(t, [[_obs(1, 100, 100)] for _ in range(6)])

    # Moving, but barely — 5 px on a 40 px object is an eighth of its length.
    for ts in (6.0, 7.0, 8.0, 9.0, 10.0):
        assert t.observe(ts, [_obs(1, 105, 100, still=False)]) == []
    assert t.incidents[0].state == CLEARING

    # Now a full object length away.
    updates = t.observe(11.0, [_obs(1, 160, 100, still=False)])
    assert [u.action for u in updates] == ["clear"]
    assert updates[0].reason == "moved"


def test_retrigger_before_clear_returns_to_active_without_resetting():
    """An object that twitches and settles has not been safe for one second."""
    t = IncidentTracker("stalled", t_raise=5.0, t_clear=3.0, confirmations=3)
    _run(t, [[_obs(1, 100, 100)] for _ in range(6)])
    first_ts = t.incidents[0].first_trigger_ts

    t.observe(6.0, [_obs(1, 100, 100, still=False)])
    assert t.incidents[0].state == CLEARING

    t.observe(7.0, [_obs(1, 100, 100, still=True)])
    assert t.incidents[0].state == ACTIVE
    assert t.incidents[0].first_trigger_ts == first_ts  # not restarted


def test_absent_object_eventually_clears():
    """Held through the grace window, but a departed object must not linger."""
    t = IncidentTracker("stalled", t_raise=5.0, t_clear=3.0, occlusion_grace_s=8.0, confirmations=3)
    _run(t, [[_obs(1, 100, 100)] for _ in range(6)])

    updates = []
    for ts in range(6, 25):
        updates.extend(t.observe(float(ts), []))
    assert [u.action for u in updates] == ["clear"]
    assert updates[0].reason == "absent"


def test_escalation_is_one_shot_per_level():
    t = IncidentTracker("stalled", t_raise=5.0, t_escalate=(10.0, 20.0), confirmations=3)
    updates = []
    for ts in range(0, 31):
        updates.extend(t.observe(float(ts), [_obs(1, 100, 100)]))

    actions = [(u.action, u.level) for u in updates]
    assert actions == [("raise", 0), ("escalate", 1), ("escalate", 2)]


def test_candidate_that_moves_away_never_alerts():
    """Nothing was announced, so nothing needs clearing."""
    t = IncidentTracker("stalled", t_raise=10.0, confirmations=3)
    _run(t, [[_obs(1, 100, 100)] for _ in range(5)])
    updates = t.observe(5.0, [_obs(1, 100, 100, still=False)])
    assert updates == []
    assert t.incidents == []


def test_cooldown_suppresses_a_reopen_at_the_same_spot():
    t = IncidentTracker("stalled", t_raise=3.0, t_clear=1.0, cooldown_s=30.0, confirmations=3)
    _run(t, [[_obs(1, 100, 100)] for _ in range(4)])
    t.observe(5.0, [_obs(1, 100, 100, still=False)])
    updates = t.observe(7.0, [_obs(1, 200, 100, still=False)])
    assert [u.action for u in updates] == ["clear"]

    # Something stops at the same place again while the cooldown is live.
    for ts in range(8, 20):
        assert t.observe(float(ts), [_obs(2, 100, 100)]) == []
    assert t.incidents == []


def test_suspension_blocks_transitions_but_not_timers():
    """A camera bump must neither manufacture alerts nor clear real ones."""
    t = IncidentTracker("stalled", t_raise=5.0, confirmations=3)
    _run(t, [[_obs(1, 100, 100)] for _ in range(6)])
    assert t.incidents[0].state == ACTIVE

    # Everything appears to move at once; suspended, so no clear is emitted.
    for ts in (6.0, 7.0, 8.0, 9.0, 10.0, 11.0):
        assert t.observe(ts, [_obs(1, 300, 300, still=False)], suspended=True) == []
    assert t.incidents[0].state in (ACTIVE, CLEARING)


def test_suspension_prevents_new_incidents():
    t = IncidentTracker("stalled", t_raise=1.0, confirmations=1)
    for ts in range(0, 6):
        assert t.observe(float(ts), [_obs(1, 100, 100)], suspended=True) == []
    assert t.incidents == []


def test_zones_are_independent():
    t = IncidentTracker("stalled", t_raise=3.0, confirmations=3)
    updates = []
    for ts in range(0, 5):
        updates.extend(
            t.observe(
                float(ts),
                [_obs(1, 100, 100, zone="a"), _obs(2, 500, 500, zone="b")],
            )
        )
    assert sorted(u.incident.zone for u in updates) == ["a", "b"]


def test_incident_ids_are_unique_and_time_sortable():
    t = IncidentTracker("stalled", t_raise=1.0, confirmations=1)
    _run(t, [[_obs(1, 100, 100)] for _ in range(3)])
    _run(t, [[_obs(2, 500, 500)] for _ in range(3)], start=100.0)
    ids = [i.id for i in t.incidents]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)  # lexicographic order is chronological order


def test_incident_does_not_follow_a_vehicle_that_parks_elsewhere():
    """The incident marks a place, so it must not be dragged along by its track.

    Without this, a car that stalls, drives off, and stops again 300 px away
    would carry its mature timer with it and alert instantly at the new spot.
    """
    t = IncidentTracker(
        "stalled", t_raise=5.0, t_clear=3.0, clear_displacement=1.0, confirmations=3
    )
    _run(t, [[_obs(1, 100, 100)] for _ in range(6)])
    assert t.incidents[0].state == ACTIVE
    original = t.incidents[0].id

    # Same track, stopped again far away.
    updates = t.observe(7.0, [_obs(1, 400, 100, still=True)])
    assert [u.action for u in updates] == ["clear"]
    assert updates[0].incident.id == original

    # A fresh incident starts building at the new location, from zero.
    t.observe(8.0, [_obs(1, 400, 100, still=True)])
    assert t.incidents[0].id != original
    assert t.incidents[0].state == CANDIDATE
