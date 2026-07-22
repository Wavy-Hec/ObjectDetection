"""Tests for AnalyticsManager's stats merge, alert dispatcher, and fault isolation."""

import numpy as np

from flowcount.analytics import AnalyticsManager
from flowcount.analytics.base import Analyzer, Event


class StatsAnalyzer(Analyzer):
    def __init__(self, payload, events=None):
        self.payload = payload
        self._events = events or []

    def update(self, ctx):
        return list(self._events)

    def stats(self):
        return self.payload


class BrokenAnalyzer(Analyzer):
    def update(self, ctx):
        raise RuntimeError("boom")

    def draw(self, frame):
        raise RuntimeError("boom")

    def stats(self):
        raise RuntimeError("boom")


def _event(kind="stalled"):
    return Event(kind, 1, "car", 1, 0.0)


def test_stats_merges_every_analyzer(make_ctx):
    manager = AnalyticsManager([StatsAnalyzer({"a": 1}), StatsAnalyzer({"b": 2})])
    assert manager.stats() == {"a": 1, "b": 2}


def test_stats_default_is_empty(make_ctx):
    assert AnalyticsManager().stats() == {}


def test_one_broken_analyzer_does_not_kill_the_frame(make_ctx):
    """A new analyzer failing should degrade its own feature, not the process."""
    good = StatsAnalyzer({"ok": True}, events=[_event()])
    manager = AnalyticsManager([BrokenAnalyzer(), good])

    events = manager.update(make_ctx([]))

    assert [e.kind for e in events] == ["stalled"]  # the healthy analyzer still ran
    assert manager.stats() == {"ok": True}
    manager.draw(np.zeros((10, 10, 3), dtype=np.uint8))  # must not raise


def test_alert_dispatcher_receives_events_only_when_they_fire(make_ctx):
    class RecordingDispatcher:
        def __init__(self):
            self.batches = []
            self.closed = False

        def submit(self, events, ctx):
            self.batches.append(list(events))

        def close(self):
            self.closed = True

    dispatcher = RecordingDispatcher()
    manager = AnalyticsManager([StatsAnalyzer({}, events=[])], alert_dispatcher=dispatcher)

    manager.update(make_ctx([]))
    assert dispatcher.batches == []  # quiet frames cost nothing

    manager.analyzers = [StatsAnalyzer({}, events=[_event()])]
    manager.update(make_ctx([]))
    assert [e.kind for e in dispatcher.batches[0]] == ["stalled"]

    manager.close()
    assert dispatcher.closed


def test_manager_with_only_a_dispatcher_is_not_empty():
    """is_empty gates whether the CLI builds analytics at all."""

    class Dispatcher:
        def submit(self, events, ctx):
            pass

        def close(self):
            pass

    assert AnalyticsManager().is_empty
    assert not AnalyticsManager(alert_dispatcher=Dispatcher()).is_empty
