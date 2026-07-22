"""AnalyticsManager — fans each frame's tracks out to all analyzers, exporters,
and the optional event-clip recorder.

This is the single object the Pipeline talks to (via ``update(ctx)`` and
``draw(frame)``), keeping the pipeline agnostic of which analytics are active.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from .base import Analyzer, Event, FrameContext

logger = logging.getLogger(__name__)


class AnalyticsManager:
    def __init__(
        self,
        analyzers: Sequence[Analyzer] | None = None,
        exporters: Sequence | None = None,
        recorder=None,
        *,
        alert_dispatcher=None,
    ):
        """
        Args:
            analyzers: Per-frame analytics components.
            exporters: Objects with ``write_tracks(ctx)`` / ``write_events(events)``
                / ``close()``. Called synchronously on the frame thread.
            recorder: Optional event-triggered clip recorder.
            alert_dispatcher: Optional sink with ``submit(events, ctx)`` and
                ``close()``, for delivering alerts off-box. It must never block
                the frame thread — hand off to a queue and return.
        """
        self.analyzers: list[Analyzer] = list(analyzers or [])
        self.exporters = list(exporters or [])
        self.recorder = recorder
        self.alert_dispatcher = alert_dispatcher
        self.events: list[Event] = []  # events from the most recent frame

    def update(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        for analyzer in self.analyzers:
            # One misbehaving analyzer degrades its own feature rather than
            # taking down the whole frame (and, in the CLI, the process).
            try:
                events.extend(analyzer.update(ctx))
            except Exception:
                logger.exception("analyzer %s failed", type(analyzer).__name__)

        for exporter in self.exporters:
            exporter.write_tracks(ctx)
            if events:
                exporter.write_events(events)

        if self.recorder is not None and ctx.frame is not None:
            self.recorder.process(ctx.frame, events, ctx.frame_index)

        if self.alert_dispatcher is not None and events:
            self.alert_dispatcher.submit(events, ctx)

        for e in events:
            logger.info("event: %s", e.summary())

        self.events = events
        return events

    def draw(self, frame: np.ndarray) -> None:
        for analyzer in self.analyzers:
            try:
                analyzer.draw(frame)
            except Exception:
                logger.exception("analyzer %s draw failed", type(analyzer).__name__)

    def stats(self) -> dict:
        """Merged live counters from every analyzer that publishes them.

        Lets front ends surface a new analyzer without reaching into its
        internals. Later analyzers win on key collisions.
        """
        merged: dict = {}
        for analyzer in self.analyzers:
            try:
                merged.update(analyzer.stats() or {})
            except Exception:
                logger.exception("analyzer %s stats failed", type(analyzer).__name__)
        return merged

    def save(self, path_prefix: str) -> list[str]:
        """Ask each analyzer to persist any artifact (e.g. heatmap image)."""
        saved = []
        for analyzer in self.analyzers:
            path = analyzer.save(path_prefix)
            if path:
                saved.append(path)
                logger.info("Saved analytics artifact: %s", path)
        return saved

    def close(self) -> None:
        for exporter in self.exporters:
            exporter.close()
        if self.recorder is not None:
            self.recorder.close()
        if self.alert_dispatcher is not None:
            self.alert_dispatcher.close()

    @property
    def is_empty(self) -> bool:
        return not (self.analyzers or self.exporters or self.recorder or self.alert_dispatcher)
