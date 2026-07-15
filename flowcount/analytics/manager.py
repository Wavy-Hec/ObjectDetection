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
    ):
        self.analyzers: list[Analyzer] = list(analyzers or [])
        self.exporters = list(exporters or [])
        self.recorder = recorder
        self.events: list[Event] = []  # events from the most recent frame

    def update(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        for analyzer in self.analyzers:
            events.extend(analyzer.update(ctx))

        for exporter in self.exporters:
            exporter.write_tracks(ctx)
            if events:
                exporter.write_events(events)

        if self.recorder is not None and ctx.frame is not None:
            self.recorder.process(ctx.frame, events, ctx.frame_index)

        for e in events:
            logger.info("event: %s", e.summary())

        self.events = events
        return events

    def draw(self, frame: np.ndarray) -> None:
        for analyzer in self.analyzers:
            analyzer.draw(frame)

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

    @property
    def is_empty(self) -> bool:
        return not (self.analyzers or self.exporters or self.recorder)
