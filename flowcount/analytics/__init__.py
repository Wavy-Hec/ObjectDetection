"""Analytics & events subpackage.

Each analyzer consumes a :class:`FrameContext` (the frame's tracks + metadata)
and emits :class:`Event` objects. The :class:`AnalyticsManager` ties them
together and is the single hook the :class:`~flowcount.pipeline.Pipeline` drives.
"""

from .base import Analyzer, Event, FrameContext
from .exporters import CSVExporter, SQLiteExporter
from .heatmap import HeatmapAccumulator
from .line_counter import LineCrossingCounter
from .manager import AnalyticsManager
from .recorder import EventClipRecorder
from .zones import Zone, ZoneManager

__all__ = [
    "Analyzer",
    "Event",
    "FrameContext",
    "LineCrossingCounter",
    "Zone",
    "ZoneManager",
    "HeatmapAccumulator",
    "EventClipRecorder",
    "CSVExporter",
    "SQLiteExporter",
    "AnalyticsManager",
]
