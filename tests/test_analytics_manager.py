"""Tests for the AnalyticsManager fan-out."""

import os

import numpy as np

from flowcount.analytics import (
    AnalyticsManager,
    CSVExporter,
    HeatmapAccumulator,
    LineCrossingCounter,
)


def test_manager_fans_out_to_analyzers_and_exporters(make_track, make_ctx, tmp_path):
    line = LineCrossingCounter((0, 100), (200, 100))
    heatmap = HeatmapAccumulator(radius=5)
    csv_path = str(tmp_path / "m.csv")
    manager = AnalyticsManager([line, heatmap], [CSVExporter(csv_path)])

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    manager.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=1, timestamp=0.0, frame=frame))
    events = manager.update(make_ctx([make_track(1, [40, 140, 60, 160])], frame_index=2, timestamp=1.0, frame=frame))

    assert any(e.kind == "line_cross" for e in events)
    assert manager.events == events

    # draw runs across all analyzers without error
    manager.draw(frame.copy())

    saved = manager.save(str(tmp_path / "art"))
    manager.close()

    assert any("heatmap" in s for s in saved)
    assert os.path.exists(csv_path)


def test_empty_manager():
    assert AnalyticsManager().is_empty
