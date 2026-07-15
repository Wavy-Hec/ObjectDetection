"""Tests for the activity heatmap accumulator."""

import os

import numpy as np

from flowcount.analytics import HeatmapAccumulator


def test_accumulates_and_renders(make_track, make_ctx):
    hm = HeatmapAccumulator(radius=10)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hm.update(make_ctx([make_track(1, [40, 40, 60, 60])], frame_index=1, frame=frame))
    assert hm.accumulator is not None
    assert hm.accumulator.max() > 0
    img = hm.render()
    assert img is not None
    assert img.shape == (100, 100, 3)


def test_no_frame_means_no_accumulation(make_track, make_ctx):
    hm = HeatmapAccumulator()
    out = hm.update(make_ctx([make_track(1, [1, 1, 2, 2])], frame_index=1, frame=None))
    assert out == []
    assert hm.render() is None


def test_draw_blends_onto_canvas(make_track, make_ctx):
    hm = HeatmapAccumulator(radius=10, alpha=0.6)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    hm.update(make_ctx([make_track(1, [20, 20, 30, 30])], frame_index=1, frame=frame))
    canvas = np.zeros((50, 50, 3), dtype=np.uint8)
    hm.draw(canvas)
    assert canvas.sum() > 0  # heatmap drew pixels


def test_save_writes_image(make_track, make_ctx, tmp_path):
    hm = HeatmapAccumulator(radius=8)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    hm.update(make_ctx([make_track(1, [10, 10, 20, 20])], frame_index=1, frame=frame))
    path = hm.save(str(tmp_path / "activity"))
    assert path is not None
    assert os.path.exists(path)
