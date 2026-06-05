"""Tests for the reusable Pipeline.

Uses lightweight stub detector/tracker objects so the pipeline logic is
verified without loading any ML model.
"""

import numpy as np

from src.detector import Detection
from src.pipeline import FrameStats, Pipeline, ProcessResult
from src.tracker import Track


class StubDetector:
    def detect(self, frame):
        return [Detection([10, 10, 50, 50], "person", 0.9)]


class StubTracker:
    def update(self, detections):
        return [Track(
            id=1, bbox=[10, 10, 50, 50], class_label="person", confidence=0.9,
            age=1, hits=1, time_since_update=0,
        )]


def _frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_process_frame_returns_result():
    pipeline = Pipeline(StubDetector(), StubTracker(), draw_masks=False)
    result = pipeline.process_frame(_frame())

    assert isinstance(result, ProcessResult)
    assert isinstance(result.stats, FrameStats)
    assert result.stats.num_detections == 1
    assert result.stats.num_tracks == 1
    assert result.stats.frame_index == 1
    assert result.frame.shape == (100, 100, 3)


def test_annotate_does_not_mutate_input_frame():
    pipeline = Pipeline(StubDetector(), StubTracker(), draw_masks=False)
    frame = _frame()
    result = pipeline.process_frame(frame)
    assert frame.sum() == 0          # original untouched
    assert result.frame is not frame  # annotated copy returned


def test_no_annotate_returns_same_frame():
    pipeline = Pipeline(StubDetector(), StubTracker())
    frame = _frame()
    result = pipeline.process_frame(frame, annotate=False)
    assert result.frame is frame


def test_frame_index_increments():
    pipeline = Pipeline(StubDetector(), StubTracker(), draw_masks=False)
    pipeline.process_frame(_frame())
    second = pipeline.process_frame(_frame())
    assert second.stats.frame_index == 2


def test_analytics_manager_hook_called():
    calls = {"update": 0, "draw": 0}

    class StubAnalytics:
        def update(self, tracks, frame_index):
            calls["update"] += 1

        def draw(self, frame):
            calls["draw"] += 1

    pipeline = Pipeline(StubDetector(), StubTracker(),
                        draw_masks=False, analytics_manager=StubAnalytics())
    pipeline.process_frame(_frame())
    assert calls["update"] == 1
    assert calls["draw"] == 1
