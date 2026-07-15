"""Tests for the reusable Pipeline.

Uses lightweight stub detector/tracker objects so the pipeline logic is
verified without loading any ML model.
"""

import numpy as np

from flowcount.detector import Detection
from flowcount.pipeline import FrameStats, Pipeline, ProcessResult
from flowcount.tracker import Track


class StubDetector:
    def detect(self, frame):
        return [Detection([10, 10, 50, 50], "person", 0.9)]


class StubTracker:
    def update(self, detections):
        return [
            Track(
                id=1,
                bbox=[10, 10, 50, 50],
                class_label="person",
                confidence=0.9,
                age=1,
                hits=1,
                time_since_update=0,
            )
        ]


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
    assert frame.sum() == 0  # original untouched
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


def test_detect_every_n_coasts_between_detection_frames():
    class CountingDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            return [Detection([10, 10, 50, 50], "person", 0.9)]

    class CoastingTracker(StubTracker):
        def __init__(self):
            self.update_calls = 0
            self.coast_calls = 0

        def update(self, detections):
            self.update_calls += 1
            return super().update(detections)

        def predict_only(self, max_coast):
            self.coast_calls += 1
            assert max_coast == 3
            return super().update([])

    detector, tracker = CountingDetector(), CoastingTracker()
    pipeline = Pipeline(detector, tracker, draw_masks=False, detect_every=3)

    ran = [pipeline.process_frame(_frame()).stats.detection_ran for _ in range(6)]

    assert ran == [True, False, False, True, False, False]
    assert detector.calls == 2
    assert tracker.update_calls == 2
    assert tracker.coast_calls == 4


def test_detect_every_default_detects_every_frame():
    pipeline = Pipeline(StubDetector(), StubTracker(), draw_masks=False)
    stats = pipeline.process_frame(_frame()).stats
    assert stats.detection_ran is True
    assert pipeline.detect_every == 1


def test_analytics_manager_hook_called_and_events_propagate():
    calls = {"update": 0, "draw": 0}

    class StubAnalytics:
        def update(self, ctx):
            calls["update"] += 1
            # ctx carries the frame's tracks + metadata
            assert ctx.frame_index == 1
            assert ctx.frame is not None
            return ["fake-event"]

        def draw(self, frame):
            calls["draw"] += 1

    pipeline = Pipeline(
        StubDetector(), StubTracker(), draw_masks=False, analytics_manager=StubAnalytics()
    )
    result = pipeline.process_frame(_frame())
    assert calls["update"] == 1
    assert calls["draw"] == 1
    assert result.events == ["fake-event"]
