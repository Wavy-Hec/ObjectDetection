"""Tests for the event-triggered clip recorder (ring buffer + post-roll)."""

import numpy as np

from src.analytics import EventClipRecorder
from src.analytics.base import Event


class FakeWriter:
    def __init__(self, *args, **kwargs):
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


def _frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_preroll_flushed_and_postroll_closes(tmp_path):
    writers = []

    def factory(path, fps, size):
        w = FakeWriter()
        writers.append(w)
        return w

    rec = EventClipRecorder(str(tmp_path), fps=10, pre_seconds=0.3, post_seconds=0.3,
                            writer_factory=factory)
    assert rec.pre_frames == 3 and rec.post_frames == 3

    # 5 frames with no events -> only the ring buffer fills, no writer yet
    for i in range(5):
        rec.process(_frame(), [], i)
    assert writers == []

    # event on frame 5 -> open writer, flush pre-roll (3) + current (1)
    rec.process(_frame(), [Event("line_cross", 1, "p", 5, 0.0, {})], 5)
    assert len(writers) == 1
    assert len(writers[0].frames) == 4

    # 3 post-roll frames with no events -> writer closes after countdown
    for i in range(6, 9):
        rec.process(_frame(), [], i)
    assert writers[0].released is True
    assert len(writers[0].frames) == 7  # 3 pre + 1 trigger + 3 post
    assert rec.clips and len(rec.clips) == 1


def test_new_event_extends_postroll(tmp_path):
    writers = []
    rec = EventClipRecorder(str(tmp_path), fps=10, pre_seconds=0.1, post_seconds=0.2,
                            writer_factory=lambda *a: writers.append(FakeWriter()) or writers[-1])
    rec.process(_frame(), [Event("e", 1, "p", 0, 0.0, {})], 0)  # opens, post=2
    rec.process(_frame(), [], 1)                                  # post -> 1
    rec.process(_frame(), [Event("e", 1, "p", 2, 0.0, {})], 2)   # re-extends post -> 2
    assert writers[0].released is False
    rec.close()
    assert writers[0].released is True
