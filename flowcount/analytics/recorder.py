"""Event-triggered clip recorder.

Keeps a ring buffer of recent frames (pre-roll). When an event fires, it opens
a clip, flushes the pre-roll, then keeps recording for a post-roll window,
extending the window if more events arrive.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .base import Event

logger = logging.getLogger(__name__)


def _default_writer(path: str, fps: float, size) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size)


class EventClipRecorder:
    def __init__(
        self,
        out_dir: str,
        fps: float = 30.0,
        pre_seconds: float = 2.0,
        post_seconds: float = 3.0,
        writer_factory: Callable | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps if fps and fps > 0 else 30.0
        self.pre_frames = max(1, int(pre_seconds * self.fps))
        self.post_frames = max(1, int(post_seconds * self.fps))
        self.writer_factory = writer_factory or _default_writer

        self.buffer: deque = deque(maxlen=self.pre_frames)
        self.writer = None
        self.post_remaining = 0
        self.size = None
        self.current_path: str | None = None
        self.clips: list[str] = []

    def process(self, frame: np.ndarray, events: list[Event], frame_index: int) -> None:
        """Feed one frame (+ any events from this frame) to the recorder."""
        self.size = (frame.shape[1], frame.shape[0])

        if self.writer is None:
            if events:
                # Open a clip: flush the pre-roll history, then the trigger frame.
                self._open(frame_index)
                for buffered in self.buffer:
                    self.writer.write(buffered)
                self.buffer.clear()
                self.writer.write(frame.copy())
                self.post_remaining = self.post_frames
            else:
                self.buffer.append(frame.copy())
        else:
            self.writer.write(frame.copy())
            if events:
                self.post_remaining = self.post_frames  # extend on new activity
            else:
                self.post_remaining -= 1
                if self.post_remaining <= 0:
                    self._close_current()

    def _open(self, frame_index: int) -> None:
        path = str(self.out_dir / f"event_{frame_index:06d}.mp4")
        self.writer = self.writer_factory(path, self.fps, self.size)
        self.current_path = path
        logger.info("Event clip started: %s", path)

    def _close_current(self) -> None:
        if self.writer is not None:
            self.writer.release()
            if self.current_path:
                self.clips.append(self.current_path)
                logger.info("Event clip saved: %s", self.current_path)
        self.writer = None
        self.current_path = None
        self.post_remaining = 0

    def close(self) -> None:
        self._close_current()
