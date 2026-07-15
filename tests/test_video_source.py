"""Tests for the video source factory and routing logic.

``cv2.VideoCapture`` is monkeypatched so these tests need no real camera or
video file.
"""

import cv2
import numpy as np
import pytest

from src.video_source import (
    LatestFrameGrabber,
    StreamSource,
    VideoFileSource,
    WebcamSource,
    create_video_source,
)


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture."""

    def __init__(self, *args, **kwargs):
        self.args = args

    def isOpened(self):
        return True

    def set(self, *args, **kwargs):
        return True

    def get(self, prop):
        return 0  # width/height/fps/frame-count all read as 0

    def read(self):
        return False, None

    def release(self):
        pass


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_video_source(str(tmp_path / "does_not_exist.mp4"))


def test_factory_routes_digit_string_to_webcam(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    src = create_video_source("0")
    assert isinstance(src, WebcamSource)
    assert src.get_properties()["source_type"] == "webcam"


def test_factory_routes_int_to_webcam(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    assert isinstance(create_video_source(1), WebcamSource)


def test_factory_routes_path_to_file(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")  # file just needs to exist
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    src = create_video_source(str(video))
    assert isinstance(src, VideoFileSource)
    assert src.get_properties()["source_type"] == "file"


def test_factory_routes_url_to_stream(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    src = create_video_source("rtsp://camera.example/stream1")
    assert isinstance(src, StreamSource)
    props = src.get_properties()
    assert props["source_type"] == "stream"
    assert props["total_frames"] == -1
    assert props["fps"] == 25  # streams reporting 0 fall back to 25


def test_webcam_requests_configured_resolution(monkeypatch):
    requested = {}

    class _RecordingCapture(_FakeCapture):
        def set(self, prop, value):
            requested[prop] = value
            return True

    monkeypatch.setattr(cv2, "VideoCapture", _RecordingCapture)
    create_video_source("0", webcam_width=640, webcam_height=480, webcam_fps=15)
    assert requested[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert requested[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert requested[cv2.CAP_PROP_FPS] == 15


def test_file_source_fps_zero_falls_back(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)  # get() reads fps as 0
    src = VideoFileSource(str(video))
    assert src.fps == 30  # never propagates fps=0 to VideoWriter/recorder


def test_stream_reconnects_after_read_failure(monkeypatch):
    class _FlakyCapture(_FakeCapture):
        opens = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _FlakyCapture.opens += 1
            self.generation = _FlakyCapture.opens

        def read(self):
            if self.generation == 1:
                return False, None  # first connection dies immediately
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, "VideoCapture", _FlakyCapture)
    src = StreamSource("rtsp://camera.example/live", reconnect_backoff=0.0)
    ok, frame = src.read()
    assert ok and frame is not None
    assert _FlakyCapture.opens == 2  # reopened once


def test_latest_frame_grabber_hands_out_newest_frame_once():
    class _CountingSource:
        """Fake source emitting frames stamped with a sequence number."""

        def __init__(self):
            self.n = 0

        def read(self):
            import time
            time.sleep(0.001)  # behave like a real camera, not a busy loop
            self.n += 1
            frame = np.zeros((2, 2, 3), dtype=np.uint16)
            frame[0, 0, 0] = self.n  # monotonic stamp
            return True, frame

        def get_properties(self):
            return {"source_type": "fake", "width": 2, "height": 2,
                    "fps": 30, "total_frames": -1}

        def is_opened(self):
            return True

        def release(self):
            pass

    with LatestFrameGrabber(_CountingSource(), read_timeout=2.0) as grabber:
        ok1, f1 = grabber.read()
        stamp1 = int(f1[0, 0, 0])
        ok2, f2 = grabber.read()
        stamp2 = int(f2[0, 0, 0])
        assert ok1 and ok2
        assert stamp2 > stamp1  # strictly newer frame, never a repeat
        assert grabber.get_properties()["source_type"] == "fake"
    assert not grabber.is_opened()  # released on context exit


def test_latest_frame_grabber_reports_source_failure():
    class _DeadSource:
        def read(self):
            return False, None

        def get_properties(self):
            return {}

        def is_opened(self):
            return False

        def release(self):
            pass

    grabber = LatestFrameGrabber(_DeadSource(), read_timeout=2.0)
    ok, frame = grabber.read()
    assert not ok and frame is None
    grabber.release()
