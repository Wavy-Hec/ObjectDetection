"""Tests for the video source factory and routing logic.

``cv2.VideoCapture`` is monkeypatched so these tests need no real camera or
video file.
"""

import cv2
import pytest

from src.video_source import (
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
