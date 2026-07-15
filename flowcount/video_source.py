"""
Video source abstraction: webcam, video file, and network streams (RTSP/HTTP).

All sources share the same surface (``read`` / ``get_properties`` /
``is_opened`` / ``release``) and support use as context managers. For live
sources, :class:`LatestFrameGrabber` can wrap any source to decouple capture
from inference: frames are read on a background thread and stale ones are
dropped, so slow models never build up latency.
"""

import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# URL schemes routed to StreamSource by the factory.
STREAM_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")


class VideoSourceError(Exception):
    """Exception for video source errors."""

    pass


class VideoSource(ABC):
    """Abstract base class for video sources."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read next frame. Returns (success, frame)."""
        pass

    @abstractmethod
    def get_properties(self) -> dict:
        """Get video properties (width, height, fps, total_frames)."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is opened."""
        pass

    @abstractmethod
    def release(self):
        """Release resources."""
        pass

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class WebcamSource(VideoSource):
    """Webcam video source."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: int | None = None,
    ):
        """
        Initialize webcam source.

        Args:
            device_index: Camera device index (0 for default)
            width: Requested capture width (the driver may pick the closest mode)
            height: Requested capture height
            fps: Requested capture FPS
            backend: OpenCV capture backend. Defaults to V4L2 on Linux and
                the platform default elsewhere.
        """
        self.device_index = device_index
        if backend is None:
            backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device_index, backend)

        if not self.cap.isOpened():
            raise VideoSourceError(f"Cannot open webcam {device_index}")

        # Use MJPEG for much higher FPS (default YUV is very slow at higher res)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or fps

        logger.info(
            "Webcam %s opened: %dx%d @ %d FPS", device_index, self.width, self.height, self.fps
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read frame from webcam."""
        return self.cap.read()

    def get_properties(self) -> dict:
        """Get webcam properties."""
        return {
            "source_type": "webcam",
            "device_index": self.device_index,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": -1,  # Infinite for webcam
        }

    def is_opened(self) -> bool:
        """Check if webcam is opened."""
        return self.cap.isOpened()

    def release(self):
        """Release webcam."""
        self.cap.release()


class VideoFileSource(VideoSource):
    """Video file source."""

    def __init__(self, file_path: str):
        """
        Initialize video file source.

        Args:
            file_path: Path to video file
        """
        self.file_path = file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        self.cap = cv2.VideoCapture(file_path)

        if not self.cap.isOpened():
            raise VideoSourceError(f"Cannot open video file: {file_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            # A zero FPS would corrupt downstream consumers (VideoWriter,
            # clip recorder, dwell timing), so fall back to a sane default.
            logger.warning("Video reports FPS=0; assuming 30")
            self.fps = 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        logger.info("Video file opened: %s", os.path.basename(file_path))
        logger.info("  Resolution: %dx%d", self.width, self.height)
        logger.info("  FPS: %s", self.fps)
        logger.info("  Total frames: %s", self.total_frames)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read next frame from video file."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame

    def get_properties(self) -> dict:
        """Get video file properties."""
        return {
            "source_type": "file",
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame,
        }

    def get_progress(self) -> float:
        """Get progress percentage."""
        if self.total_frames <= 0:
            return 0.0
        return (self.current_frame / self.total_frames) * 100.0

    def is_opened(self) -> bool:
        """Check if video file is opened."""
        return self.cap.isOpened()

    def release(self):
        """Release video file."""
        self.cap.release()


class StreamSource(VideoSource):
    """Network stream source (RTSP / HTTP / MJPEG) with automatic reconnect.

    A failed read triggers up to ``reconnect_attempts`` reopen attempts with
    exponential backoff, so a camera hiccup or a brief network drop does not
    end the session the way end-of-file does.
    """

    def __init__(self, url: str, reconnect_attempts: int = 3, reconnect_backoff: float = 0.5):
        """
        Args:
            url: Stream URL (rtsp://, http://, ...).
            reconnect_attempts: Reopen attempts after a failed read before
                giving up and returning (False, None).
            reconnect_backoff: Initial backoff in seconds; doubles per attempt.
        """
        self.url = url
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff = reconnect_backoff
        self.cap = self._open()

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25  # streams often report 0

        logger.info("Stream opened: %s (%dx%d @ %d FPS)", url, self.width, self.height, self.fps)

    def _open(self) -> "cv2.VideoCapture":
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise VideoSourceError(f"Cannot open stream: {self.url}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame, reconnecting with backoff on failure."""
        ok, frame = self.cap.read()
        if ok:
            return ok, frame

        for attempt in range(1, self.reconnect_attempts + 1):
            delay = self.reconnect_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Stream read failed; reconnect %d/%d to %s in %.1fs",
                attempt,
                self.reconnect_attempts,
                self.url,
                delay,
            )
            time.sleep(delay)
            self.cap.release()
            try:
                self.cap = self._open()
            except VideoSourceError:
                continue
            ok, frame = self.cap.read()
            if ok:
                logger.info("Stream reconnected: %s", self.url)
                return ok, frame

        logger.error(
            "Stream lost after %d reconnect attempts: %s", self.reconnect_attempts, self.url
        )
        return False, None

    def get_properties(self) -> dict:
        """Get stream properties."""
        return {
            "source_type": "stream",
            "url": self.url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": -1,  # Unbounded for streams
        }

    def is_opened(self) -> bool:
        """Check if stream is opened."""
        return self.cap.isOpened()

    def release(self):
        """Release stream."""
        self.cap.release()


class LatestFrameGrabber:
    """Wraps a VideoSource with a capture thread that keeps only the newest frame.

    Live cameras produce frames at a fixed rate whether or not inference keeps
    up; reading them synchronously in the processing loop lets frames age in
    the driver buffer and latency grow without bound. This grabber reads the
    source continuously on a daemon thread and exposes the same VideoSource
    surface, so it is a drop-in wrapper:

        source = LatestFrameGrabber(create_video_source("rtsp://..."))
        ok, frame = source.read()   # newest frame, never a stale backlog

    ``read()`` blocks briefly until a frame newer than the last one returned
    is available (each frame is handed out at most once). ``latest()`` is the
    non-blocking variant used by pollers.
    """

    def __init__(self, source: VideoSource, read_timeout: float = 5.0):
        """
        Args:
            source: Any VideoSource; the grabber takes ownership (its
                ``release`` releases the wrapped source).
            read_timeout: Max seconds ``read()`` waits for a new frame before
                reporting failure.
        """
        self._source = source
        self._read_timeout = read_timeout
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_returned_seq = 0
        self._stopped = False
        self._failed = False
        self._thread = threading.Thread(
            target=self._capture_loop, name="frame-grabber", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stopped:
            ok, frame = self._source.read()
            with self._cond:
                if not ok:
                    self._failed = True
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Block until a frame newer than the last returned one arrives."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq > self._last_returned_seq or self._failed or self._stopped,
                timeout=self._read_timeout,
            )
            if self._seq > self._last_returned_seq:
                self._last_returned_seq = self._seq
                return True, self._frame
            return False, None

    def latest(self) -> tuple[int, np.ndarray | None]:
        """Non-blocking: return (sequence_number, newest frame or None)."""
        with self._cond:
            return self._seq, self._frame

    def get_properties(self) -> dict:
        """Properties of the wrapped source."""
        return self._source.get_properties()

    def is_opened(self) -> bool:
        """True while the capture thread is alive and the source is healthy."""
        return not self._failed and not self._stopped and self._source.is_opened()

    def release(self):
        """Stop the capture thread and release the wrapped source."""
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._source.release()

    def __enter__(self) -> "LatestFrameGrabber":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def create_video_source(
    source, *, webcam_width: int = 1280, webcam_height: int = 720, webcam_fps: int = 30
) -> VideoSource:
    """
    Factory function to create the appropriate video source.

    Args:
        source: Device index (int or digit string), stream URL
            (rtsp/http/https/udp/tcp), or a video file path.
        webcam_width: Requested webcam capture width (webcam sources only).
        webcam_height: Requested webcam capture height (webcam sources only).
        webcam_fps: Requested webcam capture FPS (webcam sources only).

    Returns:
        VideoSource instance (WebcamSource, StreamSource, or VideoFileSource)
    """
    if isinstance(source, int):
        return WebcamSource(source, width=webcam_width, height=webcam_height, fps=webcam_fps)

    if isinstance(source, str) and source.isdigit():
        return WebcamSource(int(source), width=webcam_width, height=webcam_height, fps=webcam_fps)

    if isinstance(source, str) and source.lower().startswith(STREAM_SCHEMES):
        return StreamSource(source)

    return VideoFileSource(source)
