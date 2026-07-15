"""FlowCount — real-time object detection, tracking, and traffic analytics.

This package bundles detection (YOLO11/YOLOv8 / YOLO-World), ByteTrack-style
tracking on a SORT core, video I/O (webcam / file / RTSP), visualization, a
reusable processing pipeline, analytics, and typed config.

Public names are exported lazily so that importing the package (or a light
submodule like :mod:`flowcount.config`) does not eagerly pull heavy ML dependencies
(``torch`` / ``ultralytics`` / ``filterpy``) unless they are actually used.
"""

import importlib
from typing import Any

__version__ = "0.4.0"

__all__ = [
    # detector
    "ObjectDetector", "Detection", "needs_yolo_world", "COCO_CLASSES",
    # tracker
    "Tracker", "Track", "KalmanBoxTracker",
    # video source
    "create_video_source", "VideoSource", "VideoSourceError",
    "WebcamSource", "VideoFileSource", "StreamSource", "LatestFrameGrabber",
    # visualization
    "draw_tracks", "draw_segmentation_masks", "get_color_for_track",
    # pipeline
    "Pipeline", "ProcessResult", "FrameStats", "FPSTracker",
    # config + logging
    "AppConfig", "load_config", "setup_logging",
    # analytics
    "AnalyticsManager", "LineCrossingCounter", "Zone", "ZoneManager",
    "HeatmapAccumulator", "EventClipRecorder", "CSVExporter", "SQLiteExporter",
    "Event", "FrameContext",
]

# Map each exported name to the submodule that defines it.
_EXPORTS = {
    "ObjectDetector": "detector", "Detection": "detector",
    "needs_yolo_world": "detector", "COCO_CLASSES": "detector",
    "Tracker": "tracker", "Track": "tracker", "KalmanBoxTracker": "tracker",
    "create_video_source": "video_source", "VideoSource": "video_source",
    "VideoSourceError": "video_source", "WebcamSource": "video_source",
    "VideoFileSource": "video_source", "StreamSource": "video_source",
    "LatestFrameGrabber": "video_source",
    "draw_tracks": "visualization",
    "draw_segmentation_masks": "visualization", "get_color_for_track": "visualization",
    "Pipeline": "pipeline", "ProcessResult": "pipeline",
    "FrameStats": "pipeline", "FPSTracker": "pipeline",
    "AppConfig": "config", "load_config": "config",
    "setup_logging": "logging_config",
    "AnalyticsManager": "analytics", "LineCrossingCounter": "analytics",
    "Zone": "analytics", "ZoneManager": "analytics",
    "HeatmapAccumulator": "analytics", "EventClipRecorder": "analytics",
    "CSVExporter": "analytics", "SQLiteExporter": "analytics",
    "Event": "analytics", "FrameContext": "analytics",
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
