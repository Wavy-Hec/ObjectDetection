"""ObjectDetection — real-time object detection and multi-object tracking.

This package bundles detection (YOLOv8 / YOLO-World), SORT tracking,
video I/O, visualization, a reusable processing pipeline, and typed config.

Public names are exported lazily so that importing the package (or a light
submodule like :mod:`src.config`) does not eagerly pull heavy ML dependencies
(``torch`` / ``ultralytics`` / ``filterpy``) unless they are actually used.
"""

import importlib
from typing import Any

__all__ = [
    # detector
    "ObjectDetector", "Detection", "needs_yolo_world", "COCO_CLASSES",
    # tracker
    "Tracker", "Track", "KalmanBoxTracker",
    # video source
    "create_video_source", "VideoSource", "VideoSourceError",
    "WebcamSource", "VideoFileSource",
    # visualization
    "draw_tracks", "draw_detections", "draw_segmentation_masks",
    "get_color_for_track",
    # pipeline
    "Pipeline", "ProcessResult", "FrameStats", "FPSTracker",
    # config + logging
    "AppConfig", "load_config", "setup_logging",
]

# Map each exported name to the submodule that defines it.
_EXPORTS = {
    "ObjectDetector": "detector", "Detection": "detector",
    "needs_yolo_world": "detector", "COCO_CLASSES": "detector",
    "Tracker": "tracker", "Track": "tracker", "KalmanBoxTracker": "tracker",
    "create_video_source": "video_source", "VideoSource": "video_source",
    "VideoSourceError": "video_source", "WebcamSource": "video_source",
    "VideoFileSource": "video_source",
    "draw_tracks": "visualization", "draw_detections": "visualization",
    "draw_segmentation_masks": "visualization", "get_color_for_track": "visualization",
    "Pipeline": "pipeline", "ProcessResult": "pipeline",
    "FrameStats": "pipeline", "FPSTracker": "pipeline",
    "AppConfig": "config", "load_config": "config",
    "setup_logging": "logging_config",
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
