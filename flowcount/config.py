"""Typed configuration for the object tracking system.

Loads ``config.yaml`` into dataclasses so the rest of the codebase reads
defaults from one place. CLI flags override these defaults (see ``main.py``).

Uses only the standard library + ``pyyaml`` (already a project dependency) —
no extra packages required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

logger = logging.getLogger(__name__)

# config.yaml lives at the repository root, one level above this file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class DetectorConfig:
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.20
    image_size: int = 640
    device: str = "auto"
    use_half: bool = True
    target_classes: list[str] | None = None


@dataclass
class TrackerConfig:
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    class_aware: bool = True
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1


@dataclass
class VisualizationConfig:
    show_trails: bool = False


@dataclass
class VideoConfig:
    output_path: str | None = None
    display: bool = True
    fps_limit: float | None = None
    webcam_width: int = 1280
    webcam_height: int = 720
    webcam_fps: int = 30


T = TypeVar("T")


def _build(dc_type: type[T], data: dict[str, Any] | None) -> T:
    """Build a dataclass from a dict, ignoring unknown keys and filling
    missing keys with the dataclass defaults."""
    if not data:
        return dc_type()  # type: ignore[call-arg]
    known = {f.name for f in fields(dc_type)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s", dc_type.__name__, sorted(unknown))
    kwargs = {k: v for k, v in data.items() if k in known}
    return dc_type(**kwargs)  # type: ignore[call-arg]


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AppConfig:
        data = data or {}
        return cls(
            detector=_build(DetectorConfig, data.get("detector")),
            tracker=_build(TrackerConfig, data.get("tracker")),
            visualization=_build(VisualizationConfig, data.get("visualization")),
            video=_build(VideoConfig, data.get("video")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> AppConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML, falling back to defaults if absent.

    Args:
        path: Path to a YAML config file. Defaults to the repo's
            ``config.yaml``.

    Returns:
        A populated :class:`AppConfig` (all defaults if the file is missing).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning("Config file not found at %s; using built-in defaults", path)
        return AppConfig()
    logger.info("Loaded configuration from %s", path)
    return AppConfig.from_yaml(path)
