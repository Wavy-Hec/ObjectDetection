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
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

logger = logging.getLogger(__name__)

# config.yaml lives at the repository root, one level above this file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class DetectorConfig:
    model_name: str = "yolov8l.pt"
    confidence_threshold: float = 0.20
    image_size: int = 640
    device: str = "auto"
    use_half: bool = True
    target_classes: Optional[List[str]] = None


@dataclass
class TrackerConfig:
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    class_aware: bool = True


@dataclass
class VisualizationConfig:
    show_detections: bool = False
    show_track_ids: bool = True
    show_class_labels: bool = True
    show_confidence: bool = False
    show_trails: bool = False
    trail_length: int = 30
    box_thickness: int = 2
    font_scale: float = 0.6


@dataclass
class VideoConfig:
    input_source: str = "0"
    output_path: Optional[str] = None
    display: bool = True
    save_detections: bool = False
    fps_limit: Optional[float] = None


@dataclass
class PerformanceConfig:
    benchmark_mode: bool = False
    log_interval: int = 30
    save_metrics: bool = False
    metrics_file: str = "metrics.json"


@dataclass
class EvaluationConfig:
    dataset_path: Optional[str] = None
    ground_truth: Optional[str] = None
    save_results: bool = True
    results_format: str = "mot"
    compute_metrics: bool = True


T = TypeVar("T")


def _build(dc_type: Type[T], data: Optional[Dict[str, Any]]) -> T:
    """Build a dataclass from a dict, ignoring unknown keys and filling
    missing keys with the dataclass defaults."""
    if not data:
        return dc_type()  # type: ignore[call-arg]
    known = {f.name for f in fields(dc_type)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s",
                       dc_type.__name__, sorted(unknown))
    kwargs = {k: v for k, v in data.items() if k in known}
    return dc_type(**kwargs)  # type: ignore[call-arg]


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AppConfig":
        data = data or {}
        return cls(
            detector=_build(DetectorConfig, data.get("detector")),
            tracker=_build(TrackerConfig, data.get("tracker")),
            visualization=_build(VisualizationConfig, data.get("visualization")),
            video=_build(VideoConfig, data.get("video")),
            performance=_build(PerformanceConfig, data.get("performance")),
            evaluation=_build(EvaluationConfig, data.get("evaluation")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


def load_config(path: Optional[Path] = None) -> AppConfig:
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
