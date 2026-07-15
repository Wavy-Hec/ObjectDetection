"""Tests for the dataclass-based configuration loader."""

from pathlib import Path

from src.config import AppConfig, load_config

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def test_defaults():
    cfg = AppConfig()
    assert cfg.detector.model_name == "yolo11n.pt"
    assert cfg.tracker.max_age == 30
    assert cfg.tracker.class_aware is True


def test_from_dict_partial_keeps_other_defaults():
    cfg = AppConfig.from_dict({"tracker": {"max_age": 99}})
    assert cfg.tracker.max_age == 99
    assert cfg.tracker.min_hits == 3  # untouched default


def test_from_dict_ignores_unknown_keys():
    cfg = AppConfig.from_dict({"detector": {"model_name": "x.pt", "bogus": 1}})
    assert cfg.detector.model_name == "x.pt"


def test_from_dict_ignores_unknown_sections():
    cfg = AppConfig.from_dict({"made_up_section": {"a": 1}})
    assert cfg.detector.model_name == "yolo11n.pt"


def test_load_repo_config_matches_yaml():
    cfg = load_config(REPO_CONFIG)
    assert cfg.detector.confidence_threshold == 0.20
    assert cfg.tracker.iou_threshold == 0.3
    assert cfg.evaluation.results_format == "mot"


def test_load_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.detector.model_name == "yolo11n.pt"
