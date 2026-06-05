"""Tests for the detector's pure geometry/routing helpers.

These exercise ``needs_yolo_world`` and the static NMS filter without loading
any model, so they run without ``ultralytics`` installed (it is imported
lazily inside ``ObjectDetector.__init__``).
"""

from src.detector import COCO_CLASSES, Detection, ObjectDetector, needs_yolo_world


def test_needs_yolo_world_none():
    assert needs_yolo_world(None) is False


def test_needs_yolo_world_coco_subset():
    assert needs_yolo_world({"person", "car", "cell phone"}) is False


def test_needs_yolo_world_custom_class():
    assert needs_yolo_world({"person", "screwdriver"}) is True


def test_coco_classes_sane():
    assert "person" in COCO_CLASSES
    assert len(COCO_CLASSES) == 80


def test_nms_removes_overlapping_lower_confidence():
    d_high = Detection([0, 0, 100, 100], "person", 0.9)
    d_low = Detection([5, 5, 105, 105], "person", 0.6)  # ~0.85 IoU with d_high
    kept = ObjectDetector._nms_filter([d_high, d_low], iou_thresh=0.5)
    assert len(kept) == 1
    assert kept[0].confidence == 0.9  # higher-confidence box survives


def test_nms_keeps_disjoint_boxes():
    d1 = Detection([0, 0, 50, 50], "person", 0.9)
    d2 = Detection([200, 200, 260, 260], "car", 0.8)
    kept = ObjectDetector._nms_filter([d1, d2], iou_thresh=0.5)
    assert len(kept) == 2


def test_nms_empty_input():
    assert ObjectDetector._nms_filter([], iou_thresh=0.5) == []
