# Task 6: Unit Testing and Validation

## Overview
Write comprehensive unit tests and evaluation routines to ensure modules behave correctly and to quantify tracking performance against ground-truth annotations. This validates system functionality and provides metrics for reporting.

## Priority
**MEDIUM** - Quality assurance before deployment

## Dependencies
- Task 3: Detection Module (detector.py)
- Task 4: Tracking Module (tracker.py)
- Task 5: Integration Pipeline (main.py)

## Inputs
- Source code from all modules
- Test images and videos
- Ground-truth annotations (MOT16 or custom)
- Test fixtures and mock data

## Outputs
- Unit tests for detector, tracker, main modules
- Test coverage report (>80% target)
- evaluation.py script for MOTA/MOTP calculation
- Performance metrics summary document

## Implementation Steps

### 1. Setup Testing Framework
```bash
pip install pytest pytest-cov

# Create test structure
tests/
├── __init__.py
├── test_detector.py
├── test_tracker.py
├── test_integration.py
├── test_evaluation.py
└── fixtures/
    ├── test_images/
    └── test_videos/
```

### 2. Detector Unit Tests
```python
# tests/test_detector.py

import pytest
import numpy as np
from src.detector import ObjectDetector, Detection

def test_detector_initialization():
    """Test detector initializes correctly."""
    detector = ObjectDetector()
    assert detector.conf_threshold == 0.25
    assert detector.img_size == (640, 480)

def test_detect_empty_frame():
    """Test detection on empty frame."""
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)

def test_detection_class():
    """Test Detection data class."""
    det = Detection([100, 100, 200, 200], "person", 0.9)
    assert det.class_label == "person"
    assert det.confidence == 0.9
    assert len(det.bbox) == 4

def test_confidence_filtering():
    """Test detections filtered by confidence."""
    detector = ObjectDetector(conf_threshold=0.5)
    # Test with real image containing objects
    # Verify all returned detections have confidence >= 0.5

def test_class_filtering():
    """Test only person and car classes returned."""
    detector = ObjectDetector()
    # Test with image containing multiple object types
    # Verify only person and car in results
```

### 3. Tracker Unit Tests
```python
# tests/test_tracker.py

import pytest
from src.tracker import Tracker, Track, compute_iou
from src.detector import Detection

def test_iou_calculation():
    """Test IOU computation."""
    bbox1 = [0, 0, 10, 10]
    bbox2 = [5, 5, 15, 15]
    iou = compute_iou(bbox1, bbox2)
    assert 0 < iou < 1

def test_tracker_initialization():
    """Test tracker initializes correctly."""
    tracker = Tracker(max_age=30, min_hits=3)
    assert tracker.max_age == 30
    assert tracker.min_hits == 3

def test_track_creation():
    """Test new track created from detection."""
    tracker = Tracker(min_hits=1)
    det = Detection([100, 100, 200, 200], "person", 0.9)
    tracks = tracker.update([det])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1

def test_track_persistence():
    """Test track ID persists across frames."""
    tracker = Tracker(min_hits=1)
    det1 = Detection([100, 100, 200, 200], "person", 0.9)
    tracks1 = tracker.update([det1])
    
    det2 = Detection([105, 105, 205, 205], "person", 0.9)
    tracks2 = tracker.update([det2])
    
    assert tracks1[0].track_id == tracks2[0].track_id

def test_track_deletion():
    """Test old tracks deleted."""
    tracker = Tracker(max_age=5, min_hits=1)
    det = Detection([100, 100, 200, 200], "person", 0.9)
    tracker.update([det])
    
    # Update with empty detections for max_age frames
    for _ in range(6):
        tracker.update([])
    
    tracks = tracker.update([])
    assert len(tracks) == 0
```

### 4. Integration Tests
```python
# tests/test_integration.py

import pytest
import cv2
import tempfile
from src.main import main

def test_video_processing():
    """Test full pipeline on test video."""
    # Create temporary test video
    # Run main pipeline
    # Verify output video created
    # Check video has same frame count

def test_webcam_fallback():
    """Test webcam mode with mock."""
    # Mock cv2.VideoCapture
    # Test initialization
```

### 5. Evaluation Script (evaluation.py)
```python
# src/evaluation.py

import motmetrics as mm
import pandas as pd

class TrackingEvaluator:
    """Evaluate tracking performance against ground truth."""
    
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)
    
    def update(self, ground_truth_boxes, ground_truth_ids, 
               predicted_boxes, predicted_ids):
        """Update accumulator with frame results."""
        # Compute distance matrix
        distances = mm.distances.iou_matrix(
            ground_truth_boxes, predicted_boxes, max_iou=0.5
        )
        
        self.acc.update(
            ground_truth_ids, predicted_ids, distances
        )
    
    def compute_metrics(self):
        """Compute MOTA, MOTP, and other metrics."""
        mh = mm.metrics.create()
        summary = mh.compute(
            self.acc,
            metrics=['mota', 'motp', 'num_frames', 'num_switches', 
                     'num_false_positives', 'num_misses'],
            name='tracking'
        )
        return summary

def evaluate_video(video_path, ground_truth_path, detector, tracker):
    """Evaluate tracking on annotated video."""
    evaluator = TrackingEvaluator()
    
    # Load video and annotations
    # For each frame:
    #   - Get ground truth boxes and IDs
    #   - Run detector and tracker
    #   - Update evaluator
    
    # Compute and return metrics
    metrics = evaluator.compute_metrics()
    return metrics
```

### 6. Run Tests and Coverage
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html  # or browse to file
```

## Acceptance Criteria

### Must Have
- [ ] Unit tests for detector module (>5 tests)
- [ ] Unit tests for tracker module (>5 tests)
- [ ] Integration test for main pipeline
- [ ] Test coverage >80% for all modules
- [ ] All tests pass without errors
- [ ] evaluation.py computes MOTA and MOTP
- [ ] Metrics calculated on at least one MOT16 sequence

### Should Have
- [ ] Tests run in <30 seconds total
- [ ] Test fixtures organized in tests/fixtures/
- [ ] Parametrized tests for multiple scenarios
- [ ] Mock external dependencies (video I/O)
- [ ] Continuous integration ready (GitHub Actions)

### Nice to Have
- [ ] Property-based testing (Hypothesis)
- [ ] Performance benchmarks
- [ ] Visual regression tests (compare output frames)
- [ ] Automated metric comparison against baselines

## Test Coverage Goals

| Module | Coverage Target | Priority |
|--------|----------------|----------|
| detector.py | >85% | High |
| tracker.py | >80% | High |
| main.py | >70% | Medium |
| evaluation.py | >75% | Medium |

## Validation Commands

```bash
# Run specific test file
pytest tests/test_detector.py -v

# Run tests matching pattern
pytest tests/ -k "test_track" -v

# Run with output
pytest tests/ -v -s

# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run evaluation on test video
python src/evaluation.py --video data/input/MOT16-02.mp4 --gt data/annotations/MOT16-02.txt
```

## Performance Benchmarks

| Benchmark | Target | Measurement |
|-----------|--------|-------------|
| Detector speed | <50ms/frame | Average over 100 frames (GPU) |
| Tracker speed | <10ms/frame | With 10 active tracks |
| End-to-end | >30 FPS | Full pipeline on test video |
| MOTA | >0.80 | On MOT16-02 sequence |
| ID switches | <5/min | On MOT16-02 sequence |

## Test Data Requirements

### Test Images
- Image with single person
- Image with single car
- Image with multiple people and cars
- Image with no target objects
- Image with occlusions

### Test Videos
- Short clip (10 seconds) with simple tracking
- MOT16-02 sequence (or subset) with annotations
- Video with crossing paths
- Video with objects entering/leaving frame

### Ground Truth Annotations
Format (MOT Challenge):
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```

Example:
```
1, 1, 100, 100, 50, 100, 1, 1, 0.8
1, 2, 300, 200, 60, 120, 1, 1, 1.0
2, 1, 105, 102, 50, 100, 1, 1, 0.7
```

## Estimated Time
- Setting up test framework: 1 hour
- Writing detector tests: 2 hours
- Writing tracker tests: 3 hours
- Integration tests: 2 hours
- Evaluation script: 2 hours
- Running metrics on datasets: 1 hour
- **Total: 11 hours**

## Success Metrics
- All tests pass: 100%
- Code coverage: >80%
- Test execution time: <1 minute
- MOTA on MOT16-02: >0.80
- Zero regressions from changes

## Common Issues & Solutions

### Issue 1: Tests fail due to random initialization
**Solution:** Set random seeds in tests (np.random.seed, torch.manual_seed)

### Issue 2: Coverage doesn't include visualization code
**Solution:** Mock cv2.imshow/waitKey, test drawing functions separately

### Issue 3: Evaluation metrics vary between runs
**Solution:** Use deterministic dataset splits, fix random seeds

### Issue 4: MOT16 annotations not loading
**Solution:** Verify format matches expected, handle edge cases (missing frames)

### Issue 5: Tests too slow
**Solution:** Use smaller test videos, mock expensive operations, parallelize tests

## Integration Notes
- Tests import modules from src/ directory
- Use pytest fixtures for shared test data
- Mock video I/O to avoid file dependencies
- Evaluation script uses motmetrics library
- Ground truth format follows MOT Challenge specification

## Documentation

### Writing New Tests
1. Import module to test
2. Create test function with `test_` prefix
3. Setup test data (fixtures or inline)
4. Call function under test
5. Assert expected behavior

### Running Tests in CI/CD
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=src
```

## Related Tasks
- Task 3: Detection Module (test target)
- Task 4: Tracking Module (test target)
- Task 5: Integration Pipeline (test target)
- Task 7: Demo and Presentation (uses metrics results)

## Status
⏳ **PENDING** - Awaits completion of Tasks 3, 4, 5
