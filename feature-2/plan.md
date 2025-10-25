# Feature-2 Implementation Plan: SORT Tracker

**Feature:** SORT Multi-Object Tracking  
**Estimated Duration:** 3-4 hours  
**Dependencies:** Feature-1 (detector.py) complete

---

## Task Overview

| Task | Description | Time | Dependencies |
|------|-------------|------|--------------|
| Task 1 | Implement Track dataclass | 15 min | None |
| Task 2 | Implement KalmanBoxTracker | 1.5 hours | Task 1 |
| Task 3 | Implement Tracker class | 1 hour | Task 2 |
| Task 4 | Write unit tests | 1 hour | Task 3 |
| Task 5 | Integration testing | 30 min | Task 4 |
| Task 6 | Documentation & cleanup | 15 min | Task 5 |

**Total Estimated Time:** 4 hours 30 minutes

---

## Task 1: Implement Track Dataclass

**Duration:** 15 minutes  
**Priority:** High  
**Dependencies:** None

### Description
Create the Track dataclass to represent tracked objects with persistent IDs.

### Steps
1. Import dataclass decorator
2. Define Track with fields:
   - id: int
   - bbox: List[float]
   - class_label: str
   - confidence: float
   - age: int
   - hits: int
   - time_since_update: int
3. Add __repr__ method for debugging
4. Add type hints

### Acceptance Criteria
- ✓ Track can be instantiated with all fields
- ✓ Fields are accessible and mutable
- ✓ __repr__ returns readable string
- ✓ Type hints are correct

### Implementation
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Track:
    """Represents a tracked object with persistent ID."""
    id: int
    bbox: List[float]
    class_label: str
    confidence: float
    age: int
    hits: int
    time_since_update: int
    
    def __repr__(self):
        return f"Track(id={self.id}, {self.class_label}, conf={self.confidence:.2f})"
```

---

## Task 2: Implement KalmanBoxTracker

**Duration:** 1.5 hours  
**Priority:** High  
**Dependencies:** Task 1

### Description
Implement Kalman filter for bounding box prediction and update.

### Steps
1. Install filterpy if not present: `pip install filterpy`
2. Define KalmanBoxTracker class
3. Implement __init__:
   - Create KalmanFilter (dim_x=7, dim_z=4)
   - Set state transition matrix F
   - Set measurement matrix H
   - Set noise covariances R, Q, P
   - Initialize state from bbox
4. Implement _bbox_to_z (convert [x1,y1,x2,y2] → [x,y,s,r])
5. Implement _z_to_bbox (convert [x,y,s,r] → [x1,y1,x2,y2])
6. Implement predict():
   - Call kf.predict()
   - Update age, time_since_update
   - Return predicted bbox
7. Implement update(bbox):
   - Convert bbox to measurement
   - Call kf.update()
   - Update hits, hit_streak
8. Implement get_state() to return current bbox

### Acceptance Criteria
- ✓ KalmanBoxTracker initializes correctly
- ✓ _bbox_to_z and _z_to_bbox are inverses
- ✓ predict() returns valid bbox
- ✓ update() modifies internal state
- ✓ No NaN or invalid values
- ✓ Global ID counter increments

### Test Cases
- Initialize with bbox [100, 100, 200, 200]
- Predict without update → bbox should move slightly
- Update with same bbox → state should stabilize
- Predict multiple times → bbox drifts with velocity

---

## Task 3: Implement Tracker Class

**Duration:** 1 hour  
**Priority:** High  
**Dependencies:** Task 2

### Description
Implement main SORT tracking algorithm with data association.

### Steps
1. Define Tracker class with __init__
2. Implement _iou(bbox1, bbox2):
   - Calculate intersection area
   - Calculate union area
   - Return IOU = intersection / union
3. Implement _associate_detections_to_tracks:
   - Build IOU matrix (detections × trackers)
   - Use scipy.optimize.linear_sum_assignment
   - Filter matches below iou_threshold
   - Return matched, unmatched_dets, unmatched_trks
4. Implement update(detections):
   - Predict all existing trackers
   - Associate detections to trackers
   - Update matched trackers
   - Create new trackers for unmatched detections
   - Delete stale trackers (time_since_update > max_age)
   - Return confirmed tracks (hits >= min_hits)

### Acceptance Criteria
- ✓ Tracker initializes with parameters
- ✓ _iou returns correct values for known cases
- ✓ update() handles empty detections
- ✓ update() creates new tracks
- ✓ update() updates existing tracks
- ✓ update() deletes stale tracks
- ✓ Track IDs are persistent

### Test Cases
- Empty detections → no tracks
- 1 detection → 1 new track created
- Same detection 3 times → track confirmed
- Detection disappears → track deleted after max_age
- 2 overlapping detections → correct association

---

## Task 4: Write Unit Tests

**Duration:** 1 hour  
**Priority:** High  
**Dependencies:** Task 3

### Description
Create comprehensive unit tests for tracker module.

### Steps
1. Create test_tracker.py
2. Import pytest, numpy, tracker module
3. Write tests for Track:
   - test_track_creation
   - test_track_repr
4. Write tests for KalmanBoxTracker:
   - test_kalman_init
   - test_bbox_to_z_conversion
   - test_z_to_bbox_conversion
   - test_predict_updates_state
   - test_update_with_detection
   - test_id_counter_increments
5. Write tests for Tracker:
   - test_tracker_init
   - test_iou_computation
   - test_empty_detections
   - test_new_detections_create_tracks
   - test_matched_detections_update_tracks
   - test_stale_tracks_deleted
   - test_track_confirmation (min_hits)
6. Create fixtures:
   - sample_bbox
   - sample_detection
   - tracker_instance

### Acceptance Criteria
- ✓ All tests pass
- ✓ Code coverage > 80%
- ✓ Edge cases tested (empty, NaN, invalid)
- ✓ Fixtures reduce code duplication

### Implementation Outline
```python
# test_tracker.py
import pytest
import numpy as np
from tracker import Track, KalmanBoxTracker, Tracker
from detector import Detection

class TestTrack:
    def test_track_creation(self):
        track = Track(1, [10,20,30,40], "person", 0.9, 5, 3, 0)
        assert track.id == 1
        # ...

class TestKalmanBoxTracker:
    def test_kalman_init(self):
        kf = KalmanBoxTracker([100, 100, 200, 200])
        assert kf.id >= 0
        # ...
    
    def test_bbox_conversions(self):
        # Test _bbox_to_z and _z_to_bbox are inverses
        # ...

class TestTracker:
    @pytest.fixture
    def tracker(self):
        return Tracker(max_age=1, min_hits=3, iou_threshold=0.3)
    
    def test_iou_computation(self):
        # Known IOU values
        # ...
    
    def test_new_tracks_created(self, tracker):
        # ...
```

---

## Task 5: Integration Testing

**Duration:** 30 minutes  
**Priority:** Medium  
**Dependencies:** Task 4

### Description
Test detector + tracker integration with real/synthetic video frames.

### Steps
1. Create integration test script or test function
2. Initialize detector and tracker
3. Create test scenario:
   - Generate frames with moving objects (or use test video)
   - Run detector on each frame
   - Feed detections to tracker
   - Verify track IDs persist
4. Test edge cases:
   - Object enters frame → new track created
   - Object exits frame → track deleted
   - Brief occlusion → track maintained
5. Visualize results (optional but helpful)

### Acceptance Criteria
- ✓ Detector → Tracker pipeline works
- ✓ Track IDs persist across frames
- ✓ New objects get new IDs
- ✓ Exiting objects removed
- ✓ No crashes on edge cases

### Implementation
```python
def test_detector_tracker_integration():
    from detector import ObjectDetector
    from tracker import Tracker
    import cv2
    
    detector = ObjectDetector()
    tracker = Tracker()
    
    # Load test video or images
    cap = cv2.VideoCapture('test_video.mp4')
    
    track_ids_seen = set()
    
    for i in range(30):  # 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        
        for track in tracks:
            track_ids_seen.add(track.id)
    
    # Verify tracking worked
    assert len(track_ids_seen) > 0
    # ...
```

---

## Task 6: Documentation & Cleanup

**Duration:** 15 minutes  
**Priority:** Low  
**Dependencies:** Task 5

### Description
Finalize documentation and clean up code.

### Steps
1. Ensure all public methods have docstrings
2. Add module-level docstring to tracker.py
3. Update README.md with tracker usage
4. Create simple usage example
5. Run code formatter (black or autopep8)
6. Remove debug print statements
7. Check PEP 8 compliance

### Acceptance Criteria
- ✓ All functions documented
- ✓ Code formatted consistently
- ✓ No debug prints
- ✓ README updated with examples
- ✓ Type hints complete

### Documentation Template
```python
"""
SORT tracker module for multi-object tracking.

This module implements the SORT (Simple Online and Realtime Tracking) algorithm
using Kalman filters for motion prediction and IOU matching for data association.

Example:
    from detector import ObjectDetector
    from tracker import Tracker
    
    detector = ObjectDetector()
    tracker = Tracker(max_age=1, min_hits=3, iou_threshold=0.3)
    
    for frame in video:
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        
        for track in tracks:
            print(f"ID {track.id}: {track.class_label}")
"""
```

---

## Validation Checklist

Before marking feature-2 complete:

### Functional
- [ ] Track dataclass implemented
- [ ] KalmanBoxTracker predict/update works
- [ ] Tracker.update() returns List[Track]
- [ ] IDs are persistent across frames
- [ ] New detections create new tracks
- [ ] Stale tracks are deleted

### Testing
- [ ] Unit tests pass (> 80% coverage)
- [ ] Integration test passes
- [ ] Edge cases handled (empty, NaN, etc.)

### Documentation
- [ ] All classes/methods documented
- [ ] README updated
- [ ] Usage example provided

### Performance
- [ ] Runs at > 30 FPS (detector + tracker)
- [ ] No memory leaks
- [ ] Reasonable tracking accuracy

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Kalman filter instability | Use proven filterpy library; set conservative noise params |
| Poor IOU matching | Tune iou_threshold; consider Hungarian algorithm |
| Too many ID switches | Adjust max_age and min_hits parameters |
| Integration issues | Test detector output format first |

---

## Next Steps After Completion

1. Merge feature-2 branch to main
2. Update PAD.md with tracker details
3. Create post_development_analysis.md
4. Plan feature-3 (integration pipeline, visualization)

---

**Created:** 2025-01-XX  
**Status:** Ready to execute
