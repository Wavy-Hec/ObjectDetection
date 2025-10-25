# Product Requirements Document: SORT Tracking (Feature-2)

**Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Status:** Active Development  
**Feature Number:** feature-2

---

## Executive Summary

Feature-2 implements the SORT (Simple Online and Realtime Tracking) algorithm to track detected objects across video frames. This builds on the detection capabilities from feature-1 by adding persistent object IDs, trajectory prediction using Kalman filters, and data association through IOU matching.

---

## 1. Feature Overview

### 1.1 Objective
Implement multi-object tracking to maintain consistent identities for detected people and cars across video frames, enabling:
- Persistent tracking with unique IDs
- Trajectory prediction during brief occlusions
- Robust association of detections to tracks

### 1.2 User Stories
- As a traffic analyst, I want to count unique vehicles passing through a scene, so that I can measure traffic flow
- As a security operator, I want to follow specific individuals across frames, so that I can monitor their movements
- As a researcher, I want stable track IDs, so that I can analyze object trajectories over time

### 1.3 Use Cases
- Vehicle counting at intersections
- Pedestrian flow analysis in public spaces
- Trajectory analysis for autonomous vehicle testing
- Security and surveillance applications

### 1.4 Out of Scope (Deferred to Feature-3+)
- Deep SORT with appearance features
- Cross-camera tracking
- Long-term re-identification after occlusion
- 3D trajectory estimation

---

## 2. Functional Requirements

### 2.1 Tracking Core
**REQ-TRK-001:** The system shall implement the SORT algorithm as described in https://arxiv.org/abs/1602.00763

**REQ-TRK-002:** The system shall assign unique, sequential integer IDs to each tracked object

**REQ-TRK-003:** The system shall maintain track state using Kalman filters for position and velocity

**REQ-TRK-004:** The system shall use IOU (Intersection over Union) for detection-to-track association

**REQ-TRK-005:** The system shall support configurable IOU threshold (default: 0.3)

### 2.2 Track Lifecycle
**REQ-LIFE-001:** The system shall create new tracks for unmatched detections with confidence > threshold

**REQ-LIFE-002:** The system shall update existing tracks when matched with new detections

**REQ-LIFE-003:** The system shall predict track positions using Kalman filter when no detection is matched

**REQ-LIFE-004:** The system shall delete tracks after N consecutive frames without detection (default: max_age=1)

**REQ-LIFE-005:** The system shall require min_hits=3 consecutive detections before displaying a track

### 2.3 Data Structures
**REQ-DATA-001:** Track objects shall contain: id, bbox, class_label, confidence, age, hits, time_since_update

**REQ-DATA-002:** The system shall maintain a list of active tracks updated each frame

**REQ-DATA-003:** The system shall return List[Track] with current frame's visible tracks

### 2.4 Integration with Detector
**REQ-INT-001:** The tracker shall accept List[Detection] from detector.py as input

**REQ-INT-002:** The tracker shall preserve class labels from detections

**REQ-INT-003:** The tracker shall preserve confidence scores from detections

**REQ-INT-004:** The tracker shall operate independently of the detection module

---

## 3. Non-Functional Requirements

### 3.1 Performance Requirements
**REQ-PERF-001:** Tracking overhead shall add < 5ms per frame on GPU

**REQ-PERF-002:** The system shall maintain > 30 FPS with detection + tracking on RTX 3050

**REQ-PERF-003:** Maximum of 100 simultaneous tracks supported

### 3.2 Accuracy Requirements
**REQ-ACC-001:** ID switches shall be < 5 per minute on MOT16 dataset

**REQ-ACC-002:** MOTA (Multiple Object Tracking Accuracy) shall be > 0.70 on test sequences

**REQ-ACC-003:** Track fragmentation shall be minimal (tracks should persist across brief occlusions)

### 3.3 Code Quality
**REQ-QUAL-001:** tracker.py shall have > 80% test coverage

**REQ-QUAL-002:** All public methods shall have docstrings

**REQ-QUAL-003:** Code shall follow PEP 8 style guidelines

---

## 4. Technical Architecture

### 4.1 Module: tracker.py

**Classes:**

```python
class Track:
    """Represents a tracked object."""
    id: int                    # Unique track ID
    bbox: List[float]          # [x1, y1, x2, y2]
    class_label: str           # "person" or "car"
    confidence: float          # Detection confidence [0, 1]
    age: int                   # Frames since creation
    hits: int                  # Consecutive successful updates
    time_since_update: int     # Frames since last detection
    
class KalmanBoxTracker:
    """Kalman filter for bounding box tracking."""
    # State: [x, y, s, r, vx, vy, vs]
    # x, y: center position
    # s: scale (area)
    # r: aspect ratio (width/height)
    # vx, vy, vs: velocities
    
    def __init__(self, bbox: List[float])
    def predict() -> np.ndarray  # Predict next position
    def update(bbox: List[float])  # Update with detection
    def get_state() -> List[float]  # Get current bbox
    
class Tracker:
    """Main SORT tracker."""
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3)
    def update(detections: List[Detection]) -> List[Track]
    def _associate_detections_to_tracks(...)  # IOU matching
    def _iou(bbox1, bbox2) -> float  # Compute IOU
```

### 4.2 Algorithm Flow

```
Input: List[Detection] from detector
↓
1. Predict existing track positions (Kalman predict)
↓
2. Compute IOU matrix between detections and predicted tracks
↓
3. Hungarian algorithm for optimal matching (or greedy)
↓
4. Update matched tracks with detections (Kalman update)
↓
5. Create new tracks for unmatched detections
↓
6. Delete tracks with time_since_update > max_age
↓
7. Filter tracks with hits < min_hits
↓
Output: List[Track] for current frame
```

---

## 5. Dependencies

### 5.1 New Dependencies
| Package | Purpose |
|---------|---------|
| filterpy | Kalman filter implementation |
| scipy | Hungarian algorithm (linear_sum_assignment) |

### 5.2 Existing Dependencies (from feature-1)
- numpy (array operations)
- ultralytics (YOLOv5 detector)
- torch (GPU acceleration)

---

## 6. Success Metrics

### 6.1 Functional Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Track creation | Works for all unmatched detections | Unit test |
| Track update | Correctly updates matched tracks | Unit test |
| Track deletion | Removes stale tracks after max_age | Unit test |
| IOU matching | Correctly associates detections | Unit test |

### 6.2 Performance Metrics
| Metric | Target | Dataset |
|--------|--------|---------|
| MOTA | > 0.70 | MOT16 subset |
| ID Switches | < 5/min | Custom test video |
| FPS | > 30 | RTX 3050 |
| Latency overhead | < 5ms/frame | Profiling |

---

## 7. Testing Strategy

### 7.1 Unit Tests
- Test Track class creation and updates
- Test KalmanBoxTracker predict/update
- Test IOU computation
- Test Tracker.update() with various scenarios:
  - No detections (all tracks predict)
  - All new detections (create new tracks)
  - Perfect matches (update existing)
  - Partial matches (update + create + delete)

### 7.2 Integration Tests
- Test detector + tracker pipeline
- Test with real video frames
- Test edge cases (empty frames, dense crowds)

---

## 8. Acceptance Criteria

The feature is complete when:
- ✓ tracker.py module implemented with all required classes
- ✓ Unit tests pass with > 80% coverage
- ✓ Integration test shows detector→tracker working
- ✓ IDs persist across frames in demo video
- ✓ Performance: > 30 FPS with tracking on GPU
- ✓ Code documented and follows PEP 8

---

## 9. Timeline

**Estimated Duration:** 3-4 hours

### Task Breakdown:
1. **Implement Track class** (30 min)
2. **Implement KalmanBoxTracker** (1 hour)
3. **Implement Tracker.update()** (1 hour)
4. **Write unit tests** (45 min)
5. **Integration testing** (30 min)
6. **Documentation and cleanup** (15 min)

---

## 10. References

1. SORT Paper: https://arxiv.org/abs/1602.00763
2. Kalman Filter Tutorial: https://www.kalmanfilter.net/
3. filterpy Documentation: https://filterpy.readthedocs.io/
4. MOT Challenge: https://motchallenge.net/

---

**Document Version:** 1.0  
**Author:** Feature-2 Team  
**Status:** Active Development
