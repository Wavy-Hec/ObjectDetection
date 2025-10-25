# Feature-2 Design Document: SORT Tracker Implementation

**Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Status:** Active

---

## 1. Design Overview

This document details the technical design for implementing the SORT (Simple Online and Realtime Tracking) algorithm in `tracker.py`. The tracker will consume detections from `detector.py` and maintain persistent object IDs across frames.

---

## 2. Architecture

### 2.1 System Integration

```
Video Frame
    ↓
detector.detect(frame)
    ↓
List[Detection]
    ↓
tracker.update(detections)
    ↓
List[Track]  # Objects with persistent IDs
    ↓
Visualization
```

### 2.2 Module Structure

```
project/src/tracker.py
│
├── Track (dataclass)
│   └── Stores track state and metadata
│
├── KalmanBoxTracker
│   ├── __init__(bbox)
│   ├── predict() → predicted_bbox
│   ├── update(bbox)
│   └── get_state() → current_bbox
│
└── Tracker
    ├── __init__(max_age, min_hits, iou_threshold)
    ├── update(detections) → tracks
    ├── _associate_detections_to_tracks(...)
    └── _iou(bbox1, bbox2) → float
```

---

## 3. Detailed Design

### 3.1 Track Class

**Purpose:** Container for track state and metadata

```python
@dataclass
class Track:
    """Represents a tracked object with persistent ID."""
    
    id: int                    # Unique identifier
    bbox: List[float]          # [x1, y1, x2, y2]
    class_label: str           # "person" or "car"
    confidence: float          # Latest detection confidence
    age: int                   # Total frames since creation
    hits: int                  # Consecutive successful updates
    time_since_update: int     # Frames since last detection match
    
    def __repr__(self):
        return f"Track(id={self.id}, {self.class_label}, conf={self.confidence:.2f})"
```

**Design Decisions:**
- Use dataclass for simplicity and auto-generated methods
- Store original class_label and confidence from detection
- Maintain both `age` (total lifetime) and `time_since_update` (staleness)

---

### 3.2 KalmanBoxTracker Class

**Purpose:** Kalman filter for bounding box state estimation

**State Vector (7D):**
```
[x, y, s, r, vx, vy, vs]

x, y:     Center coordinates
s:        Scale (area = width * height)
r:        Aspect ratio (width / height)
vx, vy:   Velocity of center
vs:       Velocity of scale
```

**Implementation:**
```python
class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes."""
    
    count = 0  # Global track ID counter
    
    def __init__(self, bbox: List[float]):
        """
        Initialize Kalman filter with initial bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        from filterpy.kalman import KalmanFilter
        
        # Initialize Kalman filter
        # State: [x, y, s, r, vx, vy, vs] (7 dimensions)
        # Observation: [x, y, s, r] (4 dimensions)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1],  # s = s + vs
            [0, 0, 0, 1, 0, 0, 0],  # r = r (constant)
            [0, 0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1],  # vs = vs
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        
        # Measurement noise (high for position, low for aspect ratio)
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise (uncertainty in velocity)
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty in velocities
        self.kf.P *= 10.0
        
        # Covariance
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state from bbox
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def _bbox_to_z(self, bbox: List[float]) -> np.ndarray:
        """
        Convert [x1, y1, x2, y2] to [x, y, s, r].
        
        x, y: center
        s: scale (area)
        r: aspect ratio (w/h)
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # scale (area)
        r = w / float(h)  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _z_to_bbox(self, z: np.ndarray) -> List[float]:
        """
        Convert [x, y, s, r] to [x1, y1, x2, y2].
        """
        x, y, s, r = z[0], z[1], z[2], z[3]
        w = np.sqrt(s * r)
        h = s / w
        return [
            float(x - w / 2.0),  # x1
            float(y - h / 2.0),  # y1
            float(x + w / 2.0),  # x2
            float(y + h / 2.0),  # y2
        ]
    
    def predict(self) -> np.ndarray:
        """
        Predict next state using Kalman filter.
        
        Returns:
            Predicted bbox [x1, y1, x2, y2]
        """
        # Handle velocity decay for stationary objects
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self._z_to_bbox(self.kf.x[:4])
    
    def update(self, bbox: List[float]):
        """
        Update Kalman filter with new detection.
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
    
    def get_state(self) -> List[float]:
        """
        Get current bounding box estimate.
        
        Returns:
            bbox [x1, y1, x2, y2]
        """
        return self._z_to_bbox(self.kf.x[:4])
```

**Design Decisions:**
- Use filterpy library (robust, well-tested implementation)
- Constant velocity motion model (simple but effective)
- High process noise for velocities (handles erratic motion)
- Convert bbox ↔ [x, y, s, r] representation (better for Kalman)

---

### 3.3 Tracker Class

**Purpose:** Main SORT tracking algorithm

```python
class Tracker:
    """SORT tracker for multi-object tracking."""
    
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # List of KalmanBoxTracker
        self.frame_count = 0
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of Track objects for current frame
        """
        self.frame_count += 1
        
        # Get predicted positions from existing trackers
        trks = np.zeros((len(self.trackers), 5))  # [x1, y1, x2, y2, id]
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, trks
        )
        
        # Update matched trackers with detections
        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(detections[det_idx].bbox)
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i].bbox)
            self.trackers.append(trk)
        
        # Prepare output tracks
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(self.trackers.index(trk))
                continue
            
            # Only return tracks with enough hits
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                # Find corresponding detection for class and confidence
                class_label = "unknown"
                confidence = 0.0
                
                for m in matched:
                    if self.trackers.index(trk) == m[1]:
                        det = detections[m[0]]
                        class_label = det.class_label
                        confidence = det.confidence
                        break
                
                track = Track(
                    id=trk.id,
                    bbox=d,
                    class_label=class_label,
                    confidence=confidence,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update
                )
                ret.append(track)
        
        return ret
    
    def _associate_detections_to_tracks(self, detections, trackers):
        """
        Assign detections to tracked objects using IOU.
        
        Returns:
            matched: [[det_idx, trk_idx], ...]
            unmatched_dets: [det_idx, ...]
            unmatched_trks: [trk_idx, ...]
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        # Compute IOU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det.bbox, trk[:4])
        
        # Use Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        
        if min(iou_matrix.shape) > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)  # Maximize IOU
            matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter matches below IOU threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    @staticmethod
    def _iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute Intersection over Union between two bboxes.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
            
        Returns:
            IOU value [0, 1]
        """
        # Intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
```

---

## 4. Design Patterns

### 4.1 Kalman Filter Pattern
- **Predict-Update Cycle:** Standard Kalman filter workflow
- **State Representation:** [x, y, s, r] more stable than [x1, y1, x2, y2]

### 4.2 Data Association Pattern
- **Hungarian Algorithm:** Optimal bipartite matching
- **IOU Metric:** Simple, effective for bounding boxes
- **Threshold Filtering:** Reject poor matches

### 4.3 Track Lifecycle Pattern
- **Three States:** Tentative → Confirmed → Deleted
- **min_hits:** Prevents false track creation
- **max_age:** Removes stale tracks

---

## 5. Integration Points

### 5.1 Input: detector.py
```python
from detector import Detection, ObjectDetector

detector = ObjectDetector()
detections = detector.detect(frame)  # List[Detection]
```

### 5.2 Output: main.py (future)
```python
from tracker import Tracker

tracker = Tracker(max_age=1, min_hits=3, iou_threshold=0.3)
tracks = tracker.update(detections)  # List[Track]

for track in tracks:
    # Draw bbox with ID
    cv2.rectangle(frame, ...)
    cv2.putText(frame, f"ID:{track.id}", ...)
```

---

## 6. Configuration Parameters

| Parameter | Default | Description | Tuning Guide |
|-----------|---------|-------------|--------------|
| max_age | 1 | Frames to keep track without detection | ↑ for occlusions, ↓ for fast scenes |
| min_hits | 3 | Hits before track confirmed | ↑ to reduce false positives |
| iou_threshold | 0.3 | Minimum IOU for match | ↑ for crowded scenes, ↓ for sparse |

---

## 7. Testing Strategy

### 7.1 Unit Tests
1. **Test Track creation:** Verify all fields initialized
2. **Test KalmanBoxTracker:**
   - Predict without update
   - Update with detection
   - State conversion (bbox ↔ [x,y,s,r])
3. **Test IOU computation:** Known bbox pairs
4. **Test Tracker.update():**
   - Empty detections
   - New detections (create tracks)
   - Matched detections (update tracks)
   - Stale tracks (deletion)

### 7.2 Integration Tests
- Feed real video frames through detector → tracker
- Verify IDs persist across frames
- Check track count matches object count

---

## 8. Performance Considerations

### 8.1 Computational Complexity
- **IOU matrix:** O(D × T) where D=detections, T=tracks
- **Hungarian:** O(min(D,T)³)
- **Total per frame:** ~1-5ms for typical scenes

### 8.2 Memory
- **Per track:** ~500 bytes (Kalman state + metadata)
- **100 tracks:** ~50KB (negligible)

---

## 9. Error Handling

1. **NaN in predictions:** Remove invalid trackers
2. **Empty detections:** All tracks predict, stale ones deleted
3. **No matches:** Create new tracks, delete old ones

---

## 10. Future Extensions

1. **Class-specific tracking:** Separate trackers per class
2. **Adaptive thresholds:** Adjust IOU based on scene density
3. **Deep SORT:** Add appearance features
4. **Track smoothing:** Post-process trajectories

---

**Last Updated:** 2025-01-XX  
**Status:** Ready for implementation
