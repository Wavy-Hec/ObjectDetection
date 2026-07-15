"""
Multi-object tracker: SORT core with ByteTrack-style two-stage association.

Motion is modelled per track by a Kalman filter (constant-velocity, SORT
state); data association is IOU + Hungarian matching, run in two stages a la
ByteTrack: high-confidence detections match first and may spawn new tracks,
then low-confidence detections recover tracks that stage 1 left unmatched.
Tracks confirm after ``min_hits`` and stay confirmed (latch), so brief
occlusions or skipped detection frames do not flicker IDs. Supports
class-aware matching for all COCO object categories, plus a
``predict_only()`` coast mode for detect-every-N-frames pipelines.

Example:
    from flowcount.detector import ObjectDetector
    from flowcount.tracker import Tracker

    detector = ObjectDetector()
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)

    for frame in video:
        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for track in tracks:
            print(f"ID {track.id}: {track.class_label}")
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    """Represents a tracked object with persistent ID."""
    
    id: int                    # Unique track identifier
    bbox: List[float]          # Bounding box [x1, y1, x2, y2]
    class_label: str           # Object class (any COCO class)
    confidence: float          # Detection confidence [0, 1]
    age: int                   # Total frames since creation
    hits: int                  # Consecutive successful updates
    time_since_update: int     # Frames since last detection
    velocity: Tuple[float, float] = (0.0, 0.0)  # Velocity (vx, vy) in pixels/frame
    history: List[Tuple[float, float]] = None    # Track history [(x, y), ...]
    class_id: int = -1         # COCO class ID
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def __repr__(self):
        return f"Track(id={self.id}, {self.class_label}, conf={self.confidence:.2f}, speed={self.get_speed():.1f}px/frame)"
    
    def get_speed(self) -> float:
        """Calculate speed magnitude from velocity."""
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes.
    
    State vector: [x, y, s, r, vx, vy, vs]
    - x, y: center coordinates
    - s: scale (area)
    - r: aspect ratio (width/height)
    - vx, vy, vs: velocities
    """
    
    count = 0  # Global track ID counter
    
    def __init__(self, bbox: List[float], class_label: str = "unknown",
                 confidence: float = 0.0, class_id: int = -1):
        """
        Initialize Kalman filter with bounding box.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            class_label: Object class name
            confidence: Detection confidence
            class_id: COCO class ID
        """
        # Initialize 7D state Kalman filter (x, y, s, r, vx, vy, vs)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x' = x + vx
            [0, 1, 0, 0, 0, 1, 0],  # y' = y + vy
            [0, 0, 1, 0, 0, 0, 1],  # s' = s + vs
            [0, 0, 0, 1, 0, 0, 0],  # r' = r (constant)
            [0, 0, 0, 0, 1, 0, 0],  # vx' = vx
            [0, 0, 0, 0, 0, 1, 0],  # vy' = vy
            [0, 0, 0, 0, 0, 0, 1],  # vs' = vs
        ])
        
        # Measurement matrix (observe x, y, s, r)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        
        # Measurement noise (higher for scale/ratio)
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise (uncertainty in velocity)
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty in velocities initially
        self.kf.P *= 10.0
        
        # Covariance (low for velocity terms)
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state from bbox
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.age = 0
        self.confirmed = False  # latches True once the track reaches min_hits
        
        # Store class info persistently
        self.class_label = class_label
        self.confidence = confidence
        self.class_id = class_id
        
        # Persistent history across frames
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.history_points: List[Tuple[float, float]] = [(cx, cy)]
    
    def _bbox_to_z(self, bbox: List[float]) -> np.ndarray:
        """
        Convert bounding box [x1, y1, x2, y2] to measurement [x, y, s, r].
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Measurement vector [x, y, s, r] where:
            - x, y: center coordinates
            - s: scale (area)
            - r: aspect ratio (width/height)
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # scale (area)
        r = w / float(h) if h > 0 else 1.0  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _z_to_bbox(self, z: np.ndarray) -> List[float]:
        """
        Convert measurement [x, y, s, r] to bounding box [x1, y1, x2, y2].
        
        Args:
            z: Measurement vector [x, y, s, r]
            
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        x, y, s, r = float(z.flat[0]), float(z.flat[1]), float(z.flat[2]), float(z.flat[3])
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
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
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Prevent negative scale
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        return self._z_to_bbox(self.kf.x[:4])
    
    def update(self, bbox: List[float], class_label: str = None,
               confidence: float = None, class_id: int = None):
        """
        Update Kalman filter with new detection.
        
        Args:
            bbox: Detected bounding box [x1, y1, x2, y2]
            class_label: Updated class label
            confidence: Updated confidence
            class_id: Updated class ID
        """
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self._bbox_to_z(bbox))
        
        # Update class info if provided
        if class_label is not None:
            self.class_label = class_label
        if confidence is not None:
            self.confidence = confidence
        if class_id is not None:
            self.class_id = class_id
        
        # Append center to persistent history
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.history_points.append((cx, cy))
        # Keep last 60 points
        if len(self.history_points) > 60:
            self.history_points = self.history_points[-60:]
    
    def get_state(self) -> List[float]:
        """
        Get current bounding box estimate.
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return self._z_to_bbox(self.kf.x[:4])


class Tracker:
    """SORT tracker for multi-object tracking with class-aware association."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3,
                 class_aware: bool = True, track_high_thresh: float = 0.5,
                 track_low_thresh: float = 0.1, output_coast: int = 1):
        """
        Initialize the tracker (SORT + ByteTrack-style two-stage association).

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: Minimum IOU for detection-track matching
            class_aware: If True, only match detections to tracks of the same class
            track_high_thresh: Detections >= this start the first association stage
                and may spawn new tracks (ByteTrack "high" pool).
            track_low_thresh: Detections in [low, high) are used only to recover
                already-tracked objects in the second stage (ByteTrack "low" pool).
            output_coast: Confirmed tracks missed for up to this many consecutive
                detection frames are still emitted at their predicted position,
                so a single dropped detection doesn't blink the track off screen.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.class_aware = class_aware
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.output_coast = output_coast
        self.trackers = []  # List of KalmanBoxTracker objects
        self.frame_count = 0
    
    def update(self, detections: List) -> List[Track]:
        """Update with detections using two-stage (ByteTrack-style) association.

        Stage 1 matches high-confidence detections against all predicted tracks;
        stage 2 recovers any still-unmatched tracks using low-confidence
        detections (which would otherwise be discarded). New tracks are spawned
        only from unmatched high-confidence detections.
        """
        self.frame_count += 1
        self._predict_and_prune()

        detections = detections or []
        high = [d for d in detections if d.confidence >= self.track_high_thresh]
        low = [d for d in detections
               if self.track_low_thresh <= d.confidence < self.track_high_thresh]

        track_indices = list(range(len(self.trackers)))

        # Stage 1: high-confidence detections vs all predicted tracks.
        matches1, unmatched_high, unmatched_trks = self._match(high, track_indices)
        for di, ti in matches1:
            self._update_tracker(ti, high[di])

        # Stage 2: recover remaining tracks with low-confidence detections.
        matches2, _unmatched_low, unmatched_trks = self._match(low, unmatched_trks)
        for di, ti in matches2:
            self._update_tracker(ti, low[di])

        # New tracks come only from unmatched HIGH-confidence detections.
        for di in unmatched_high:
            det = high[di]
            self.trackers.append(KalmanBoxTracker(
                det.bbox, class_label=det.class_label,
                confidence=det.confidence, class_id=getattr(det, 'class_id', -1)))

        # Emit tracks updated this frame, plus confirmed tracks coasting
        # within the output_coast window (prevents single-miss flicker).
        return self._collect_outputs(max_coast=self.output_coast)

    def predict_only(self, max_coast: int = 1) -> List[Track]:
        """Advance all tracks by Kalman prediction with no detections.

        Used by detect-every-N-frames: confirmed tracks coast on their predicted
        positions (up to ``max_coast`` frames) between detection frames.
        """
        self.frame_count += 1
        self._predict_and_prune()
        return self._collect_outputs(max_coast=max_coast)

    def _predict_and_prune(self) -> None:
        """Predict every tracker and drop any whose state became NaN."""
        for trk in self.trackers:
            trk.predict()
        self.trackers = [t for t in self.trackers
                         if not np.any(np.isnan(t.get_state()))]

    def _update_tracker(self, ti: int, det) -> None:
        self.trackers[ti].update(
            det.bbox, class_label=det.class_label,
            confidence=det.confidence, class_id=getattr(det, 'class_id', -1))
    
    def _match(self, detections: List, track_indices: List[int]):
        """IOU + Hungarian matching of detections to a subset of tracks.

        Args:
            detections: Detection objects to match.
            track_indices: Global indices into ``self.trackers`` to match against.

        Returns:
            (matches, unmatched_detection_idxs, unmatched_global_track_idxs) where
            matches is a list of ``(detection_idx, global_track_idx)`` tuples.
        """
        if not detections or not track_indices:
            return [], list(range(len(detections))), list(track_indices)

        iou = np.zeros((len(detections), len(track_indices)))
        for d, det in enumerate(detections):
            for j, ti in enumerate(track_indices):
                trk = self.trackers[ti]
                if self.class_aware and det.class_label != trk.class_label:
                    continue  # leave IOU at 0 -> never matched across classes
                iou[d, j] = self._iou(det.bbox, trk.get_state())

        row_ind, col_ind = linear_sum_assignment(-iou)
        matches = []
        matched_d, matched_j = set(), set()
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] < self.iou_threshold:
                continue
            matches.append((int(r), track_indices[c]))
            matched_d.add(int(r))
            matched_j.add(c)

        unmatched_d = [d for d in range(len(detections)) if d not in matched_d]
        unmatched_t = [track_indices[j] for j in range(len(track_indices))
                       if j not in matched_j]
        return matches, unmatched_d, unmatched_t

    def _collect_outputs(self, max_coast: int) -> List[Track]:
        """Cull dead tracks; return confirmed tracks seen within ``max_coast`` frames."""
        ret = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue
            if not trk.confirmed and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                trk.confirmed = True  # latch
            if trk.confirmed and trk.time_since_update <= max_coast:
                ret.append(self._to_track(trk))
        return ret

    def _to_track(self, trk: "KalmanBoxTracker") -> Track:
        """Build the public Track dataclass from a Kalman tracker."""
        return Track(
            id=trk.id,
            bbox=trk.get_state(),
            class_label=trk.class_label,
            confidence=trk.confidence,
            age=trk.age,
            hits=trk.hits,
            time_since_update=trk.time_since_update,
            velocity=(float(trk.kf.x[4]), float(trk.kf.x[5])),
            history=list(trk.history_points),
            class_id=trk.class_id,
        )

    @staticmethod
    def _iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: Bounding box [x1, y1, x2, y2]
            bbox2: Bounding box [x1, y1, x2, y2]
            
        Returns:
            IOU value in range [0, 1]
        """
        # Intersection rectangle
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
