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
    class_label: str           # Object class ("person" or "car")
    confidence: float          # Detection confidence [0, 1]
    age: int                   # Total frames since creation
    hits: int                  # Consecutive successful updates
    time_since_update: int     # Frames since last detection
    velocity: Tuple[float, float] = (0.0, 0.0)  # Velocity (vx, vy) in pixels/frame
    history: List[Tuple[float, float]] = None   # Track history [(x, y), ...]
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def __repr__(self):
        return f"Track(id={self.id}, {self.class_label}, conf={self.confidence:.2f}, speed={self.get_speed():.1f}px/s)"
    
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
    
    def __init__(self, bbox: List[float]):
        """
        Initialize Kalman filter with bounding box.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
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
        self.hit_streak = 0
        self.age = 0
    
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
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self._z_to_bbox(self.kf.x[:4])
    
    def update(self, bbox: List[float]):
        """
        Update Kalman filter with new detection.
        
        Args:
            bbox: Detected bounding box [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
    
    def get_state(self) -> List[float]:
        """
        Get current bounding box estimate.
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return self._z_to_bbox(self.kf.x[:4])


class Tracker:
    """SORT tracker for multi-object tracking."""
    
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum consecutive hits before track is confirmed
            iou_threshold: Minimum IOU for detection-track matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # List of KalmanBoxTracker objects
        self.frame_count = 0
    
    def update(self, detections: List) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects from detector
            
        Returns:
            List of Track objects for current frame
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove trackers with NaN predictions
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, trks
        )
        
        # Update matched trackers with detections
        for m in matched:
            det_idx, trk_idx = int(m[0]), int(m[1])
            self.trackers[trk_idx].update(detections[det_idx].bbox)
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i].bbox)
            self.trackers.append(trk)
        
        # Prepare output tracks
        ret = []
        i = len(self.trackers)
        
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop()
                i -= 1
                continue
            
            # Only return tracks with enough hits and valid age
            if (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits):
                # Find corresponding detection for class label and confidence
                class_label = "unknown"
                confidence = 0.0
                
                # Check if this tracker was matched
                for m in matched:
                    if i - 1 == int(m[1]):  # Adjust index for reversed iteration
                        det = detections[int(m[0])]
                        class_label = det.class_label
                        confidence = det.confidence
                        break
                
                # Get velocity from Kalman filter state
                vx = float(trk.kf.x[4])  # x velocity
                vy = float(trk.kf.x[5])  # y velocity
                
                # Get center point for history
                bbox = d
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                track = Track(
                    id=trk.id,
                    bbox=d,
                    class_label=class_label,
                    confidence=confidence,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update,
                    velocity=(vx, vy)
                )
                
                # Add to history (keep last 30 points)
                track.history.append((center_x, center_y))
                if len(track.history) > 30:
                    track.history = track.history[-30:]
                
                ret.append(track)
            
            i -= 1
        
        return ret
    
    def _associate_detections_to_tracks(
        self, detections: List, trackers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign detections to tracked objects using IOU matching.
        
        Args:
            detections: List of Detection objects
            trackers: Array of predicted tracker bboxes
            
        Returns:
            Tuple of (matched, unmatched_detections, unmatched_trackers)
            - matched: Array of [det_idx, trk_idx] pairs
            - unmatched_detections: Array of detection indices
            - unmatched_trackers: Array of tracker indices
        """
        if len(trackers) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int)
            )
        
        # Compute IOU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t in range(len(trackers)):
                iou_matrix[d, t] = self._iou(det.bbox, trackers[t][:4])
        
        # Use Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            # Maximize IOU (minimize negative IOU)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.column_stack([row_ind, col_ind])
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter matches below IOU threshold
        unmatched_detections = []
        unmatched_trackers = []
        matches = []
        
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        # Find unmatched detections
        if len(matched_indices) > 0:
            matched_det_indices = matched_indices[:, 0]
        else:
            matched_det_indices = np.array([])
        
        for d in range(len(detections)):
            if d not in matched_det_indices:
                unmatched_detections.append(d)
        
        # Find unmatched trackers
        if len(matched_indices) > 0:
            matched_trk_indices = matched_indices[:, 1]
        else:
            matched_trk_indices = np.array([])
        
        for t in range(len(trackers)):
            if t not in matched_trk_indices:
                unmatched_trackers.append(t)
        
        # Combine matches
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
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


if __name__ == "__main__":
    """Simple test of the tracker module."""
    from detector import Detection
    
    print("Testing SORT Tracker...")
    
    # Create tracker
    tracker = Tracker(max_age=1, min_hits=3, iou_threshold=0.3)
    print(f"Tracker initialized: max_age={tracker.max_age}, min_hits={tracker.min_hits}")
    
    # Simulate detections
    # Frame 1: One person
    det1 = [Detection([100, 100, 200, 200], "person", 0.9)]
    tracks1 = tracker.update(det1)
    print(f"\nFrame 1: {len(det1)} detections → {len(tracks1)} tracks")
    
    # Frame 2: Same person moved slightly
    det2 = [Detection([110, 105, 210, 205], "person", 0.9)]
    tracks2 = tracker.update(det2)
    print(f"Frame 2: {len(det2)} detections → {len(tracks2)} tracks")
    
    # Frame 3: Same person (should be confirmed now)
    det3 = [Detection([120, 110, 220, 210], "person", 0.9)]
    tracks3 = tracker.update(det3)
    print(f"Frame 3: {len(det3)} detections → {len(tracks3)} tracks")
    for track in tracks3:
        print(f"  {track}")
    
    # Frame 4: Person + new car
    det4 = [
        Detection([130, 115, 230, 215], "person", 0.9),
        Detection([300, 150, 450, 250], "car", 0.85)
    ]
    tracks4 = tracker.update(det4)
    print(f"\nFrame 4: {len(det4)} detections → {len(tracks4)} tracks")
    for track in tracks4:
        print(f"  {track}")
    
    print("\nTracker module test completed successfully!")
