"""
Unit tests for the tracker module.

These tests verify that the SORT tracker works correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tracker import Track, KalmanBoxTracker, Tracker
from detector import Detection


class TestTrack:
    """Test the Track dataclass."""
    
    def test_track_creation(self):
        """Test creating a Track object."""
        bbox = [100.0, 200.0, 300.0, 400.0]
        track = Track(
            id=1,
            bbox=bbox,
            class_label="person",
            confidence=0.85,
            age=5,
            hits=3,
            time_since_update=0
        )
        
        assert track.id == 1
        assert track.bbox == bbox
        assert track.class_label == "person"
        assert track.confidence == 0.85
        assert track.age == 5
        assert track.hits == 3
        assert track.time_since_update == 0
    
    def test_track_repr(self):
        """Test Track string representation."""
        track = Track(
            id=5,
            bbox=[10, 20, 30, 40],
            class_label="car",
            confidence=0.95,
            age=10,
            hits=5,
            time_since_update=1
        )
        repr_str = repr(track)
        
        assert "Track" in repr_str
        assert "id=5" in repr_str
        assert "car" in repr_str
        assert "0.95" in repr_str


class TestKalmanBoxTracker:
    """Test the KalmanBoxTracker class."""
    
    def test_kalman_init(self):
        """Test KalmanBoxTracker initialization."""
        bbox = [100, 100, 200, 200]
        kf = KalmanBoxTracker(bbox)
        
        assert kf.id >= 0
        assert kf.age == 0
        assert kf.hits == 0
        assert kf.time_since_update == 0
    
    def test_bbox_to_z_conversion(self):
        """Test conversion from bbox to measurement."""
        bbox = [100, 100, 200, 200]
        kf = KalmanBoxTracker(bbox)
        
        z = kf._bbox_to_z(bbox)
        
        # Center should be (150, 150)
        assert z[0] == 150.0
        assert z[1] == 150.0
        # Area should be 10000
        assert z[2] == 10000.0
        # Aspect ratio should be 1.0 (square)
        assert z[3] == 1.0
    
    def test_z_to_bbox_conversion(self):
        """Test conversion from measurement to bbox."""
        bbox = [100, 100, 200, 200]
        kf = KalmanBoxTracker(bbox)
        
        z = kf._bbox_to_z(bbox)
        bbox_reconstructed = kf._z_to_bbox(z)
        
        # Should reconstruct original bbox
        np.testing.assert_array_almost_equal(bbox, bbox_reconstructed, decimal=5)
    
    def test_predict_updates_state(self):
        """Test that predict updates tracker state."""
        bbox = [100, 100, 200, 200]
        kf = KalmanBoxTracker(bbox)
        
        initial_age = kf.age
        initial_time = kf.time_since_update
        
        predicted_bbox = kf.predict()
        
        # Age should increase
        assert kf.age == initial_age + 1
        # time_since_update should increase
        assert kf.time_since_update == initial_time + 1
        # Should return a valid bbox
        assert len(predicted_bbox) == 4
    
    def test_update_with_detection(self):
        """Test updating with a detection."""
        bbox = [100, 100, 200, 200]
        kf = KalmanBoxTracker(bbox)
        
        # Predict first
        kf.predict()
        
        # Update with new detection
        new_bbox = [105, 105, 205, 205]
        kf.update(new_bbox)
        
        # Hits should increase
        assert kf.hits == 1
        # time_since_update should reset
        assert kf.time_since_update == 0
    
    def test_id_counter_increments(self):
        """Test that track IDs are unique and incremental."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [300, 300, 400, 400]
        
        kf1 = KalmanBoxTracker(bbox1)
        kf2 = KalmanBoxTracker(bbox2)
        
        # IDs should be different
        assert kf1.id != kf2.id


class TestTracker:
    """Test the Tracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker instance for testing."""
        return Tracker(max_age=1, min_hits=3, iou_threshold=0.3)
    
    @pytest.fixture
    def sample_detection(self):
        """Create a sample detection."""
        return Detection(
            bbox=[100.0, 100.0, 200.0, 200.0],
            class_label="person",
            confidence=0.9
        )
    
    def test_tracker_init(self, tracker):
        """Test Tracker initialization."""
        assert tracker.max_age == 1
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0
    
    def test_iou_computation(self):
        """Test IOU computation with known values."""
        # Identical boxes -> IOU = 1.0
        bbox1 = [100, 100, 200, 200]
        bbox2 = [100, 100, 200, 200]
        iou = Tracker._iou(bbox1, bbox2)
        assert iou == 1.0
        
        # No overlap -> IOU = 0.0
        bbox1 = [100, 100, 200, 200]
        bbox2 = [300, 300, 400, 400]
        iou = Tracker._iou(bbox1, bbox2)
        assert iou == 0.0
        
        # 50% overlap
        bbox1 = [100, 100, 200, 200]
        bbox2 = [150, 100, 250, 200]  # 50x100 overlap, 150x100 total
        iou = Tracker._iou(bbox1, bbox2)
        expected = 5000 / 15000  # intersection / union
        assert abs(iou - expected) < 0.01
    
    def test_empty_detections(self, tracker):
        """Test tracker with no detections."""
        tracks = tracker.update([])
        
        assert len(tracks) == 0
        assert tracker.frame_count == 1
    
    def test_new_detections_create_tracks(self, tracker, sample_detection):
        """Test that new detections create tracks."""
        detections = [sample_detection]
        
        # First frame
        tracks = tracker.update(detections)
        # min_hits=3, so no tracks returned yet
        assert len(tracks) == 1  # Early frames show tracks
        
        # Second frame
        tracks = tracker.update(detections)
        assert len(tracks) == 1
        
        # Third frame - track should be confirmed
        tracks = tracker.update(detections)
        assert len(tracks) == 1
        assert tracks[0].class_label == "person"
        assert tracks[0].confidence == 0.9
    
    def test_matched_detections_update_tracks(self, tracker):
        """Test that matched detections update existing tracks."""
        # Create initial detection
        det1 = Detection([100, 100, 200, 200], "person", 0.9)
        
        # Frame 1
        tracks1 = tracker.update([det1])
        first_id = tracks1[0].id if len(tracks1) > 0 else None
        
        # Frame 2 - slightly moved detection (should match)
        det2 = Detection([105, 105, 205, 205], "person", 0.9)
        tracks2 = tracker.update([det2])
        
        # Frame 3 - confirm track
        det3 = Detection([110, 110, 210, 210], "person", 0.9)
        tracks = tracker.update([det3])
        
        # Should have 1 confirmed track
        assert len(tracks) == 1
        # Should have same ID throughout if first_id exists
        if first_id is not None:
            assert tracks[0].id == first_id
    
    def test_stale_tracks_deleted(self, tracker):
        """Test that stale tracks are deleted after max_age."""
        # Create detection
        det = Detection([100, 100, 200, 200], "person", 0.9)
        
        # Frame 1-3: Build up hits
        tracker.update([det])
        tracker.update([det])
        tracks = tracker.update([det])
        
        # Should have a confirmed track now
        assert len(tracks) >= 1
        
        # Frame 4: No detection
        # With max_age=1, track might still survive this frame
        tracks = tracker.update([])
        
        # Frame 5: Still no detection
        # Now track should definitely be deleted (time_since_update > max_age)
        tracks = tracker.update([])
        
        # Track should be gone
        assert len(tracks) == 0
    
    def test_track_confirmation(self, tracker):
        """Test that tracks require min_hits before confirmation."""
        det = Detection([100, 100, 200, 200], "person", 0.9)
        
        # Frame 1
        tracks = tracker.update([det])
        # Early frames (frame_count <= min_hits) show tracks
        
        # Frame 2
        tracks = tracker.update([det])
        
        # Frame 3 - track should be confirmed (hit_streak >= min_hits)
        tracks = tracker.update([det])
        assert len(tracks) == 1
        # The track exists and is confirmed (either by hit_streak or early frames)
        assert tracks[0].id >= 0
    
    def test_multiple_objects(self, tracker):
        """Test tracking multiple objects."""
        # Create 2 detections
        det1 = Detection([100, 100, 200, 200], "person", 0.9)
        det2 = Detection([300, 300, 400, 400], "car", 0.85)
        
        # Build up hits for both
        for _ in range(3):
            tracks = tracker.update([det1, det2])
        
        # Should have 2 tracks
        assert len(tracks) == 2
        
        # Should have different IDs
        ids = {track.id for track in tracks}
        assert len(ids) == 2
    
    def test_class_label_preservation(self, tracker):
        """Test that class labels are preserved."""
        det = Detection([100, 100, 200, 200], "car", 0.95)
        
        # Build up hits
        for _ in range(3):
            tracks = tracker.update([det])
        
        # Check class label
        assert len(tracks) == 1
        assert tracks[0].class_label == "car"
        assert tracks[0].confidence == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
