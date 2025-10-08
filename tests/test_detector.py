"""
Unit tests for the detector module.

These tests verify that the YOLOv5s detector works correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detector import Detection, ObjectDetector


class TestDetection:
    """Test the Detection class."""
    
    def test_detection_creation(self):
        """Test creating a Detection object."""
        bbox = [100.0, 200.0, 300.0, 400.0]
        label = "person"
        confidence = 0.85
        
        det = Detection(bbox, label, confidence)
        
        assert det.bbox == bbox
        assert det.class_label == label
        assert det.confidence == confidence
    
    def test_detection_repr(self):
        """Test Detection string representation."""
        det = Detection([10, 20, 30, 40], "car", 0.95)
        repr_str = repr(det)
        
        assert "car" in repr_str
        assert "0.95" in repr_str
        assert "Detection" in repr_str


class TestObjectDetector:
    """Test the ObjectDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        # Use small confidence threshold for testing
        return ObjectDetector(conf_threshold=0.25)
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.conf_threshold == 0.25
        assert detector.img_size == (640, 480)
        assert detector.device in ['cuda', 'cpu']
        assert 'person' in detector.TARGET_CLASSES
        assert 'car' in detector.TARGET_CLASSES
    
    def test_detect_with_blank_frame(self, detector):
        """Test detection on a blank frame (should return empty list)."""
        # Create a blank 640x480 frame
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(blank_frame)
        
        # Blank frame should have no detections
        assert isinstance(detections, list)
        assert len(detections) == 0
    
    def test_detect_returns_list(self, detector):
        """Test that detect always returns a list."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        
        assert isinstance(result, list)
    
    def test_detect_with_random_frame(self, detector):
        """Test detection on random noise (should return few/no detections)."""
        # Random noise frame
        random_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(random_frame)
        
        # Should return a list (might be empty or have false positives)
        assert isinstance(detections, list)
        
        # All detections should be valid
        for det in detections:
            assert isinstance(det, Detection)
            assert det.class_label in ['person', 'car']
            assert 0 <= det.confidence <= 1
            assert len(det.bbox) == 4
    
    def test_target_classes(self, detector):
        """Test that target classes are correctly set."""
        assert detector.TARGET_CLASSES == {'person', 'car'}


class TestDetectorConfiguration:
    """Test different detector configurations."""
    
    def test_custom_confidence_threshold(self):
        """Test detector with custom confidence threshold."""
        detector = ObjectDetector(conf_threshold=0.5)
        assert detector.conf_threshold == 0.5
    
    def test_custom_image_size(self):
        """Test detector with custom image size."""
        detector = ObjectDetector(img_size=(416, 416))
        assert detector.img_size == (416, 416)


# Integration test (can be slow, marked as such)
@pytest.mark.slow
class TestDetectorIntegration:
    """Integration tests for the detector."""
    
    def test_full_detection_pipeline(self):
        """Test the full detection pipeline."""
        detector = ObjectDetector()
        
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection
        detections = detector.detect(frame)
        
        # Verify output format
        assert isinstance(detections, list)
        for det in detections:
            assert hasattr(det, 'bbox')
            assert hasattr(det, 'class_label')
            assert hasattr(det, 'confidence')


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
