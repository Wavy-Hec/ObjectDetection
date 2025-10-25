"""
Object detection module using YOLOv8 model.

This module loads a pre-trained YOLOv8s model from Ultralytics and provides
a detection function that filters for person and car classes.
"""

import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import torch


class Detection:
    """Represents a single object detection."""
    
    def __init__(self, bbox: List[float], class_label: str, confidence: float, mask: Optional[np.ndarray] = None):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_label: Class name ("person" or "car")
            confidence: Confidence score [0, 1]
            mask: Optional segmentation mask (H×W binary array)
        """
        self.bbox = bbox
        self.class_label = class_label
        self.confidence = confidence
        self.mask = mask
    
    def __repr__(self):
        mask_str = f", mask={self.mask.shape}" if self.mask is not None else ""
        return f"Detection({self.class_label}, conf={self.confidence:.2f}, bbox={self.bbox}{mask_str})"


class ObjectDetector:
    """YOLOv8-based object detector for real-time tracking."""
    
    # Target classes from COCO dataset
    TARGET_CLASSES = {'person', 'car'}
    
    def __init__(self, model_name: str = 'yolov8s.pt', conf_threshold: float = 0.25, 
                 img_size: Tuple[int, int] = (640, 480), use_segmentation: bool = False):
        """
        Initialize the object detector.
        
        Args:
            model_name: YOLOv8 model variant (default: yolov8s.pt)
            conf_threshold: Confidence threshold for filtering detections
            img_size: Target image size (width, height)
            use_segmentation: If True, use YOLOv8-seg for instance segmentation
        """
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.use_segmentation = use_segmentation
        
        # Load YOLOv8 model (segmentation or detection)
        if use_segmentation and not model_name.endswith('-seg.pt'):
            # Use segmentation model
            model_name = model_name.replace('.pt', '-seg.pt')
        
        self.model = YOLO(model_name)
        
        # Determine device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        print(f"ObjectDetector initialized on {self.device}")
        print(f"Model: {model_name}, Confidence threshold: {conf_threshold}")
        print(f"Segmentation: {'enabled' if use_segmentation else 'disabled'}")
        print(f"Target classes: {self.TARGET_CLASSES}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a single frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            List of Detection objects for person and car classes
        """
        # Run inference
        results = self.model(frame, imgsz=self.img_size, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            
            for idx, box in enumerate(boxes):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Extract class and confidence
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Filter for target classes only
                if class_name in self.TARGET_CLASSES:
                    # Extract mask if available
                    mask = None
                    if masks is not None and self.use_segmentation:
                        mask = masks.data[idx].cpu().numpy()
                    
                    detection = Detection(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        class_label=class_name,
                        confidence=conf,
                        mask=mask
                    )
                    detections.append(detection)
        
        return detections


def main():
    """Simple test of the detector on a sample image."""
    import cv2
    import urllib.request
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Try to download a test image with people and cars
    try:
        print("\nDownloading test image...")
        url = "https://ultralytics.com/images/bus.jpg"
        urllib.request.urlretrieve(url, "test_image.jpg")
        test_frame = cv2.imread("test_image.jpg")
        print(f"Test image loaded: {test_frame.shape}")
    except Exception as e:
        print(f"Could not download test image: {e}")
        print("Using blank test frame instead...")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect(test_frame)
    
    print(f"\nTest detection completed.")
    print(f"Number of detections: {len(detections)}")
    
    for det in detections:
        print(f"  {det}")
    
    print("\nDetector module test completed successfully!")


if __name__ == "__main__":
    main()
