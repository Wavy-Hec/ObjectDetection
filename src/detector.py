"""
Object detection module using YOLOv8 and YOLO-World models.

This module loads a pre-trained YOLO model from Ultralytics and provides
a detection function. Two modes are supported:

  1. **Standard YOLOv8** – Detects the 80 COCO classes (person, car, tie,
     scissors, etc.) with high speed and accuracy.
  2. **YOLO-World (open-vocabulary)** – Detects *any* object you name,
     including classes outside COCO like "pen", "screwdriver", "wrench",
     "pliers", etc.  Automatically activated when custom classes that are
     not in COCO are requested.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from ultralytics import YOLO
import torch


# All 80 COCO class names (used to decide if YOLO-World is needed)
COCO_CLASSES: Set[str] = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
}


class Detection:
    """Represents a single object detection."""
    
    def __init__(self, bbox: List[float], class_label: str, confidence: float,
                 class_id: int = -1, mask: Optional[np.ndarray] = None):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_label: Class name (e.g. "person", "cell phone", "pen")
            confidence: Confidence score [0, 1]
            class_id: Class ID
            mask: Optional segmentation mask (H×W binary array)
        """
        self.bbox = bbox
        self.class_label = class_label
        self.confidence = confidence
        self.class_id = class_id
        self.mask = mask
    
    def __repr__(self):
        mask_str = f", mask={self.mask.shape}" if self.mask is not None else ""
        return f"Detection({self.class_label}, conf={self.confidence:.2f}, bbox={self.bbox}{mask_str})"


def needs_yolo_world(target_classes: Optional[Set[str]]) -> bool:
    """Return True if any requested class is outside the COCO-80 set."""
    if target_classes is None:
        return False
    return not target_classes.issubset(COCO_CLASSES)


# Descriptive prompts that help YOLO-World detect objects more accurately.
# Single words like "pen" are ambiguous; adding context drastically reduces
# false positives.
WORLD_CLASS_PROMPTS = {
    # Tools
    'screwdriver':  'screwdriver tool with handle',
    'hammer':       'hammer tool with handle',
    'wrench':       'metal wrench tool',
    'pliers':       'pliers hand tool',
    'drill':        'power drill tool',
    'saw':          'hand saw tool',
    'tape':         'roll of tape',
    'ruler':        'ruler measuring tool',
    # Stationery
    'pen':          'pen for writing',
    'pencil':       'pencil for writing',
    'marker':       'marker pen',
    'stapler':      'stapler on desk',
    'notebook':     'paper notebook',
    'eraser':       'eraser for pencil',
    # Accessories
    'glasses':      'eyeglasses on face',
    'watch':        'wristwatch on wrist',
    'hat':          'hat on head',
    'glove':        'glove on hand',
    'shoe':         'shoe on foot',
    'wallet':       'wallet',
    'key':          'metal key',
    'badge':        'name badge or ID badge',
    'lanyard':      'lanyard around neck',
    'headphones':   'headphones or earbuds',
    'cable':        'cable or wire',
    'mug':          'coffee mug',
}


class ObjectDetector:
    """YOLOv8 / YOLO-World object detector for real-time multi-object tracking.

    When *target_classes* contains names that are **not** in the COCO-80 set
    (e.g. "pen", "screwdriver"), the detector automatically switches to a
    YOLO-World open-vocabulary model so those objects can be recognised.
    """

    # Default YOLO-World model used for open-vocabulary detection
    DEFAULT_WORLD_MODEL = 'yolov8x-worldv2.pt'

    # YOLO-World needs a higher confidence threshold to avoid false positives
    DEFAULT_WORLD_CONFIDENCE = 0.35

    def __init__(self, model_name: str = 'yolov8l.pt', conf_threshold: float = 0.20,
                 img_size: int = 640, use_segmentation: bool = False,
                 target_classes: Optional[set] = None, use_half: bool = True):
        """
        Initialize the object detector.

        Args:
            model_name: YOLO model variant.
                        Standard: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
                                  'yolov8l.pt', 'yolov8x.pt'
                        World:    'yolov8s-worldv2.pt', 'yolov8m-worldv2.pt',
                                  'yolov8l-worldv2.pt'
            conf_threshold: Confidence threshold for filtering detections
            img_size: Input image size (single int, YOLO resizes internally)
            use_segmentation: If True, use YOLOv8-seg for instance segmentation
                              (not available with YOLO-World)
            target_classes: Set of class names to detect. None = detect ALL
                            COCO classes.  If any name is not in COCO the
                            detector will auto-switch to YOLO-World.
            use_half: Use FP16 half precision on GPU for faster inference
        """
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.use_segmentation = use_segmentation
        self.target_classes = target_classes  # None means detect everything
        self.is_world_model = False
        self._world_label_map = {}  # maps descriptive prompt -> user label

        # ── Auto-switch to YOLO-World when custom classes are requested ──
        if needs_yolo_world(target_classes):
            non_coco = target_classes - COCO_CLASSES
            print(f"  ⚡ Custom classes detected outside COCO: {non_coco}")
            print(f"  ⚡ Auto-switching to YOLO-World for open-vocabulary detection")
            if 'world' not in model_name.lower():
                model_name = self.DEFAULT_WORLD_MODEL
            self.is_world_model = True
            self.use_segmentation = False
            # Raise confidence for World model to reduce false positives
            if conf_threshold < self.DEFAULT_WORLD_CONFIDENCE:
                self.conf_threshold = self.DEFAULT_WORLD_CONFIDENCE
                print(f"  ⚡ Confidence raised to {self.conf_threshold} for YOLO-World (reduces noise)")
        elif 'world' in model_name.lower():
            self.is_world_model = True
            self.use_segmentation = False
            if conf_threshold < self.DEFAULT_WORLD_CONFIDENCE:
                self.conf_threshold = self.DEFAULT_WORLD_CONFIDENCE

        # ── Handle segmentation model suffix ──
        if self.use_segmentation and not model_name.endswith('-seg.pt'):
            model_name = model_name.replace('.pt', '-seg.pt')

        # ── Load model ──
        self.model = YOLO(model_name)

        # ── Set custom vocabulary for YOLO-World ──
        if self.is_world_model and target_classes:
            # Use descriptive prompts for better accuracy
            class_list = []
            for cls in sorted(target_classes):
                prompt = WORLD_CLASS_PROMPTS.get(cls, cls)
                class_list.append(prompt)
                if prompt != cls:
                    self._world_label_map[prompt] = cls
            print(f"  Setting YOLO-World vocabulary: {class_list}")
            self.model.set_classes(class_list)

        # Determine device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # Half precision for GPU speed boost
        self.use_half = use_half and self.device == 'cuda'

        # Warmup the model
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, imgsz=self.img_size, conf=self.conf_threshold, verbose=False,
                   half=self.use_half)

        # Get available class names from model
        self.class_names = self.model.names  # dict {id: name}

        print(f"ObjectDetector initialized on {self.device} (half={self.use_half})")
        print(f"Model: {model_name}, Confidence threshold: {conf_threshold}")
        if self.is_world_model:
            print(f"Mode: YOLO-World open-vocabulary ({len(self.class_names)} classes)")
        else:
            print(f"Mode: Standard YOLOv8 ({len(self.class_names)} COCO classes)")
        print(f"Segmentation: {'enabled' if self.use_segmentation else 'disabled'}")
        if self.target_classes:
            print(f"Target classes: {self.target_classes}")
        else:
            print(f"Detecting ALL {len(self.class_names)} classes")
    
    def _map_world_label(self, label: str) -> str:
        """Map a YOLO-World descriptive prompt back to the user's short label."""
        return self._world_label_map.get(label, label)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a single frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            List of Detection objects for all detected classes
        """
        # Run inference with half precision if available
        results = self.model(
            frame, imgsz=self.img_size, conf=self.conf_threshold,
            verbose=False, half=self.use_half
        )
        
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

                # Map descriptive YOLO-World prompts back to short labels
                if self.is_world_model:
                    class_name = self._map_world_label(class_name)
                
                # Filter by target classes if specified, otherwise keep all
                if self.target_classes is not None and class_name not in self.target_classes:
                    continue

                # Skip tiny boxes that are likely noise (< 20px in any dimension)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue
                
                # Extract mask if available
                mask = None
                if masks is not None and self.use_segmentation:
                    mask = masks.data[idx].cpu().numpy()
                
                detection = Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    class_label=class_name,
                    confidence=conf,
                    class_id=class_id,
                    mask=mask
                )
                detections.append(detection)

        # For YOLO-World: apply additional NMS to remove overlapping boxes
        if self.is_world_model and len(detections) > 1:
            detections = self._nms_filter(detections, iou_thresh=0.5)

        return detections

    @staticmethod
    def _nms_filter(detections: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
        """Apply class-agnostic NMS to remove overlapping detections."""
        if not detections:
            return detections

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]


def main():
    """Simple test of the detector on a sample image."""
    import cv2
    import urllib.request

    print("\n--- Standard YOLOv8l detector test ---")
    detector = ObjectDetector()  # defaults to yolov8l.pt

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

    detections = detector.detect(test_frame)
    print(f"\nStandard detections: {len(detections)}")
    for det in detections:
        print(f"  {det}")

    # Test YOLO-World with custom classes
    print("\n--- YOLO-World open-vocabulary test ---")
    world_detector = ObjectDetector(
        target_classes={'person', 'bus', 'pen', 'tie', 'screwdriver'}
    )
    world_detections = world_detector.detect(test_frame)
    print(f"\nYOLO-World detections: {len(world_detections)}")
    for det in world_detections:
        print(f"  {det}")

    print("\nDetector module test completed successfully!")


if __name__ == "__main__":
    main()
