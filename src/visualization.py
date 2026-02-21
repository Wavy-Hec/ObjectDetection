"""
Visualization utilities for drawing detections and tracks on frames.
Supports all COCO classes and open-vocabulary YOLO-World classes with
category-based color coding.
"""

import cv2
import numpy as np
from typing import List, Tuple
import random

# Import types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from detector import Detection
from tracker import Track


# Color palette for different track IDs (BGR format)
COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
    (255, 165, 0),   # Orange
    (0, 128, 128),   # Teal
    (128, 128, 0),   # Olive
    (203, 192, 255), # Pink
    (42, 42, 165),   # Brown
    (180, 105, 255), # Hot Pink
    (255, 191, 0),   # Deep Sky Blue
    (50, 205, 50),   # Lime green
    (230, 216, 173), # Light Blue
]

# Class-based colors (BGR) for common COCO categories + YOLO-World extras
CLASS_COLORS = {
    # --- People ---
    'person':        (0, 200, 0),      # Green
    # --- Vehicles ---
    'bicycle':       (255, 144, 30),    # Dodger Blue
    'car':           (0, 0, 255),       # Red
    'motorcycle':    (0, 69, 255),      # Orange Red
    'bus':           (0, 50, 200),      # Dark Red
    'truck':         (50, 50, 180),     # Dark Red 2
    'airplane':      (200, 200, 50),    # Teal
    'train':         (100, 50, 50),     # Dark Blue
    'boat':          (200, 150, 50),    # Slate Blue
    # --- Electronics ---
    'cell phone':    (255, 0, 255),     # Magenta
    'laptop':        (255, 255, 0),     # Cyan
    'keyboard':      (200, 200, 0),     # Cyan-ish
    'mouse':         (200, 150, 0),     # Teal-ish
    'remote':        (180, 0, 180),     # Purple
    'tv':            (128, 0, 128),     # Purple
    # --- Kitchen / Food ---
    'bottle':        (0, 255, 255),     # Yellow
    'cup':           (0, 200, 200),     # Dark Yellow
    'wine glass':    (180, 130, 200),   # Mauve
    'fork':          (150, 150, 150),   # Grey
    'knife':         (50, 50, 255),     # Bright Red
    'spoon':         (160, 160, 160),   # Light Grey
    'bowl':          (100, 180, 220),   # Sandy
    'banana':        (0, 230, 255),     # Bright Yellow
    'apple':         (50, 50, 220),     # Apple Red
    'sandwich':      (80, 170, 220),    # Tan
    'pizza':         (30, 100, 230),    # Pizza Orange
    'donut':         (120, 100, 210),   # Donut Brown
    'cake':          (180, 150, 220),   # Cake Pink
    # --- Furniture ---
    'chair':         (128, 128, 0),     # Olive
    'couch':         (100, 100, 0),     # Dark Olive
    'dining table':  (100, 80, 60),     # Slate
    'bed':           (140, 100, 120),   # Mauve
    'toilet':        (200, 200, 200),   # Light Grey
    # --- Accessories ---
    'backpack':      (50, 150, 200),    # Sandy
    'handbag':       (100, 50, 200),    # Brown-ish
    'umbrella':      (200, 50, 100),    # Blue-ish
    'tie':           (220, 80, 180),    # Violet
    'suitcase':      (80, 80, 160),     # Brown
    # --- Animals ---
    'dog':           (80, 200, 255),    # Gold
    'cat':           (255, 100, 80),    # Coral Blue
    'bird':          (180, 230, 100),   # Light Green
    'horse':         (60, 120, 180),    # Saddle Brown
    # --- Misc COCO ---
    'scissors':      (80, 80, 255),     # Red
    'book':          (255, 200, 100),   # Light Blue
    'clock':         (200, 180, 60),    # Steel Blue
    'vase':          (160, 100, 200),   # Orchid
    'toothbrush':    (255, 220, 180),   # Light Cyan
    'potted plant':  (50, 180, 50),     # Forest Green
    'teddy bear':    (100, 160, 230),   # Sandy Brown
    # --- YOLO-World custom / open-vocabulary ---
    'pen':           (255, 50, 50),     # Blue
    'pencil':        (240, 100, 50),    # Blue-ish
    'marker':        (200, 0, 200),     # Purple
    'screwdriver':   (0, 180, 255),     # Orange
    'hammer':        (50, 100, 180),    # Brown
    'wrench':        (100, 100, 200),   # Steel
    'pliers':        (80, 120, 200),    # Rust
    'drill':         (0, 150, 200),     # Dark Orange
    'saw':           (50, 50, 200),     # Dark Red
    'tape':          (200, 200, 50),    # Teal
    'ruler':         (150, 200, 100),   # Sage
    'stapler':       (100, 50, 150),    # Eggplant
    'glasses':       (200, 180, 150),   # Pale Blue
    'watch':         (180, 180, 0),     # Dark Cyan
    'hat':           (100, 150, 220),   # Tan
    'glove':         (200, 150, 100),   # Slate Blue
    'shoe':          (80, 60, 100),     # Dark Purple
    'tool':          (50, 130, 200),    # Bronze
    'wire':          (150, 150, 50),    # Dark Teal
    'cable':         (120, 120, 80),    # Olive Drab
    'headphones':    (200, 50, 200),    # Hot Pink
    'wallet':        (60, 80, 140),     # Sienna
    'key':           (170, 200, 50),    # Chartreuse
    'badge':         (50, 200, 200),    # Gold
    'lanyard':       (180, 60, 120),    # Rose
    'mug':           (100, 180, 180),   # Cadet Blue
    'notebook':      (220, 180, 120),   # Powder Blue
}


def get_color_for_track(track_id: int, class_label: str = None) -> Tuple[int, int, int]:
    """
    Get a consistent color for a track, preferring class-based colors.
    
    Args:
        track_id: The track ID
        class_label: Object class name (optional)
        
    Returns:
        BGR color tuple
    """
    if class_label and class_label in CLASS_COLORS:
        return CLASS_COLORS[class_label]
    return COLORS[track_id % len(COLORS)]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """
    Get a consistent color for a track ID (legacy compatibility).
    
    Args:
        track_id: The track ID
        
    Returns:
        BGR color tuple
    """
    return COLORS[track_id % len(COLORS)]


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    show_confidence: bool = True,
    thickness: int = 2
) -> None:
    """
    Draw detection bounding boxes on frame (in-place)
    
    Args:
        frame: The frame to draw on (modified in-place)
        detections: List of detections
        show_confidence: Whether to show confidence scores
        thickness: Box line thickness
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        
        # Draw bounding box (green for detections)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        
        # Prepare label
        label = det.class_label
        if show_confidence:
            label += f" {det.confidence:.2f}"
        
        # Draw label background
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    show_id: bool = True,
    show_label: bool = True,
    show_speed: bool = True,
    show_trajectory: bool = True,
    thickness: int = 2
) -> None:
    """
    Draw track bounding boxes with IDs on frame (in-place)
    
    Args:
        frame: The frame to draw on (modified in-place)
        tracks: List of tracks
        show_id: Whether to show track IDs
        show_label: Whether to show class labels
        show_speed: Whether to show speed estimation
        show_trajectory: Whether to draw trajectory path
        thickness: Box line thickness
    """
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        
        # Get color for this track (class-based if available)
        color = get_color_for_track(track.id, track.class_label)
        
        # Draw trajectory if available
        if show_trajectory and hasattr(track, 'history') and len(track.history) > 1:
            points = np.array(track.history, dtype=np.int32)
            # Draw trajectory line
            for i in range(len(points) - 1):
                # Fade older points
                alpha = (i + 1) / len(points)
                line_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, tuple(points[i]), tuple(points[i+1]), line_color, 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label_parts = []
        if show_id:
            label_parts.append(f"ID:{track.id}")
        if show_label:
            label_parts.append(track.class_label)
        if show_speed and hasattr(track, 'get_speed'):
            speed = track.get_speed()
            label_parts.append(f"{speed:.1f}px/s")
        
        label = " ".join(label_parts)
        
        # Draw label background
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


def draw_segmentation_masks(
    frame: np.ndarray,
    detections: List[Detection],
    alpha: float = 0.4
) -> None:
    """
    Draw segmentation masks on frame (in-place)
    
    Args:
        frame: The frame to draw on (modified in-place)
        detections: List of detections with masks
        alpha: Transparency of mask overlay
    """
    for idx, det in enumerate(detections):
        if det.mask is None:
            continue
        
        # Resize mask to frame size if needed
        mask = det.mask
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Create colored mask
        color = COLORS[idx % len(COLORS)]
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[mask > 0.5] = color
        
        # Blend with frame
        cv2.addWeighted(frame, 1.0, colored_mask, alpha, 0, frame)
        
        # Draw center point

def draw_track_trails(
    frame: np.ndarray,
    track_history: dict,
    max_trail_length: int = 30
) -> None:
    """
    Draw track trails showing trajectory
    
    Args:
        frame: The frame to draw on (modified in-place)
        track_history: Dict mapping track_id to list of (x, y) positions
        max_trail_length: Maximum number of points in trail
    """
    for track_id, positions in track_history.items():
        if len(positions) < 2:
            continue
        
        color = get_color_for_id(track_id)
        
        # Draw lines connecting positions
        for i in range(1, len(positions)):
            if positions[i - 1] is None or positions[i] is None:
                continue
            
            # Calculate alpha based on position in trail (fade older points)
            alpha = i / len(positions)
            thickness = max(1, int(3 * alpha))
            
            cv2.line(
                frame,
                tuple(map(int, positions[i - 1])),
                tuple(map(int, positions[i])),
                color,
                thickness
            )


def create_tracking_visualization(
    frame: np.ndarray,
    detections: List[Detection],
    tracks: List[Track],
    fps: float = 0.0,
    frame_count: int = 0
) -> np.ndarray:
    """
    Create a comprehensive visualization with detections, tracks, and info
    
    Args:
        frame: Input frame
        detections: List of detections
        tracks: List of tracks
        fps: Current FPS
        frame_count: Current frame number
        
    Returns:
        Visualized frame
    """
    vis_frame = frame.copy()
    
    # Draw tracks (primary)
    draw_tracks(vis_frame, tracks)
    
    # Add info panel at top
    info_text = [
        f"Frame: {frame_count}",
        f"FPS: {fps:.1f}",
        f"Detections: {len(detections)}",
        f"Active Tracks: {len(tracks)}"
    ]
    
    # Draw semi-transparent info panel
    panel_height = 100
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (0, 0), (vis_frame.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
    
    # Draw info text
    y_pos = 25
    for text in info_text:
        cv2.putText(
            vis_frame,
            text,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        y_pos += 25
    
    return vis_frame


if __name__ == '__main__':
    # Test visualization with dummy data
    print("Testing visualization module...")
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create test detections
    from detector import Detection
    test_detections = [
        Detection(
            bbox=np.array([100, 100, 200, 200]),
            class_label='person',
            confidence=0.95
        ),
        Detection(
            bbox=np.array([300, 150, 400, 250]),
            class_label='car',
            confidence=0.87
        )
    ]
    
    # Create test tracks
    from tracker import Track
    test_tracks = [
        Track(
            id=1,
            bbox=np.array([100, 100, 200, 200]),
            class_label='person',
            confidence=0.95,
            age=10,
            hits=10,
            time_since_update=0
        ),
        Track(
            id=2,
            bbox=np.array([300, 150, 400, 250]),
            class_label='car',
            confidence=0.87,
            age=5,
            hits=5,
            time_since_update=0
        )
    ]
    
    # Draw visualization
    vis_frame = create_tracking_visualization(
        frame, test_detections, test_tracks, fps=30.0, frame_count=100
    )
    
    print("Visualization test completed successfully!")
    print("Colors assigned:")
    for track in test_tracks:
        color = get_color_for_id(track.id)
        print(f"  Track {track.id}: BGR{color}")
