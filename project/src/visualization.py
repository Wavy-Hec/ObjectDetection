"""
Visualization utilities for drawing detections and tracks on frames
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
]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """
    Get a consistent color for a track ID
    
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
        
        # Get color for this track ID
        color = get_color_for_id(track.id)
        
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
