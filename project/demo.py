#!/usr/bin/env python3
"""
Demo script for object detection and tracking
Combines detector and tracker to process a video file or webcam
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from typing import List, Tuple

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detector import ObjectDetector, Detection
from tracker import Tracker, Track
from visualization import draw_detections, draw_tracks


def process_video(
    input_source: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.3,
    max_age: int = 1,
    min_hits: int = 3,
    display: bool = True
) -> None:
    """
    Process video with detection and tracking
    
    Args:
        input_source: Path to video file or camera index (0 for webcam)
        output_path: Optional path to save output video
        conf_threshold: Detection confidence threshold
        iou_threshold: IOU threshold for tracking
        max_age: Maximum frames to keep track without detection
        min_hits: Minimum hits before track confirmation
        display: Whether to display the video
    """
    # Initialize detector and tracker
    print("Initializing detector and tracker...")
    detector = ObjectDetector(
        model_name='yolov5su.pt',
        conf_threshold=conf_threshold
    )
    tracker = Tracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold
    )
    
    # Open video source
    if input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
        print(f"Opened webcam {input_source}")
    else:
        cap = cv2.VideoCapture(input_source)
        print(f"Opened video: {input_source}")
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {input_source}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Processing loop
    frame_count = 0
    start_time = time.time()
    
    print("\nProcessing video... (Press 'q' to quit)")
    print("-" * 50)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect objects
            detections = detector.detect(frame)
            
            # Track objects
            tracks = tracker.update(detections)
            
            # Draw results
            vis_frame = frame.copy()
            draw_tracks(vis_frame, tracks)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS text
            cv2.putText(
                vis_frame,
                f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display
            if display:
                cv2.imshow('Object Tracking Demo', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nUser interrupted")
                    break
            
            # Write to output
            if writer:
                writer.write(vis_frame)
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames | "
                      f"FPS: {current_fps:.1f} | "
                      f"Tracks: {len(tracks)}")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print("-" * 50)
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if output_path:
            print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Object Detection and Tracking Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='0',
        help='Input video file or camera index (0 for webcam)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output video file path (optional)'
    )
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.25,
        help='Detection confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.3,
        help='IOU threshold for tracking'
    )
    parser.add_argument(
        '--max-age',
        type=int,
        default=1,
        help='Maximum frames to keep track without detection'
    )
    parser.add_argument(
        '--min-hits',
        type=int,
        default=3,
        help='Minimum hits before track confirmation'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (useful for headless processing)'
    )
    
    args = parser.parse_args()
    
    try:
        process_video(
            input_source=args.input,
            output_path=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_age=args.max_age,
            min_hits=args.min_hits,
            display=not args.no_display
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
