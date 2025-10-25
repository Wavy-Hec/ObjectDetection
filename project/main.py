#!/usr/bin/env python3
"""
Real-time object tracking with webcam or video files.

Usage:
    # Webcam
    python main.py --input 0
    
    # Video file
    python main.py --input video.mp4
    
    # With recording
    python main.py --input video.mp4 --output tracked.mp4

Controls:
    q / ESC - Quit
    p       - Pause/Resume
    s       - Save current frame
"""

import cv2
import numpy as np
import argparse
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from video_source import create_video_source, VideoSourceError
from detector import ObjectDetector
from tracker import Tracker
from visualization import draw_segmentation_masks, draw_tracks


class FPSTracker:
    """Track FPS with smoothing."""
    
    def __init__(self, buffer_size: int = 30):
        """
        Initialize FPS tracker.
        
        Args:
            buffer_size: Number of samples to average
        """
        self.times = []
        self.buffer_size = buffer_size
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        Update FPS calculation.
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        
        self.times.append(elapsed)
        if len(self.times) > self.buffer_size:
            self.times.pop(0)
        
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


def add_overlays(frame: np.ndarray, fps: float, num_detections: int, num_tracks: int, is_paused: bool = False):
    """
    Add FPS and statistics overlays to frame.
    
    Args:
        frame: Frame to overlay on
        fps: Current FPS
        num_detections: Number of detections
        num_tracks: Number of tracks
        is_paused: Whether system is paused
    """
    # FPS counter (top-left, green)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Detection count
    det_text = f"Detections: {num_detections}"
    cv2.putText(frame, det_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Track count
    track_text = f"Tracks: {num_tracks}"
    cv2.putText(frame, track_text, (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Paused indicator
    if is_paused:
        pause_text = "PAUSED"
        cv2.putText(frame, pause_text, (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


def save_frame(frame: np.ndarray, prefix: str = "frame") -> str:
    """
    Save frame to file.
    
    Args:
        frame: Frame to save
        prefix: Filename prefix
        
    Returns:
        Saved filename
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename


def print_controls():
    """Print keyboard controls."""
    print("\n" + "="*60)
    print("  KEYBOARD CONTROLS")
    print("="*60)
    print("  q / ESC  - Quit")
    print("  p        - Pause/Resume")
    print("  s        - Save current frame")
    print("="*60 + "\n")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time object tracking with webcam or video files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                        help='Input source (0 for webcam, or video file path)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output video file path (optional)')
    
    # Detection
    parser.add_argument('--confidence', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--no-segmentation', action='store_true',
                        help='Disable segmentation masks')
    
    # Tracking
    parser.add_argument('--max-age', type=int, default=1,
                        help='Maximum frames to keep track alive')
    parser.add_argument('--min-hits', type=int, default=3,
                        help='Minimum hits to confirm track')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='IOU threshold for tracking')
    
    # Features
    parser.add_argument('--no-trajectories', action='store_true',
                        help='Disable trajectory visualization')
    parser.add_argument('--no-speed', action='store_true',
                        help='Disable speed overlay')
    
    # Display
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window (headless mode)')
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    
    # Parse arguments
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("  REAL-TIME OBJECT TRACKING")
    print("="*60)
    
    try:
        # 1. Create video source
        print("\n[1/4] Initializing video source...")
        video_source = create_video_source(args.input)
        props = video_source.get_properties()
        
        # 2. Initialize detector
        print("\n[2/4] Initializing detector...")
        detector = ObjectDetector(
            model_name='yolov8s.pt',
            conf_threshold=args.confidence,
            use_segmentation=not args.no_segmentation
        )
        
        # 3. Initialize tracker
        print("\n[3/4] Initializing tracker...")
        tracker = Tracker(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold
        )
        
        # 4. Setup display and recording
        print("\n[4/4] Setting up display and recording...")
        
        window_name = "Object Tracking"
        if not args.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        recorder = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            recorder = cv2.VideoWriter(
                args.output, fourcc, props['fps'],
                (props['width'], props['height'])
            )
            if not recorder.isOpened():
                print(f"⚠️  Warning: Could not open video writer for {args.output}")
                recorder = None
            else:
                print(f"✓ Recording to: {args.output}")
        
        # Print controls
        print_controls()
        
        # Print startup info
        print("="*60)
        print("  SYSTEM READY")
        print("="*60)
        print(f"  Source: {props['source_type']}")
        print(f"  Resolution: {props['width']}x{props['height']}")
        print(f"  FPS Target: {props['fps']}")
        print(f"  Detection: YOLOv8s (conf={args.confidence})")
        print(f"  Segmentation: {'Enabled' if not args.no_segmentation else 'Disabled'}")
        print(f"  Trajectories: {'Enabled' if not args.no_trajectories else 'Disabled'}")
        print(f"  Speed: {'Enabled' if not args.no_speed else 'Disabled'}")
        print("="*60 + "\n")
        
        # Main processing loop
        fps_tracker = FPSTracker()
        frame_count = 0
        is_paused = False
        
        print("Processing... Press 'q' to quit\n")
        
        while video_source.is_opened():
            # Read frame
            ret, frame = video_source.read()
            if not ret:
                print("\n✓ End of video stream")
                break
            
            frame_count += 1
            
            # Handle pause
            if is_paused:
                if not args.no_display:
                    cv2.imshow(window_name, frame)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p') or key == ord('P'):
                    is_paused = False
                    print("▶ Resumed")
                elif key == ord('q') or key == ord('Q') or key == 27:
                    print("\n⏹ Quit requested")
                    break
                continue
            
            # Process frame
            processed_frame = frame.copy()
            
            # Detection
            detections = detector.detect(frame)
            
            # Tracking
            tracks = tracker.update(detections)
            
            # Visualization
            if not args.no_segmentation and any(d.mask is not None for d in detections):
                draw_segmentation_masks(processed_frame, detections, alpha=0.4)
            
            draw_tracks(
                processed_frame,
                tracks,
                show_speed=not args.no_speed,
                show_trajectory=not args.no_trajectories
            )
            
            # Add overlays
            current_fps = fps_tracker.update()
            add_overlays(processed_frame, current_fps, len(detections), len(tracks), is_paused)
            
            # Display
            if not args.no_display:
                cv2.imshow(window_name, processed_frame)
            
            # Record
            if recorder:
                recorder.write(processed_frame)
            
            # Progress (for video files)
            if props['source_type'] == 'file' and frame_count % 30 == 0:
                progress = (frame_count / props['total_frames']) * 100
                print(f"  Progress: {progress:.1f}% | Frame {frame_count}/{props['total_frames']} | "
                      f"FPS: {current_fps:.1f} | Tracks: {len(tracks)}")
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n⏹ Quit requested")
                break
            elif key == ord('p') or key == ord('P'):
                is_paused = True
                print("⏸ Paused (press 'p' to resume)")
            elif key == ord('s') or key == ord('S'):
                filename = save_frame(processed_frame)
                print(f"💾 Saved: {filename}")
        
        # Summary
        print("\n" + "="*60)
        print("  PROCESSING COMPLETE")
        print("="*60)
        print(f"  Total frames: {frame_count}")
        print(f"  Average FPS: {frame_count / (sum(fps_tracker.times) or 1):.2f}")
        if recorder:
            print(f"  Output saved: {args.output}")
        print("="*60 + "\n")
    
    except VideoSourceError as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  • For webcam: Make sure camera is connected and not in use")
        print("  • For video: Check file path and format (MP4, AVI, MOV)")
        return 1
    
    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        video_source.release()
        if recorder:
            recorder.release()
        cv2.destroyAllWindows()
        print("✓ Done\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
