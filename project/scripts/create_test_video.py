#!/usr/bin/env python3
"""
Generate a synthetic test video for testing the tracking system
This creates a video with moving colored rectangles simulating objects
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_video(
    output_path: str = 'test_video.mp4',
    duration_sec: int = 10,
    fps: int = 30,
    width: int = 640,
    height: int = 480
):
    """
    Create a synthetic test video with moving objects
    
    Args:
        output_path: Path to save the video
        duration_sec: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
    """
    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration_sec}s @ {fps} FPS")
    print(f"Resolution: {width}x{height}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    # Create moving objects
    # Object 1: Red rectangle (simulates person)
    obj1_start = (50, 200)
    obj1_size = (80, 160)  # width x height (person-like aspect)
    obj1_speed = (3, 0)    # moves right
    
    # Object 2: Blue rectangle (simulates car)
    obj2_start = (400, 300)
    obj2_size = (120, 60)  # width x height (car-like aspect)
    obj2_speed = (-2, 1)   # moves left and down
    
    # Object 3: Green rectangle (another person)
    obj3_start = (300, 100)
    obj3_size = (70, 150)
    obj3_speed = (1, 2)    # moves right and down
    
    print(f"\nGenerating {total_frames} frames...")
    
    for frame_num in range(total_frames):
        # Create blank frame (gray background)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Calculate object positions
        obj1_x = obj1_start[0] + obj1_speed[0] * frame_num
        obj1_y = obj1_start[1] + obj1_speed[1] * frame_num
        
        obj2_x = obj2_start[0] + obj2_speed[0] * frame_num
        obj2_y = obj2_start[1] + obj2_speed[1] * frame_num
        
        obj3_x = obj3_start[0] + obj3_speed[0] * frame_num
        obj3_y = obj3_start[1] + obj3_speed[1] * frame_num
        
        # Keep objects within bounds (wrap around)
        obj1_x = obj1_x % width
        obj1_y = obj1_y % height
        obj2_x = obj2_x % width
        obj2_y = obj2_y % height
        obj3_x = obj3_x % width
        obj3_y = obj3_y % height
        
        # Draw object 1 (Red - "person")
        cv2.rectangle(
            frame,
            (int(obj1_x), int(obj1_y)),
            (int(obj1_x + obj1_size[0]), int(obj1_y + obj1_size[1])),
            (0, 0, 255),  # Red in BGR
            -1
        )
        cv2.putText(
            frame,
            "Person",
            (int(obj1_x), int(obj1_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )
        
        # Draw object 2 (Blue - "car")
        cv2.rectangle(
            frame,
            (int(obj2_x), int(obj2_y)),
            (int(obj2_x + obj2_size[0]), int(obj2_y + obj2_size[1])),
            (255, 0, 0),  # Blue in BGR
            -1
        )
        cv2.putText(
            frame,
            "Car",
            (int(obj2_x), int(obj2_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )
        
        # Draw object 3 (Green - "person")
        # Only appear after frame 60 (2 seconds)
        if frame_num > 60:
            cv2.rectangle(
                frame,
                (int(obj3_x), int(obj3_y)),
                (int(obj3_x + obj3_size[0]), int(obj3_y + obj3_size[1])),
                (0, 255, 0),  # Green in BGR
                -1
            )
            cv2.putText(
                frame,
                "Person",
                (int(obj3_x), int(obj3_y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Add frame number
        cv2.putText(
            frame,
            f"Frame: {frame_num}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Write frame
        out.write(frame)
        
        # Progress
        if (frame_num + 1) % 30 == 0:
            print(f"  Generated {frame_num + 1}/{total_frames} frames...")
    
    out.release()
    
    print(f"\n✅ Test video created: {output_path}")
    print(f"   Duration: {duration_sec}s")
    print(f"   Frames: {total_frames}")
    print(f"   Size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    print(f"\nYou can now use this video with:")
    print(f"  python demo.py --input {output_path}")
    print(f"  python benchmark.py --input {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic test video for tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='test_video.mp4',
        help='Output video file path'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=10,
        help='Video duration in seconds'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Video width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Video height'
    )
    
    args = parser.parse_args()
    
    create_test_video(
        output_path=args.output,
        duration_sec=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height
    )
