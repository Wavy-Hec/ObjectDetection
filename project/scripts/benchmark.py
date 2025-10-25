#!/usr/bin/env python3
"""
Performance benchmarking script for the tracking system
Measures FPS, latency, memory usage, and tracking metrics
"""

import cv2
import numpy as np
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List
import json
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detector import ObjectDetector
from tracker import Tracker


@dataclass
class FrameMetrics:
    """Metrics for a single frame"""
    frame_id: int
    detection_time: float  # seconds
    tracking_time: float   # seconds
    total_time: float      # seconds
    num_detections: int
    num_tracks: int
    memory_mb: float


@dataclass
class BenchmarkResults:
    """Overall benchmark results"""
    total_frames: int
    total_time: float
    avg_fps: float
    avg_detection_time: float
    avg_tracking_time: float
    avg_total_time: float
    peak_memory_mb: float
    avg_memory_mb: float
    avg_detections_per_frame: float
    avg_tracks_per_frame: float
    frame_metrics: List[Dict]


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_video(
    video_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.3,
    max_frames: int = None
) -> BenchmarkResults:
    """
    Benchmark tracking system on a video
    
    Args:
        video_path: Path to video file or camera index
        conf_threshold: Detection confidence threshold
        iou_threshold: Tracking IOU threshold
        max_frames: Maximum frames to process (None = all)
        
    Returns:
        BenchmarkResults object
    """
    print(f"Benchmarking video: {video_path}")
    print("=" * 60)
    
    # Initialize components
    print("Initializing detector and tracker...")
    detector = ObjectDetector(
        model_name='yolov5su.pt',
        conf_threshold=conf_threshold
    )
    tracker = Tracker(
        max_age=1,
        min_hits=3,
        iou_threshold=iou_threshold
    )
    
    # Open video
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Processing {total_frames} frames...")
    print("-" * 60)
    
    # Collect metrics
    frame_metrics = []
    start_time = time.time()
    
    frame_id = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_id >= max_frames):
                break
            
            # Measure detection
            det_start = time.time()
            detections = detector.detect(frame)
            det_time = time.time() - det_start
            
            # Measure tracking
            track_start = time.time()
            tracks = tracker.update(detections)
            track_time = time.time() - track_start
            
            # Total frame time
            frame_time = time.time() - det_start
            
            # Memory usage
            mem_usage = get_memory_usage()
            
            # Record metrics
            metrics = FrameMetrics(
                frame_id=frame_id,
                detection_time=det_time,
                tracking_time=track_time,
                total_time=frame_time,
                num_detections=len(detections),
                num_tracks=len(tracks),
                memory_mb=mem_usage
            )
            frame_metrics.append(metrics)
            
            # Progress update
            if (frame_id + 1) % 30 == 0:
                fps = (frame_id + 1) / (time.time() - start_time)
                print(f"Frame {frame_id + 1}/{total_frames} | "
                      f"FPS: {fps:.1f} | "
                      f"Det: {det_time*1000:.1f}ms | "
                      f"Track: {track_time*1000:.1f}ms | "
                      f"Tracks: {len(tracks)}")
            
            frame_id += 1
    
    finally:
        cap.release()
    
    total_time = time.time() - start_time
    
    # Calculate aggregate statistics
    avg_fps = frame_id / total_time if total_time > 0 else 0
    avg_det_time = np.mean([m.detection_time for m in frame_metrics])
    avg_track_time = np.mean([m.tracking_time for m in frame_metrics])
    avg_total_time = np.mean([m.total_time for m in frame_metrics])
    peak_mem = max([m.memory_mb for m in frame_metrics])
    avg_mem = np.mean([m.memory_mb for m in frame_metrics])
    avg_detections = np.mean([m.num_detections for m in frame_metrics])
    avg_tracks = np.mean([m.num_tracks for m in frame_metrics])
    
    results = BenchmarkResults(
        total_frames=frame_id,
        total_time=total_time,
        avg_fps=avg_fps,
        avg_detection_time=avg_det_time,
        avg_tracking_time=avg_track_time,
        avg_total_time=avg_total_time,
        peak_memory_mb=peak_mem,
        avg_memory_mb=avg_mem,
        avg_detections_per_frame=avg_detections,
        avg_tracks_per_frame=avg_tracks,
        frame_metrics=[asdict(m) for m in frame_metrics]
    )
    
    return results


def print_results(results: BenchmarkResults):
    """Print benchmark results in a nice format"""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n📊 Overall Performance:")
    print(f"  Total Frames:     {results.total_frames}")
    print(f"  Total Time:       {results.total_time:.2f}s")
    print(f"  Average FPS:      {results.avg_fps:.2f}")
    print(f"\n⏱️  Timing Breakdown (per frame):")
    print(f"  Detection:        {results.avg_detection_time*1000:.2f}ms")
    print(f"  Tracking:         {results.avg_tracking_time*1000:.2f}ms")
    print(f"  Total:            {results.avg_total_time*1000:.2f}ms")
    print(f"\n💾 Memory Usage:")
    print(f"  Peak:             {results.peak_memory_mb:.1f}MB")
    print(f"  Average:          {results.avg_memory_mb:.1f}MB")
    print(f"\n🎯 Detection & Tracking:")
    print(f"  Avg Detections:   {results.avg_detections_per_frame:.1f}")
    print(f"  Avg Tracks:       {results.avg_tracks_per_frame:.1f}")
    print("=" * 60)


def save_results(results: BenchmarkResults, output_path: str):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark object tracking performance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input video file or camera index'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='benchmark_results.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--conf',
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
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process'
    )
    
    args = parser.parse_args()
    
    try:
        # Run benchmark
        results = benchmark_video(
            video_path=args.input,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_frames=args.max_frames
        )
        
        # Print results
        print_results(results)
        
        # Save results
        save_results(results, args.output)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
