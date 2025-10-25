"""
Video source abstraction for webcam and video files.
"""

import cv2
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class VideoSourceError(Exception):
    """Exception for video source errors."""
    pass


class VideoSource(ABC):
    """Abstract base class for video sources."""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame. Returns (success, frame)."""
        pass
    
    @abstractmethod
    def get_properties(self) -> dict:
        """Get video properties (width, height, fps, total_frames)."""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is opened."""
        pass
    
    @abstractmethod
    def release(self):
        """Release resources."""
        pass


class WebcamSource(VideoSource):
    """Webcam video source."""
    
    def __init__(self, device_index: int = 0):
        """
        Initialize webcam source.
        
        Args:
            device_index: Camera device index (0 for default)
        """
        self.device_index = device_index
        self.cap = cv2.VideoCapture(device_index)
        
        if not self.cap.isOpened():
            raise VideoSourceError(f"Cannot open webcam {device_index}")
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if unavailable
        
        print(f"✓ Webcam {device_index} opened: {self.width}x{self.height} @ {self.fps} FPS")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from webcam."""
        return self.cap.read()
    
    def get_properties(self) -> dict:
        """Get webcam properties."""
        return {
            'source_type': 'webcam',
            'device_index': self.device_index,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': -1  # Infinite for webcam
        }
    
    def is_opened(self) -> bool:
        """Check if webcam is opened."""
        return self.cap.isOpened()
    
    def release(self):
        """Release webcam."""
        self.cap.release()


class VideoFileSource(VideoSource):
    """Video file source."""
    
    def __init__(self, file_path: str):
        """
        Initialize video file source.
        
        Args:
            file_path: Path to video file
        """
        self.file_path = file_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        self.cap = cv2.VideoCapture(file_path)
        
        if not self.cap.isOpened():
            raise VideoSourceError(f"Cannot open video file: {file_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        print(f"✓ Video file opened: {os.path.basename(file_path)}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video file."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame
    
    def get_properties(self) -> dict:
        """Get video file properties."""
        return {
            'source_type': 'file',
            'file_path': self.file_path,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'current_frame': self.current_frame
        }
    
    def get_progress(self) -> float:
        """Get progress percentage."""
        if self.total_frames <= 0:
            return 0.0
        return (self.current_frame / self.total_frames) * 100.0
    
    def is_opened(self) -> bool:
        """Check if video file is opened."""
        return self.cap.isOpened()
    
    def release(self):
        """Release video file."""
        self.cap.release()


def create_video_source(source) -> VideoSource:
    """
    Factory function to create appropriate video source.
    
    Args:
        source: Either device index (int or digit string) or file path
        
    Returns:
        VideoSource instance (WebcamSource or VideoFileSource)
    """
    # Check if it's a device index
    if isinstance(source, int):
        return WebcamSource(source)
    
    # Check if string is a digit
    if isinstance(source, str) and source.isdigit():
        return WebcamSource(int(source))
    
    # Otherwise assume it's a file path
    return VideoFileSource(source)
