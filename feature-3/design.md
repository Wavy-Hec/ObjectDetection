# Feature-3 Design Document: Real-Time Video & Webcam Support

**Version:** 1.0  
**Last Updated:** 2025-01-27  
**Status:** Active

---

## 1. Design Overview

This document details the technical design for implementing real-time video processing capabilities. The system will support both live webcam streams and video files, providing a unified interface for processing, display, recording, and user interaction.

---

## 2. Architecture

### 2.1 System Integration

```
Input Source (Webcam/Video)
    ↓
VideoSource.read()
    ↓
Frame (numpy array)
    ↓
Processor.process(frame)
    ├─→ detector.detect(frame) → detections
    ├─→ tracker.update(detections) → tracks
    └─→ visualizer.draw(frame, tracks) → annotated_frame
    ↓
Display/Recording
    ├─→ display.show(frame)
    ├─→ recorder.write(frame)
    └─→ controller.handle_input()
    ↓
Loop or Exit
```

### 2.2 Module Structure

```
project/
├── main.py                      # Entry point and main loop
├── src/
│   ├── video_source.py          # Video input abstraction
│   │   ├── VideoSource (ABC)
│   │   ├── WebcamSource
│   │   └── VideoFileSource
│   ├── processor.py             # Processing pipeline
│   │   └── FrameProcessor
│   ├── display.py               # Display and recording
│   │   ├── DisplayManager
│   │   └── VideoRecorder
│   ├── controller.py            # User interaction
│   │   └── KeyboardController
│   ├── detector.py              # Existing (feature-1)
│   ├── tracker.py               # Existing (feature-2)
│   └── visualization.py         # Existing (feature-2)
└── tests/
    ├── test_video_source.py
    ├── test_processor.py
    ├── test_display.py
    └── test_integration.py
```

---

## 3. Detailed Component Design

### 3.1 VideoSource (Abstract Base Class)

**Purpose:** Unified interface for video input (webcam or file)

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class VideoSource(ABC):
    """Abstract base class for video sources."""
    
    @abstractmethod
    def __init__(self, source):
        """Initialize video source."""
        pass
    
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
```

**Design Decisions:**
- Abstract base class ensures consistent interface
- Returns tuple (success, frame) matching cv2.VideoCapture
- Properties dict allows different sources to expose relevant metadata

---

### 3.2 WebcamSource

**Purpose:** Handle live webcam input

```python
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
            raise ValueError(f"Cannot open webcam {device_index}")
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if unavailable
        
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
        return self.cap.isOpened()
    
    def release(self):
        self.cap.release()
```

**Design Decisions:**
- Buffer size = 1 to minimize latency
- Default FPS to 30 if hardware doesn't report
- total_frames = -1 indicates infinite stream

---

### 3.3 VideoFileSource

**Purpose:** Handle video file input

```python
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
            raise ValueError(f"Cannot open video file: {file_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
    
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
    
    def is_opened(self) -> bool:
        return self.cap.isOpened()
    
    def release(self):
        self.cap.release()
```

**Design Decisions:**
- Validate file exists before opening
- Track current_frame for progress reporting
- Store all properties at initialization

---

### 3.4 FrameProcessor

**Purpose:** Pipeline for processing each frame

```python
class FrameProcessor:
    """Process video frames through detection and tracking pipeline."""
    
    def __init__(
        self,
        detector: ObjectDetector,
        tracker: Tracker,
        enable_segmentation: bool = True,
        enable_trajectories: bool = True,
        enable_speed: bool = True
    ):
        """
        Initialize frame processor.
        
        Args:
            detector: ObjectDetector instance
            tracker: Tracker instance
            enable_segmentation: Enable segmentation masks
            enable_trajectories: Enable trajectory visualization
            enable_speed: Enable speed overlay
        """
        self.detector = detector
        self.tracker = tracker
        self.enable_segmentation = enable_segmentation
        self.enable_trajectories = enable_trajectories
        self.enable_speed = enable_speed
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.total_tracks = 0
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, stats_dict)
        """
        # Make a copy for processing
        processed_frame = frame.copy()
        
        # Detection
        detections = self.detector.detect(frame)
        
        # Tracking
        tracks = self.tracker.update(detections)
        
        # Visualization
        if self.enable_segmentation and any(d.mask is not None for d in detections):
            draw_segmentation_masks(processed_frame, detections, alpha=0.4)
        
        draw_tracks(
            processed_frame,
            tracks,
            show_speed=self.enable_speed,
            show_trajectory=self.enable_trajectories
        )
        
        # Update statistics
        self.frame_count += 1
        self.total_detections += len(detections)
        self.total_tracks += len(tracks)
        
        stats = {
            'frame_num': self.frame_count,
            'num_detections': len(detections),
            'num_tracks': len(tracks),
            'avg_detections': self.total_detections / self.frame_count,
            'avg_tracks': self.total_tracks / self.frame_count
        }
        
        return processed_frame, stats
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.frame_count = 0
        self.total_detections = 0
        self.total_tracks = 0
```

**Design Decisions:**
- Encapsulates entire detection → tracking → visualization pipeline
- Flags to enable/disable features for performance
- Returns both processed frame and statistics
- Maintains running statistics for analytics

---

### 3.5 DisplayManager

**Purpose:** Handle window display and overlays

```python
class DisplayManager:
    """Manage display window and overlays."""
    
    def __init__(self, window_name: str = "Object Tracking", show_fps: bool = True):
        """
        Initialize display manager.
        
        Args:
            window_name: Name for display window
            show_fps: Show FPS counter
        """
        self.window_name = window_name
        self.show_fps = show_fps
        
        # FPS calculation
        self.fps_history = []
        self.max_fps_samples = 30
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def show(self, frame: np.ndarray, stats: dict = None):
        """
        Display frame with overlays.
        
        Args:
            frame: Frame to display
            stats: Optional statistics dictionary
        """
        display_frame = frame.copy()
        
        if self.show_fps and stats:
            self._add_fps_overlay(display_frame, stats.get('fps', 0))
        
        if stats:
            self._add_stats_overlay(display_frame, stats)
        
        cv2.imshow(self.window_name, display_frame)
    
    def _add_fps_overlay(self, frame: np.ndarray, fps: float):
        """Add FPS counter to frame."""
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 255, 0), 2
        )
    
    def _add_stats_overlay(self, frame: np.ndarray, stats: dict):
        """Add statistics overlay to frame."""
        y_offset = 70
        line_height = 35
        
        if 'num_detections' in stats:
            text = f"Detections: {stats['num_detections']}"
            cv2.putText(
                frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2
            )
            y_offset += line_height
        
        if 'num_tracks' in stats:
            text = f"Tracks: {stats['num_tracks']}"
            cv2.putText(
                frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2
            )
    
    def close(self):
        """Close display window."""
        cv2.destroyWindow(self.window_name)
```

**Design Decisions:**
- Separate class for display concerns
- Non-destructive overlay (copies frame)
- Modular overlay methods for different info types
- FPS smoothing using history buffer

---

### 3.6 VideoRecorder

**Purpose:** Record processed video to file

```python
class VideoRecorder:
    """Record video to file."""
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = 'mp4v'
    ):
        """
        Initialize video recorder.
        
        Args:
            output_path: Output file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec (default: mp4v for MP4)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot open video writer: {output_path}")
        
        self.frames_written = 0
    
    def write(self, frame: np.ndarray):
        """
        Write frame to video file.
        
        Args:
            frame: Frame to write
        """
        # Resize if necessary
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
        self.frames_written += 1
    
    def release(self):
        """Release video writer."""
        self.writer.release()
    
    def get_info(self) -> dict:
        """Get recorder information."""
        return {
            'output_path': self.output_path,
            'frames_written': self.frames_written,
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps
        }
```

**Design Decisions:**
- Auto-resize frames if size mismatch
- Track frames written for progress
- Default to mp4v codec (widely compatible)

---

### 3.7 KeyboardController

**Purpose:** Handle user keyboard input

```python
class KeyboardController:
    """Handle keyboard controls."""
    
    def __init__(self):
        """Initialize keyboard controller."""
        self.is_paused = False
        self.should_quit = False
        self.should_save = False
        
        self.key_mapping = {
            ord('q'): 'quit',
            ord('Q'): 'quit',
            ord('p'): 'pause',
            ord('P'): 'pause',
            ord('s'): 'save',
            ord('S'): 'save',
            27: 'quit'  # ESC key
        }
    
    def handle_key(self, key: int) -> Optional[str]:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            Action string or None
        """
        if key == -1:
            return None
        
        action = self.key_mapping.get(key)
        
        if action == 'quit':
            self.should_quit = True
        elif action == 'pause':
            self.is_paused = not self.is_paused
        elif action == 'save':
            self.should_save = True
        
        return action
    
    def get_state(self) -> dict:
        """Get controller state."""
        return {
            'paused': self.is_paused,
            'should_quit': self.should_quit,
            'should_save': self.should_save
        }
    
    def reset_save_flag(self):
        """Reset save flag after saving."""
        self.should_save = False
    
    @staticmethod
    def print_controls():
        """Print keyboard controls."""
        print("\n" + "="*50)
        print("  Keyboard Controls")
        print("="*50)
        print("  q / ESC  - Quit")
        print("  p        - Pause/Resume")
        print("  s        - Save current frame")
        print("="*50 + "\n")
```

**Design Decisions:**
- State machine for pause/quit/save
- Case-insensitive key handling
- ESC key as alternative quit
- Static method to print instructions

---

### 3.8 Main Application Flow

```python
def main():
    """Main application entry point."""
    
    # 1. Parse command-line arguments
    args = parse_arguments()
    
    # 2. Initialize components
    video_source = create_video_source(args.input)
    detector = ObjectDetector(
        model_name='yolov8s.pt',
        conf_threshold=args.confidence,
        use_segmentation=args.segmentation
    )
    tracker = Tracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold
    )
    processor = FrameProcessor(
        detector, tracker,
        enable_segmentation=args.segmentation,
        enable_trajectories=args.trajectories,
        enable_speed=args.speed
    )
    
    # 3. Setup display/recording
    display = DisplayManager() if not args.no_display else None
    recorder = None
    if args.output:
        props = video_source.get_properties()
        recorder = VideoRecorder(
            args.output,
            props['width'],
            props['height'],
            props['fps']
        )
    
    # 4. Setup controller
    controller = KeyboardController()
    controller.print_controls()
    
    # 5. Print startup info
    print_startup_info(video_source, detector, tracker)
    
    # 6. Main processing loop
    fps_tracker = FPSTracker()
    
    try:
        while video_source.is_opened():
            # Check quit
            if controller.should_quit:
                break
            
            # Handle pause
            if controller.is_paused:
                key = cv2.waitKey(100) & 0xFF
                controller.handle_key(key)
                continue
            
            # Read frame
            ret, frame = video_source.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, stats = processor.process(frame)
            
            # Add FPS to stats
            stats['fps'] = fps_tracker.update()
            
            # Display
            if display:
                display.show(processed_frame, stats)
            
            # Record
            if recorder:
                recorder.write(processed_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            controller.handle_key(key)
            
            # Save frame if requested
            if controller.should_save:
                save_frame(processed_frame)
                controller.reset_save_flag()
            
            # Progress (for video files)
            print_progress(video_source)
    
    finally:
        # 7. Cleanup
        video_source.release()
        if display:
            display.close()
        if recorder:
            recorder.release()
        cv2.destroyAllWindows()
        
        print_summary(processor, recorder)
```

**Design Decisions:**
- Clear separation of initialization, loop, cleanup
- Try-finally ensures resources released
- Modular component creation
- Progress reporting for file processing

---

## 4. Data Flow

### 4.1 Per-Frame Pipeline

```
1. video_source.read() → (success, frame)
2. processor.process(frame) → (processed_frame, stats)
   ├─ detector.detect(frame) → detections
   ├─ tracker.update(detections) → tracks
   └─ visualization.draw(frame, tracks)
3. fps_tracker.update() → fps
4. display.show(processed_frame, stats + fps)
5. recorder.write(processed_frame)
6. controller.handle_key(cv2.waitKey(1))
```

### 4.2 Error Handling Flow

```
Try:
    Main Loop
Except VideoError:
    Log error, attempt recovery
Except KeyboardInterrupt:
    Graceful shutdown
Finally:
    Release all resources
    Print summary
```

---

## 5. Configuration & CLI

### 5.1 Command-Line Arguments

```python
parser = argparse.ArgumentParser(
    description='Real-time object tracking',
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
parser.add_argument('--display-fps', action='store_true', default=True,
                    help='Show FPS counter')
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# test_video_source.py
def test_webcam_source_creation()
def test_video_file_source_creation()
def test_video_file_source_read()
def test_invalid_video_source()

# test_processor.py
def test_processor_initialization()
def test_processor_process_frame()
def test_processor_statistics()

# test_display.py
def test_display_manager_overlays()
def test_recorder_write()

# test_controller.py
def test_keyboard_controls()
def test_pause_resume()
```

### 6.2 Integration Tests

```python
def test_end_to_end_video_processing()
def test_webcam_simulation()
def test_recording_output()
```

---

## 7. Performance Considerations

### 7.1 Optimization Strategies

1. **Frame Buffer Reduction**
   - Set CAP_PROP_BUFFERSIZE=1 for webcam to reduce latency

2. **Conditional Features**
   - Disable segmentation for performance (--no-segmentation)
   - Disable trajectories if not needed

3. **Resolution Scaling**
   - Option to resize frames before processing (future)

4. **GPU Utilization**
   - Ensure detector uses GPU when available
   - Batch processing not needed (single frame pipeline)

### 7.2 Memory Management

- Release VideoCapture after use
- Don't accumulate frames in memory
- Clear trajectory history periodically

---

## 8. Error Handling

### 8.1 Input Errors

```python
# Invalid webcam index
if not cap.isOpened():
    raise ValueError("Cannot open webcam")

# Missing video file
if not os.path.exists(file_path):
    raise FileNotFoundError("Video file not found")

# Corrupted video
if not ret and frame_count == 0:
    raise ValueError("Cannot read first frame")
```

### 8.2 Processing Errors

```python
# Handle detection failures gracefully
try:
    detections = detector.detect(frame)
except Exception as e:
    logger.warning(f"Detection failed: {e}")
    detections = []  # Continue with empty detections
```

### 8.3 Resource Cleanup

```python
# Always cleanup in finally block
finally:
    video_source.release()
    if display:
        display.close()
    if recorder:
        recorder.release()
    cv2.destroyAllWindows()
```

---

## 9. Future Enhancements

### 9.1 Short-term (Post Feature-3)
- Frame skip option for performance
- Resolution scaling
- ROI selection
- Configuration file support

### 9.2 Long-term
- Multi-camera support
- RTSP network streams
- Web-based viewer
- Cloud integration
- Alert/notification system

---

## 10. Design Review Checklist

- ✅ Components are modular and testable
- ✅ Clear separation of concerns
- ✅ Consistent with existing codebase (feature-1, feature-2)
- ✅ Error handling throughout
- ✅ Resource management (proper cleanup)
- ✅ Performance considerations addressed
- ✅ Extensible for future features
- ✅ Well-documented interfaces

---

**Status:** Design Complete - Ready for Implementation ✅
