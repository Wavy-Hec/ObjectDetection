# Feature-3 Implementation Plan: Real-Time Video & Webcam Support

**Feature:** Real-Time Video Processing  
**Estimated Duration:** 15-21 hours (2-3 days)  
**Dependencies:** Feature-1 (detector.py) and Feature-2 (tracker.py) complete

---

## Task Overview

| Task | Description | Time | Dependencies |
|------|-------------|------|--------------|
| Task 1 | VideoSource abstraction | 2-3 hours | None |
| Task 2 | FrameProcessor pipeline | 2-3 hours | Task 1 |
| Task 3 | DisplayManager & overlays | 2-3 hours | Task 2 |
| Task 4 | KeyboardController | 1-2 hours | Task 3 |
| Task 5 | VideoRecorder | 1-2 hours | Task 2 |
| Task 6 | Main application | 3-4 hours | All previous |
| Task 7 | Testing | 3-4 hours | Task 6 |
| Task 8 | Documentation | 1-2 hours | Task 7 |

**Total Estimated Time:** 15-21 hours

---

## Task 1: VideoSource Abstraction

**Duration:** 2-3 hours  
**Priority:** High  
**Dependencies:** None

### Description
Create the VideoSource abstract base class and concrete implementations for webcam and video files.

### Files to Create
- `project/src/video_source.py`

### Implementation Steps

1. **Create Abstract Base Class** (30 min)
   ```python
   from abc import ABC, abstractmethod
   import cv2
   import numpy as np
   from typing import Tuple, Optional
   
   class VideoSource(ABC):
       @abstractmethod
       def read(self) -> Tuple[bool, Optional[np.ndarray]]:
           pass
       
       @abstractmethod
       def get_properties(self) -> dict:
           pass
       
       @abstractmethod
       def is_opened(self) -> bool:
           pass
       
       @abstractmethod
       def release(self):
           pass
   ```

2. **Implement WebcamSource** (45 min)
   - Initialize cv2.VideoCapture with device index
   - Validate webcam opens successfully
   - Set buffer size to 1 for low latency
   - Get properties (width, height, fps)
   - Implement read(), is_opened(), release()

3. **Implement VideoFileSource** (45 min)
   - Validate file exists
   - Initialize cv2.VideoCapture with file path
   - Get properties including total_frames
   - Track current_frame for progress
   - Implement all abstract methods

4. **Add Factory Function** (15 min)
   ```python
   def create_video_source(source) -> VideoSource:
       if isinstance(source, int) or source.isdigit():
           return WebcamSource(int(source))
       else:
           return VideoFileSource(source)
   ```

5. **Error Handling** (15 min)
   - Custom exceptions (VideoSourceError)
   - Validation in constructors
   - Graceful error messages

### Acceptance Criteria
- ✓ VideoSource ABC defines interface
- ✓ WebcamSource works with device index
- ✓ VideoFileSource works with file path
- ✓ Factory function correctly identifies source type
- ✓ Properties dict returns correct metadata
- ✓ Error handling for invalid sources
- ✓ Resources properly released

### Testing
```python
def test_webcam_source():
    # Mock webcam or skip if unavailable
    pass

def test_video_file_source():
    # Use test video file
    source = VideoFileSource('test.mp4')
    assert source.is_opened()
    ret, frame = source.read()
    assert ret and frame is not None
    source.release()

def test_invalid_source():
    with pytest.raises(FileNotFoundError):
        VideoFileSource('nonexistent.mp4')
```

---

## Task 2: FrameProcessor Pipeline

**Duration:** 2-3 hours  
**Priority:** High  
**Dependencies:** Task 1

### Description
Create the FrameProcessor class that orchestrates detection, tracking, and visualization.

### Files to Create
- `project/src/processor.py`

### Implementation Steps

1. **Create FrameProcessor Class** (30 min)
   - Initialize with detector, tracker
   - Add feature flags (segmentation, trajectories, speed)
   - Initialize statistics counters

2. **Implement process() Method** (1 hour)
   ```python
   def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
       processed_frame = frame.copy()
       
       # Detection
       detections = self.detector.detect(frame)
       
       # Tracking
       tracks = self.tracker.update(detections)
       
       # Visualization
       if self.enable_segmentation:
           draw_segmentation_masks(processed_frame, detections)
       draw_tracks(processed_frame, tracks, 
                  show_speed=self.enable_speed,
                  show_trajectory=self.enable_trajectories)
       
       # Statistics
       stats = self._compute_stats(detections, tracks)
       
       return processed_frame, stats
   ```

3. **Statistics Tracking** (30 min)
   - Frame counter
   - Detection/track counts
   - Running averages
   - Return stats dict

4. **Error Handling** (15 min)
   - Handle detection failures gracefully
   - Log warnings but continue processing
   - Return empty results on failure

5. **Optimization** (15 min)
   - Skip segmentation if no masks
   - Conditional drawing based on flags

### Acceptance Criteria
- ✓ Integrates detector and tracker correctly
- ✓ Returns processed frame with visualizations
- ✓ Returns statistics dictionary
- ✓ Handles errors without crashing
- ✓ Feature flags work correctly
- ✓ Statistics are accurate

### Testing
```python
def test_processor_basic(mock_detector, mock_tracker):
    processor = FrameProcessor(mock_detector, mock_tracker)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    processed, stats = processor.process(frame)
    assert processed.shape == frame.shape
    assert 'num_detections' in stats

def test_processor_with_detections(mock_detector, mock_tracker):
    # Test with actual detections and tracks
    pass
```

---

## Task 3: DisplayManager & Overlays

**Duration:** 2-3 hours  
**Priority:** High  
**Dependencies:** Task 2

### Description
Create DisplayManager for real-time display with FPS counter and statistics overlays.

### Files to Create
- `project/src/display.py`

### Implementation Steps

1. **Create DisplayManager Class** (30 min)
   - Initialize window with cv2.namedWindow
   - Set up FPS calculation
   - Configure overlay options

2. **Implement show() Method** (45 min)
   - Copy frame for non-destructive overlay
   - Add FPS counter
   - Add statistics overlay
   - Call cv2.imshow()

3. **FPS Counter** (30 min)
   ```python
   def _add_fps_overlay(self, frame, fps):
       text = f"FPS: {fps:.1f}"
       cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0, 255, 0), 2)
   ```

4. **Statistics Overlay** (30 min)
   - Display detection count
   - Display track count
   - Display frame number (for videos)
   - Proper positioning and formatting

5. **FPS Tracking Class** (30 min)
   ```python
   class FPSTracker:
       def __init__(self, buffer_size=30):
           self.times = []
           self.buffer_size = buffer_size
           self.last_time = time.time()
       
       def update(self) -> float:
           current_time = time.time()
           self.times.append(current_time - self.last_time)
           self.last_time = current_time
           
           if len(self.times) > self.buffer_size:
               self.times.pop(0)
           
           avg_time = sum(self.times) / len(self.times)
           return 1.0 / avg_time if avg_time > 0 else 0
   ```

### Acceptance Criteria
- ✓ Window displays correctly
- ✓ FPS counter updates smoothly
- ✓ Statistics overlay readable
- ✓ No flickering or lag
- ✓ Proper resource cleanup

### Testing
```python
def test_display_manager():
    display = DisplayManager()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stats = {'fps': 30, 'num_detections': 5}
    # Manual visual check
    display.show(frame, stats)
    display.close()

def test_fps_tracker():
    tracker = FPSTracker()
    for _ in range(10):
        time.sleep(0.033)  # ~30 FPS
        fps = tracker.update()
    assert 25 < fps < 35  # Roughly 30 FPS
```

---

## Task 4: KeyboardController

**Duration:** 1-2 hours  
**Priority:** Medium  
**Dependencies:** Task 3

### Description
Implement keyboard controls for quit, pause, and save functionality.

### Files to Update
- `project/src/controller.py` (new file)

### Implementation Steps

1. **Create KeyboardController Class** (30 min)
   ```python
   class KeyboardController:
       def __init__(self):
           self.is_paused = False
           self.should_quit = False
           self.should_save = False
           
           self.key_mapping = {
               ord('q'): 'quit',
               ord('p'): 'pause',
               ord('s'): 'save',
               27: 'quit'  # ESC
           }
   ```

2. **Implement handle_key() Method** (30 min)
   - Map key codes to actions
   - Update internal state
   - Return action string

3. **State Management** (15 min)
   - Pause/resume toggle
   - Quit flag
   - Save flag (auto-reset after use)

4. **Print Controls** (15 min)
   - Static method to print instructions
   - Call on startup

### Acceptance Criteria
- ✓ 'q' and ESC quit application
- ✓ 'p' pauses and resumes
- ✓ 's' triggers save
- ✓ Case-insensitive
- ✓ State accessible via get_state()

### Testing
```python
def test_keyboard_controller():
    controller = KeyboardController()
    
    # Test pause
    controller.handle_key(ord('p'))
    assert controller.is_paused
    
    # Test quit
    controller.handle_key(ord('q'))
    assert controller.should_quit
    
    # Test save
    controller.handle_key(ord('s'))
    assert controller.should_save
```

---

## Task 5: VideoRecorder

**Duration:** 1-2 hours  
**Priority:** Medium  
**Dependencies:** Task 2

### Description
Implement video recording functionality to save processed output.

### Files to Update
- `project/src/display.py` (add VideoRecorder class)

### Implementation Steps

1. **Create VideoRecorder Class** (45 min)
   ```python
   class VideoRecorder:
       def __init__(self, output_path, width, height, fps):
           fourcc = cv2.VideoWriter_fourcc(*'mp4v')
           self.writer = cv2.VideoWriter(
               output_path, fourcc, fps, (width, height)
           )
           self.frames_written = 0
   ```

2. **Implement write() Method** (15 min)
   - Resize frame if needed
   - Write to video file
   - Track frames written

3. **Auto-resize Support** (15 min)
   - Check frame dimensions
   - Resize if mismatch
   - Preserve aspect ratio option

4. **Progress Tracking** (15 min)
   - Return frames_written
   - Calculate percentage (if total known)

### Acceptance Criteria
- ✓ Creates valid MP4 file
- ✓ Preserves resolution
- ✓ Handles resize correctly
- ✓ Tracks frames written
- ✓ Proper cleanup on release

### Testing
```python
def test_video_recorder(tmp_path):
    output = tmp_path / "test_output.mp4"
    recorder = VideoRecorder(str(output), 640, 480, 30)
    
    for _ in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        recorder.write(frame)
    
    recorder.release()
    
    # Verify file exists and is valid
    assert output.exists()
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    cap.release()
```

---

## Task 6: Main Application

**Duration:** 3-4 hours  
**Priority:** Critical  
**Dependencies:** Tasks 1-5

### Description
Create main.py that ties all components together with CLI and main loop.

### Files to Create
- `project/main.py`

### Implementation Steps

1. **Command-Line Parsing** (45 min)
   ```python
   def parse_arguments():
       parser = argparse.ArgumentParser(
           description='Real-time object tracking'
       )
       parser.add_argument('--input', '-i', required=True)
       parser.add_argument('--output', '-o', default=None)
       parser.add_argument('--confidence', type=float, default=0.25)
       # ... more arguments
       return parser.parse_args()
   ```

2. **Component Initialization** (30 min)
   - Create video source
   - Initialize detector and tracker
   - Create processor
   - Setup display and recorder
   - Initialize controller

3. **Main Processing Loop** (1.5 hours)
   ```python
   while video_source.is_opened():
       # Check quit
       if controller.should_quit:
           break
       
       # Handle pause
       if controller.is_paused:
           time.sleep(0.1)
           continue
       
       # Read frame
       ret, frame = video_source.read()
       if not ret:
           break
       
       # Process
       processed_frame, stats = processor.process(frame)
       stats['fps'] = fps_tracker.update()
       
       # Display
       if display:
           display.show(processed_frame, stats)
       
       # Record
       if recorder:
           recorder.write(processed_frame)
       
       # Handle input
       key = cv2.waitKey(1) & 0xFF
       controller.handle_key(key)
       
       # Save frame if requested
       if controller.should_save:
           save_frame(processed_frame)
   ```

4. **Resource Cleanup** (15 min)
   - try-finally block
   - Release all resources
   - Print summary

5. **Progress Display** (30 min)
   - Progress bar for video files
   - ETA calculation
   - Use tqdm or custom progress

6. **Frame Saving** (15 min)
   ```python
   def save_frame(frame):
       timestamp = time.strftime('%Y%m%d_%H%M%S')
       filename = f"frame_{timestamp}.jpg"
       cv2.imwrite(filename, frame)
       print(f"Saved: {filename}")
   ```

### Acceptance Criteria
- ✓ Processes webcam input
- ✓ Processes video files
- ✓ All keyboard controls work
- ✓ Recording works when specified
- ✓ Progress display for videos
- ✓ Proper cleanup on exit
- ✓ Clear error messages
- ✓ Prints summary statistics

### Testing
```python
def test_main_with_video(test_video_path):
    # Integration test with actual video file
    args = ['--input', test_video_path, '--output', 'out.mp4']
    # Run main and verify output
```

---

## Task 7: Testing

**Duration:** 3-4 hours  
**Priority:** High  
**Dependencies:** Task 6

### Description
Write comprehensive test suite for all components.

### Files to Create
- `project/tests/test_video_source.py`
- `project/tests/test_processor.py`
- `project/tests/test_display.py`
- `project/tests/test_controller.py`
- `project/tests/test_integration.py`

### Test Categories

1. **Unit Tests** (2 hours)
   - VideoSource classes (with mocks)
   - FrameProcessor
   - DisplayManager
   - KeyboardController
   - VideoRecorder

2. **Integration Tests** (1 hour)
   - End-to-end with test video
   - Mocked webcam test
   - Recording verification

3. **Performance Tests** (30 min)
   - FPS measurement
   - Memory usage
   - Long-duration stability

4. **Edge Cases** (30 min)
   - Empty video
   - Corrupted frame
   - Invalid input
   - Resource errors

### Acceptance Criteria
- ✓ All unit tests pass
- ✓ Integration tests pass
- ✓ >80% code coverage
- ✓ Performance benchmarks met
- ✓ Edge cases handled

---

## Task 8: Documentation

**Duration:** 1-2 hours  
**Priority:** Medium  
**Dependencies:** Task 7

### Description
Update documentation and create user guide.

### Files to Update/Create
- `README.md` - Add Feature-3 section
- `feature-3/validate.md` - Validation report
- `feature-3/post_development_analysis.md` - Lessons learned

### Documentation Items

1. **README Updates** (30 min)
   - Update feature status
   - Add usage examples
   - Update project structure
   - Add troubleshooting section

2. **Usage Guide** (30 min)
   ```markdown
   ## Usage
   
   ### Webcam
   python main.py --input 0
   
   ### Video File
   python main.py --input video.mp4
   
   ### With Recording
   python main.py --input video.mp4 --output tracked.mp4
   
   ### Advanced Options
   python main.py --input 0 --confidence 0.5 --no-segmentation
   ```

3. **Validation Report** (30 min)
   - Performance benchmarks
   - Test results
   - Known issues
   - Future work

### Acceptance Criteria
- ✓ README complete with examples
- ✓ Usage instructions clear
- ✓ Validation report complete
- ✓ Post-development analysis written

---

## Implementation Checklist

### Pre-Implementation
- [ ] Review PRD and design document
- [ ] Set up feature-3 branch
- [ ] Review existing codebase (feature-1, feature-2)

### During Implementation
- [ ] Task 1: VideoSource abstraction
- [ ] Task 2: FrameProcessor pipeline
- [ ] Task 3: DisplayManager & overlays
- [ ] Task 4: KeyboardController
- [ ] Task 5: VideoRecorder
- [ ] Task 6: Main application
- [ ] Task 7: Testing
- [ ] Task 8: Documentation

### Post-Implementation
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Code review
- [ ] Merge to main

---

## Risk Management

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Webcam unavailable | High | Low | Focus on video file support, mock webcam |
| Performance issues | Medium | High | Add frame skip, resolution scaling options |
| OpenCV compatibility | Low | Medium | Test on multiple platforms |
| Memory leaks | Low | High | Careful resource management, profiling |

---

## Success Metrics

### Must Have
- ✅ Processes webcam OR video files successfully
- ✅ ≥15 FPS on GPU (640×480)
- ✅ All keyboard controls work
- ✅ Recording produces valid output
- ✅ All tests pass

### Should Have
- ✅ ≥20 FPS on GPU (640×480)
- ✅ Progress display for videos
- ✅ Graceful error handling
- ✅ Comprehensive documentation

### Nice to Have
- ⏳ Configuration file support
- ⏳ Real-time analytics overlay
- ⏳ Multiple codec support

---

**Status:** Plan Complete - Ready for Implementation ✅

**Next Step:** Start with Task 1 (VideoSource abstraction)
