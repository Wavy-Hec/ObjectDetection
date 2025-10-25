# Real-Time Object Tracking System with Enhancements

A modular real-time object tracking system that detects and tracks people and cars using YOLOv8s for detection, YOLOv8-seg for segmentation, and SORT for tracking with speed estimation and trajectory visualization.

## System Status

✅ **Feature-1:** Object Detection (YOLOv8s) - Complete  
✅ **Feature-2:** SORT Tracking + Enhancements - Complete  
✅ **Feature-3:** Real-Time Video & Webcam Support - Complete  
✅ **Enhancements:** Segmentation, Speed, Trajectories - Complete  
✅ **Validation:** All tests passing  
✅ **Production Ready**

## Key Features

🎯 **Detection:** YOLOv8s for person & car detection  
🎨 **Segmentation:** YOLOv8-seg for instance masks  
🔍 **Tracking:** SORT algorithm with persistent IDs  
⚡ **Speed:** Real-time velocity estimation (px/s)  
📍 **Trajectories:** Visual path history with fading effect  
🎥 **Video Processing:** Live webcam and video file support  
⌨️ **Interactive Controls:** Keyboard controls (q, p, s)  
📊 **Real-Time Stats:** FPS counter and object counts

---

## Project Structure

This repository follows a feature-based development structure where each feature iteration has its own subdirectory:

```
202610-week05-project-Wavy-Hec/
├── README.md                   # This file - Main project overview
├── IMPROVEMENTS.md             # Future enhancement suggestions
├── REPOSITORY_STATUS.md        # Repository organization guide
├── FEATURE3_IMPROVEMENTS.md    # Feature-3 specific improvements
│
├── templates/                  # Reusable templates (evolve as you learn)
│   ├── PRD_template.md         # Template for feature PRDs
│   └── task_template.md        # Template for task files
│
├── commands/                   # Development workflow commands (improve iteratively)
│   ├── plan.md                 # Planning phase prompt
│   ├── design.md               # Design phase prompt
│   ├── implement.md            # Implementation phase prompt
│   └── validate.md             # Validation phase prompt
│
├── feature-1/                  # First feature iteration (Basic Detection)
│   ├── PRD.md                  # Feature requirements
│   ├── design.md               # Feature design
│   ├── plan.md                 # Feature implementation plan
│   ├── post_development_analysis.md  # Lessons learned
│   └── tasks/                  # Task breakdowns
│       ├── tasks_README.md
│       ├── task6_testing.md
│       └── task7_demo.md
│
├── feature-2/                  # Second feature iteration (SORT Tracking)
│   ├── PRD.md                  # Feature requirements
│   ├── design.md               # Feature design
│   ├── plan.md                 # Feature implementation plan
│   ├── validate.md             # Validation report
│   └── post_development_analysis.md  # Lessons learned
│
├── feature-3/                  # Third feature iteration (Video & Webcam)
│   ├── PRD.md                  # Feature requirements
│   ├── design.md               # Feature design
│   ├── plan.md                 # Feature implementation plan
│   ├── QUICKSTART.md           # Quick start guide
│   ├── USAGE.md                # Usage documentation
│   └── TESTING_GUIDE.md        # Testing guide
│
└── project/                    # Actual project source code
    ├── src/                    # Core source modules
    │   ├── detector.py         # Object detection module
    │   ├── tracker.py          # SORT tracking module
    │   ├── video_source.py     # Video/webcam source abstraction
    │   └── visualization.py    # Visualization utilities
    ├── main.py                 # Real-time video processing application
    ├── tests/                  # Test suite
    │   ├── test_detector.py    # Detector tests
    │   └── test_tracker.py     # Tracker tests
    ├── scripts/                # Utility scripts
    │   ├── benchmark.py        # Performance benchmarking
    │   └── create_test_video.py # Test data generation
    ├── demos/                  # Demo images (not in git)
    │   ├── original/           # Source images
    │   └── results/            # Generated posters
    ├── environment.yml         # Conda environment specification
    ├── config.yaml             # Configuration file
    ├── demo.py                 # Demo application
    ├── yolov8s.pt              # YOLOv8s model weights
    └── yolov8s-seg.pt          # YOLOv8-seg model weights
```

### Structure Philosophy

- **README.md**: Main entry point with overview and quick start
- **IMPROVEMENTS.md**: Roadmap of 30+ enhancement suggestions
- **templates/**: Generic templates that evolve as you refine your process
- **commands/**: Reusable command prompts for AI-assisted development workflow
- **feature-N/**: Each feature iteration in its own directory with PRDs and task files
- **project/**: The actual codebase being developed

---

## Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd project/

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate object_tracking

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
```

### 2. Test the Detector

```bash
# Run basic detector test (from project directory)
python detector.py
```

Expected output:
```
ObjectDetector initialized on cuda  # or 'cpu' if no GPU
Model: yolov8s.pt, Confidence threshold: 0.25
Target classes: {'person', 'car'}

Test detection completed.
Number of detections: 0  # (empty test frame)
Detector module test completed successfully!
```

---

## Features

### ✅ Feature-1: Object Detection (YOLOv8s)
- YOLOv8s-based detector for people and cars
- GPU/CPU auto-detection
- Configurable confidence threshold
- 30-50 FPS on GPU, 5-10 FPS on CPU

### ✅ Feature-2: SORT Tracking + Enhancements
- Multi-object tracking with SORT algorithm
- Kalman filter for trajectory prediction
- IOU-based data association
- Unique ID assignment per track
- Track lifecycle management
- **NEW: Speed estimation** from Kalman velocities (px/s)
- **NEW: Trajectory visualization** with 30-point history
- **NEW: Instance segmentation** with colored transparent masks

### ✅ Feature-3: Real-Time Video & Webcam Support
- Live webcam stream processing
- Video file support (MP4, AVI, MOV)
- Real-time display with FPS counter
- Interactive keyboard controls (q=quit, p=pause, s=save)
- Video recording to file
- Performance monitoring and statistics
- Headless mode for batch processing
- Progress tracking for video files

---

## Development Workflow

This project follows a structured AI-assisted development workflow:

### For Each Feature:

1. **Create Feature Branch**
   ```bash
   git checkout -b feature-2
   ```

2. **Create Feature Directory**
   ```bash
   mkdir -p feature-2/tasks
   ```

3. **Plan Phase** - Use `commands/plan.md`
   - Review existing architecture
   - Define feature requirements (PRD)
   - Break down into concrete tasks
   
4. **Design Phase** - Use `commands/design.md`
   - Create technical design
   - Define interfaces and data structures
   - Document architecture decisions

5. **Implementation Phase** - Use `commands/implement.md`
   - Implement one task at a time
   - Write tests alongside code
   - Validate each task before moving on

6. **Validation Phase** - Use `commands/validate.md`
   - Run comprehensive tests
   - Measure performance
   - Generate validation report

7. **Post-Development Analysis**
   - Document lessons learned in `feature-N/post_development_analysis.md`
   - Update templates and commands based on learnings
   - Refine process for next iteration

8. **Merge to Main**
   ```bash
   git checkout main
   git merge feature-2
   git push origin main
   ```

---

## Usage

### Real-Time Video Processing (Feature-3)

```bash
# Process webcam feed
python project/main.py --input 0

# Process video file
python project/main.py --input video.mp4

# Process and save output
python project/main.py --input video.mp4 --output tracked.mp4

# Adjust confidence threshold
python project/main.py --input video.mp4 --confidence 0.5

# Disable specific features
python project/main.py --input 0 --no-segmentation --no-trajectories

# Headless mode (no display)
python project/main.py --input video.mp4 --output tracked.mp4 --no-display

# Keyboard controls while running:
# q / ESC - Quit
# p       - Pause/Resume
# s       - Save current frame
```

### Detection with Segmentation (Feature-1)

```python
from project.src.detector import ObjectDetector
import cv2

# Initialize detector with segmentation
detector = ObjectDetector(
    model_name='yolov8s.pt',
    conf_threshold=0.25,
    use_segmentation=True  # Enable instance segmentation
)

# Load and process a frame
frame = cv2.imread('image.jpg')
detections = detector.detect(frame)

# Process results
for det in detections:
    print(f"{det.class_label}: {det.confidence:.2f}")
    print(f"  Bbox: {det.bbox}")
    if det.mask is not None:
        print(f"  Mask shape: {det.mask.shape}")
```

### Tracking with Speed and Trajectories

```python
from project.src.tracker import Tracker

# Initialize tracker
tracker = Tracker(max_age=1, min_hits=3, iou_threshold=0.3)

# Update tracker with detections
tracks = tracker.update(detections)

# Process tracks
for track in tracks:
    print(f"Track {track.id}: {track.class_label}")
    print(f"  Speed: {track.get_speed():.1f} px/s")
    print(f"  History: {len(track.history)} points")
    print(f"  Velocity: {track.velocity}")
```

### Create Enhanced Posters

```bash
# Create poster with all features
python project/create_enhanced_poster.py image.jpg

# Features included:
#  - Original image + Enhanced view
#  - Instance segmentation (colored masks)
#  - Track IDs and bounding boxes
#  - Speed estimation (px/s)
#  - Trajectory visualization
```



---

## Requirements

### Software
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Conda or pip for package management

### Hardware
**Minimum:**
- CPU: Intel Core i5 or equivalent
- RAM: 8GB
- Storage: 2GB

**Recommended:**
- GPU: NVIDIA RTX 3050 or better
- RAM: 16GB
- Storage: 5GB

---

## Dependencies

Core packages (see `environment.yml` for versions):
- `torch` - PyTorch for deep learning
- `ultralytics` - YOLOv8 implementation
- `opencv-python` - Video I/O and visualization
- `numpy` - Array operations
- `filterpy` - Kalman filter (for feature-2)
- `scipy` - Scientific computing
- `motmetrics` - Tracking evaluation

---

## Performance Targets

### Detection (feature-1)
- GPU: Model loads successfully, inference works
- CPU: Model loads successfully, inference works

### Full System (feature-2+)
- GPU: ≥30 FPS @ 640×480
- CPU: ≥15 FPS @ 640×480
- MOTA: >0.80 on MOT16 dataset
- ID Switches: <5 per minute

---

## Testing

```bash
# Navigate to project directory
cd project/

# Run all tests
pytest -v

# Check coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest test_detector.py -v
```

---

## Project Documentation

- **README.md** - This file (overview, setup, usage)
- **feature-1/PRD.md** - Detection requirements  
- **feature-1/design.md** - Detection technical design  
- **feature-1/plan.md** - Detection task breakdown  
- **feature-1/validate.md** - Detection validation report  
- **feature-2/PRD.md** - Tracking requirements  
- **feature-2/design.md** - Tracking technical design  
- **feature-2/plan.md** - Tracking task breakdown  
- **feature-2/validate.md** - Tracking + enhancements validation  
- **feature-3/PRD.md** - Video/webcam requirements  
- **feature-3/design.md** - Video/webcam technical design  
- **feature-3/plan.md** - Video/webcam task breakdown  
- **feature-3/USAGE.md** - Video/webcam usage guide  
- **feature-3/TESTING_GUIDE.md** - Video/webcam testing guide

---

## Demo Images

**Enhanced Posters** (Original + Segmentation + Tracking + Speed + Trajectories):
- `project/enhanced_poster_DSCF0880.jpg` - 9 persons tracked
- `project/enhanced_poster_DSCF3418.jpg` - 2 persons tracked  
- `project/enhanced_poster_20250808_092327.jpg` - 3 persons tracked

**Basic Posters** (Original + Detection + Tracking):
- `project/poster_DSCF0880.jpg` - 12 persons
- `project/poster_DSCF3418.jpg` - 2 persons
- `project/poster_20250808_092327.jpg` - 3 persons

---

## Known Issues

None currently. All features working as expected.

**Completed:**
- ✅ Comprehensive test suite (21 tests passing)
- ✅ Full pipeline implemented (detection → tracking)
- ✅ Segmentation, speed, trajectories added
- ✅ Performance benchmarks validated

---

## Roadmap

### Phase 1: Foundation (Complete)
- ✅ Environment setup
- ✅ Project structure
- ✅ Basic detector module
- ✅ SORT tracker module

### Phase 2: Enhancements (Complete)
- ✅ Instance segmentation (YOLOv8-seg)
- ✅ Speed estimation from Kalman filter
- ✅ Trajectory visualization
- ✅ Enhanced poster generation
- ✅ Comprehensive testing

### Phase 3: Video & Webcam (Complete)
- ✅ Video processing pipeline (main.py)
- ✅ Real-time webcam support
- ✅ Interactive controls and recording
- ✅ FPS monitoring and statistics

### Phase 4: Future Extensions (Optional)
- ⏳ MOT16 evaluation
- ⏳ Deep SORT with appearance features
- ⏳ Multi-camera fusion

---

## References

- [SORT Paper](https://arxiv.org/abs/1602.00763)
- [Ultralytics Repository](https://github.com/ultralytics/ultralytics)
- [MOT Challenge](https://motchallenge.net/)
- [motmetrics Library](https://github.com/cheind/py-motmetrics)

---

## License

Academic/Educational Project

---

## Contact

For questions about this project structure and workflow, see the course materials.

---

**Last Updated:** 2025-01-27  
**Version:** 3.0 (All features complete: Detection, Tracking, Video/Webcam)
