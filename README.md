# Real-Time Object Tracking System

A modular real-time object tracking system that detects and tracks people and cars in video streams using YOLOv5s for detection and SORT for tracking.

## Project Status

**Current Feature:** feature-1 (Basic Detection) - ✅ Complete  
**Next Feature:** feature-2 (SORT Tracking) - ⏳ Planned  
**Overall Progress:** Foundation established, tracking in development

---

## Project Structure

```
ObjectDetection/
├── PAD.md                      # Project Architecture Document
├── README.md                   # This file
├── environment.yml             # Conda environment specification
│
├── templates/                  # Templates for future features
│   ├── PRD_template.md         # Template for feature PRDs
│   └── task_template.md        # Template for task files
│
├── commands/                   # Commands for development workflow
│   ├── design.md               # Design phase command/prompt
│   ├── plan.md                 # Planning phase command/prompt
│   ├── implement.md            # Implementation phase command/prompt
│   └── validate.md             # Validation phase command/prompt
│
├── feature-1/                  # First feature iteration (Detection)
│   ├── PRD.md                  # Feature requirements
│   ├── design.md               # Feature design
│   ├── plan.md                 # Feature plan
│   ├── post_development_analysis.md  # Lessons learned
│   └── tasks/                  # Task breakdowns
│       ├── tasks_README.md
│       ├── task6_testing.md
│       └── task7_demo.md
│
├── src/                        # Source code
│   └── detector.py             # YOLOv5s object detector
│
└── tests/                      # Test files (to be added)
```

---

## Quick Start

### 1. Environment Setup

```bash
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
# Run basic detector test
python src/detector.py
```

Expected output:
```
ObjectDetector initialized on cuda  # or 'cpu' if no GPU
Model: yolov5s.pt, Confidence threshold: 0.25
Target classes: {'person', 'car'}

Test detection completed.
Number of detections: 0  # (empty test frame)
Detector module test completed successfully!
```

---

## Features

### ✅ Feature-1: Basic Object Detection
- YOLOv5s-based detector for people and cars
- Configurable confidence threshold
- GPU/CPU auto-detection
- Modular, reusable design

### ⏳ Feature-2: SORT Tracking (Planned)
- Multi-object tracking with SORT algorithm
- Kalman filter for trajectory prediction
- IOU-based data association
- Unique ID assignment

### ⏳ Feature-3: Integration Pipeline (Planned)
- End-to-end video processing
- Real-time visualization
- CLI interface
- Performance metrics logging

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

3. **Design Phase** - Use `commands/design.md`
   - Review PAD.md and create feature PRD
   - Create design.md with architecture
   - Define interfaces and data structures

4. **Planning Phase** - Use `commands/plan.md`
   - Break design into tasks
   - Map dependencies
   - Estimate timelines

5. **Implementation Phase** - Use `commands/implement.md`
   - Implement one task at a time
   - Write tests alongside code
   - Validate each task before moving on

6. **Validation Phase** - Use `commands/validate.md`
   - Run comprehensive tests
   - Measure performance
   - Generate validation report

7. **Post-Development Analysis**
   - Document lessons learned
   - Update templates and commands
   - Improve process for next iteration

8. **Merge to Main**
   ```bash
   git checkout main
   git merge feature-2
   git push origin main
   ```

---

## Usage

### Current Capabilities (feature-1)

```python
from src.detector import ObjectDetector
import cv2

# Initialize detector
detector = ObjectDetector(
    model_name='yolov5s.pt',
    conf_threshold=0.25,
    img_size=(640, 480)
)

# Load and process a frame
frame = cv2.imread('image.jpg')
detections = detector.detect(frame)

# Process results
for det in detections:
    print(f"{det.class_label}: {det.confidence:.2f}")
    print(f"  Bbox: {det.bbox}")
```

### Coming Soon (feature-2+)

```bash
# Process video file
python main.py --input video.mp4 --output output.mp4

# Use webcam
python main.py --input 0 --display

# Adjust confidence threshold
python main.py --input video.mp4 --confidence 0.5
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
- `ultralytics` - YOLOv5 implementation
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
# Run all tests (once implemented)
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_detector.py -v
```

---

## Project Documentation

- **PAD.md** - Project Architecture Document (overall system design)
- **feature-X/PRD.md** - Requirements for each feature
- **feature-X/design.md** - Technical design for each feature
- **feature-X/plan.md** - Task breakdown for each feature
- **feature-X/post_development_analysis.md** - Lessons learned

---

## Contributing

This is a structured learning project following a specific workflow:

1. Each feature gets its own branch and directory
2. Follow the commands in `commands/` for each phase
3. Use templates in `templates/` for consistency
4. Update PAD.md carefully when architecture changes
5. Always include tests and documentation

---

## Known Issues

See `feature-1/post_development_analysis.md` for detailed analysis.

**Current:**
- No comprehensive test suite yet
- README usage examples are placeholders (full pipeline not implemented)
- No performance benchmarks yet

**To Be Addressed in Feature-2:**
- Implement pytest test suite
- Add integration tests
- Create performance benchmarks

---

## Roadmap

### Phase 1: Foundation (Current)
- ✅ Environment setup
- ✅ Project structure
- ✅ Basic detector module
- ⏳ SORT tracker module

### Phase 2: Integration
- ⏳ Main integration pipeline
- ⏳ CLI interface
- ⏳ Visualization
- ⏳ Comprehensive testing

### Phase 3: Validation
- ⏳ MOT16 evaluation
- ⏳ Performance benchmarking
- ⏳ Demo videos
- ⏳ Documentation completion

### Phase 4: Extensions (Optional)
- Deep SORT implementation
- YOLOv8 migration
- Multi-class support
- Performance optimization

---

## References

- [SORT Paper](https://arxiv.org/abs/1602.00763)
- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
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
**Version:** 1.0 (feature-1 complete)
