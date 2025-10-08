# Project Architecture Document (PAD)
# Real-Time Object Tracking System

**Version:** 1.0  
**Last Updated:** 2025-01-27  
**Status:** Active Development

---

## 1. Project Overview

### 1.1 Purpose
Build a real-time object tracking system that detects and tracks people and cars in video streams using YOLOv5s for detection and SORT for tracking.

### 1.2 Goals
- Achieve >30 FPS on GPU hardware
- Maintain tracking accuracy with MOTA >0.80
- Provide modular, maintainable codebase
- Support live video streams and recorded files

### 1.3 Target Users
- Computer vision researchers and students
- Autonomous vehicle developers
- Security and surveillance system developers

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│  - Video I/O (file, webcam, RTSP)                       │
│  - Pipeline orchestration                               │
│  - Visualization & metrics logging                      │
│  - CLI interface                                        │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
       ┌───────────────┐      ┌──────────────┐
       │ detector.py   │      │ tracker.py   │
       │  - YOLOv5s    │      │  - SORT      │
       │  - Detection  │      │  - Kalman    │
       │  - Filtering  │      │  - IOU Match │
       └───────────────┘      └──────────────┘
               │                      │
               ▼                      ▼
         External Libraries:
         - ultralytics, torch, opencv-python
         - filterpy, numpy, scipy, motmetrics
```

### 2.2 Module Descriptions

#### detector.py
- **Purpose:** Object detection using pre-trained YOLOv5s
- **Input:** RGB frame (H×W×3 numpy array)
- **Output:** List[Detection(bbox, label, confidence)]
- **Processing:** YOLOv5s inference @ 640×480
- **Key Classes:**
  - `Detection`: Data class for single detection
  - `ObjectDetector`: Main detector with detect() method

#### tracker.py
- **Purpose:** Multi-object tracking with SORT algorithm
- **Input:** List[Detection] from detector
- **Output:** List[Track(bbox, id, label, confidence)]
- **Processing:** Kalman filter + IOU matching
- **Key Classes:**
  - `Track`: Tracked object with unique ID
  - `KalmanBoxTracker`: State estimation
  - `Tracker`: Main tracker with update() method

#### main.py
- **Purpose:** End-to-end pipeline integration
- **Input:** Video path or camera index
- **Output:** Annotated video + metrics log
- **Processing:** Frame loop with detect→track→visualize
- **Key Functions:**
  - Video I/O management
  - Real-time visualization
  - Performance logging
  - CLI argument parsing

#### evaluation.py (Optional)
- **Purpose:** Quantitative evaluation
- **Input:** Predictions + ground truth annotations
- **Output:** MOTA, MOTP, ID switches, detailed report
- **Processing:** motmetrics computation

---

## 3. Data Flow

### 3.1 Processing Pipeline

```
Input Video Stream
      │
      ▼
[Frame Capture] ─────────────────┐
      │                          │
      ▼                          │
[Detector.detect()]              │
  • YOLOv5s inference            │
  • Filter by confidence         │
  • Filter by class              │
      │                          │
      ▼                          │
List[Detection]                  │
      │                          │
      ▼                          │
[Tracker.update()]               │
  • Predict with Kalman          │
  • Match detections to tracks   │
  • Update/create/delete tracks  │
      │                          │
      ▼                          │
List[Track]                      │
      │                          │
      ▼                          │
[Visualization]                  │
  • Draw bounding boxes          │
  • Show IDs and labels          │
  • Display FPS                  │
      │                          │
      ▼                          │
[Output] ────────────────────────┘
  • Display window (optional)
  • Video file (optional)
  • Metrics log
```

### 3.2 Data Structures

**Detection:**
```python
class Detection:
    bbox: List[float]      # [x1, y1, x2, y2]
    class_label: str       # "person" or "car"
    confidence: float      # [0, 1]
```

**Track:**
```python
class Track:
    id: int                # Unique track ID
    bbox: List[float]      # [x1, y1, x2, y2]
    class_label: str       # "person" or "car"
    confidence: float      # [0, 1]
    age: int               # Frames since creation
    hits: int              # Consecutive detections
```

---

## 4. Technology Stack

### 4.1 Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.10 | Runtime environment |
| torch | Latest | GPU acceleration |
| ultralytics | Latest | YOLOv5s model |
| opencv-python | ≥4.5 | Video I/O, visualization |
| numpy | ≥1.21 | Array operations |
| filterpy | ≥1.4 | Kalman filter |
| scipy | Latest | IOU computation |

### 4.2 Development Tools
| Tool | Purpose |
|------|---------|
| conda | Environment management |
| pytest | Unit testing |
| pytest-cov | Coverage reporting |
| git | Version control |

### 4.3 Hardware Requirements

**Minimum:**
- CPU: Intel Core i5 or equivalent
- RAM: 8GB
- Storage: 2GB for models and dependencies
- OS: Windows 10/11, Ubuntu 18.04+, macOS 10.15+

**Recommended:**
- GPU: NVIDIA RTX 3050 or better with CUDA support
- RAM: 16GB
- Storage: 5GB for datasets and outputs

---

## 5. Design Decisions

### 5.1 Model Selection: YOLOv5s
**Decision:** Use YOLOv5s pre-trained on COCO dataset

**Rationale:**
- Good balance between speed (>30 FPS) and accuracy
- Pre-trained on COCO includes person and car classes
- Well-supported by Ultralytics library
- Easy to upgrade to YOLOv8 later if needed

**Alternatives Considered:**
- YOLOv5n (faster but less accurate)
- YOLOv8 (better accuracy but newer, less stable)
- Faster R-CNN (too slow for real-time)

### 5.2 Tracking Algorithm: SORT
**Decision:** Implement SORT (Simple Online and Realtime Tracking)

**Rationale:**
- Lightweight and fast (meets >30 FPS requirement)
- Well-documented algorithm
- Good baseline for future improvements
- Uses Kalman filter for motion prediction

**Alternatives Considered:**
- Deep SORT (better but requires appearance features)
- ByteTrack (newer, more complex)
- IOU tracker only (simpler but less robust)

### 5.3 Resolution: 640×480
**Decision:** Standard processing resolution of 640×480

**Rationale:**
- Balances detection accuracy and speed
- Supported natively by YOLOv5s
- Consistent performance across different videos
- Can be adjusted via configuration if needed

### 5.4 Target Classes: Person and Car Only
**Decision:** Filter for only "person" and "car" classes

**Rationale:**
- Reduces false positives
- Focuses on primary use cases (traffic, pedestrians)
- Simplifies visualization and evaluation
- Easy to extend to other classes later

### 5.5 Modular Architecture
**Decision:** Separate detector, tracker, and main modules

**Rationale:**
- Clear separation of concerns
- Easier testing and debugging
- Modules can be upgraded independently
- Supports code reuse

---

## 6. Performance Targets

### 6.1 Speed Metrics
| Metric | Target | Hardware |
|--------|--------|----------|
| FPS (GPU) | ≥30 | NVIDIA RTX 3050 |
| FPS (CPU) | ≥15 | Intel Core i5 |
| Latency | <33ms | Per frame |

### 6.2 Accuracy Metrics
| Metric | Target | Dataset |
|--------|--------|---------|
| MOTA | >0.80 | MOT16 |
| MOTP | >0.75 | MOT16 |
| ID Switches | <5/min | MOT16 |
| Precision | >0.85 | Custom test set |
| Recall | >0.80 | Custom test set |

### 6.3 Code Quality Metrics
| Metric | Target |
|--------|--------|
| Code Coverage | >80% |
| Documentation | 100% of public APIs |
| PEP 8 Compliance | >95% |

---

## 7. Extension Points

### 7.1 Planned Extensions
1. **Deep SORT:** Add appearance features for better tracking
2. **YOLOv8:** Upgrade to newer YOLO version
3. **Multi-class:** Extend to all COCO classes
4. **Re-ID:** Cross-camera tracking

### 7.2 Configuration Points
- Confidence threshold (default: 0.25)
- IOU threshold for matching (default: 0.3)
- Max age for tracks (default: 1 frame)
- Min hits to confirm track (default: 3 frames)
- Input resolution (default: 640×480)

---

## 8. Constraints and Assumptions

### 8.1 Assumptions
- GPU available for production use (CPU for testing)
- Input videos are standard formats (MP4, AVI, MOV)
- Objects to track are people and cars from COCO dataset
- Frame rate of input video ≥15 FPS

### 8.2 Constraints
- Python 3.10+ required
- CUDA-capable GPU recommended
- Minimum 8GB RAM
- Internet connection for initial model download

### 8.3 Known Limitations
- ID switches during heavy occlusions
- Poor performance in extreme lighting conditions
- No re-identification after long occlusions
- Single camera view only (no multi-camera fusion)

---

## 9. Testing Strategy

### 9.1 Unit Tests
- Test each module independently
- Mock dependencies where appropriate
- Target >80% code coverage

### 9.2 Integration Tests
- Test full pipeline on sample videos
- Verify output format and metrics
- Test error handling and edge cases

### 9.3 Evaluation
- Quantitative: MOT16 dataset with motmetrics
- Qualitative: Visual inspection of demo videos
- Performance: FPS benchmarks on different hardware

---

## 10. Deployment and Usage

### 10.1 Setup Process
1. Create conda environment
2. Install dependencies via pip
3. Download YOLOv5s model (automatic on first run)
4. Run on test video to verify installation

### 10.2 Command-Line Interface
```bash
python main.py --input video.mp4 --output output.mp4 --display
python main.py --input 0  # Webcam
python main.py --input video.mp4 --confidence 0.5
```

### 10.3 Output Formats
- **Video:** MP4 with bounding boxes and IDs
- **Logs:** JSON/CSV with frame-by-frame metrics
- **Metrics:** Summary report with MOTA, FPS, etc.

---

## 11. Future Improvements

### 11.1 Short-Term (Phase 2)
- Implement Deep SORT for better tracking
- Add performance dashboard
- Support for additional COCO classes
- Optimize for edge devices (Jetson)

### 11.2 Long-Term (Phase 3+)
- Multi-camera tracking and fusion
- 3D trajectory estimation
- Action recognition
- Anomaly detection

---

## 12. References

1. SORT Paper: https://arxiv.org/abs/1602.00763
2. YOLOv5: https://github.com/ultralytics/yolov5
3. MOT Challenge: https://motchallenge.net/
4. motmetrics: https://github.com/cheind/py-motmetrics

---

**Document Maintainers:** Project Team  
**Review Cycle:** Updated with each feature iteration  
**Next Review:** After feature-1 completion
