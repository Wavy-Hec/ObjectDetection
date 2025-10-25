# Validation Phase Command

**Purpose:** Validate that feature meets all requirements  
**Input:** Implemented code and requirements  
**Output:** Validation report and metrics  
**When to use:** After all tasks complete, before feature sign-off

---

## Prompt for AI Assistant

```
You are a QA engineer validating the Real-Time Object Tracking project.

CONTEXT:
- Review PAD.md for quality standards
- Review feature-[X]/PRD.md for requirements
- Review feature-[X]/design.md for technical specifications
- Review all implemented code in src/

TASK:
Validate that the feature is complete and meets all requirements:

1. **Functional Validation**:
   - Verify all requirements from PRD are met
   - Test all specified use cases
   - Check edge cases and error conditions
   - Validate inputs and outputs
   
2. **Performance Validation**:
   - Measure FPS on GPU and CPU
   - Verify meets performance targets
   - Check memory usage
   - Test with different video sizes
   
3. **Quality Validation**:
   - Run all unit tests
   - Check code coverage (> 80%)
   - Verify code style (PEP 8)
   - Review docstring coverage
   
4. **Integration Validation**:
   - Test end-to-end pipeline
   - Verify output formats
   - Check error handling
   - Test with real-world data
   
5. **Documentation Validation**:
   - Verify README is complete
   - Check all functions documented
   - Validate examples work
   - Review setup instructions

REQUIREMENTS:
- All acceptance criteria must be met
- All tests must pass
- Performance targets must be achieved
- Documentation must be complete

OUTPUT:
- Validation report with pass/fail for each category
- Performance metrics and benchmarks
- List of any issues found
- Recommendation: Ready for release or needs fixes

Please validate the feature now.
```

---

## Validation Checklist

### Functional Requirements
Run through each requirement from PRD.md:

- [ ] REQ-DET-001: YOLOv5s detector implemented
- [ ] REQ-DET-002: Filters for person and car only
- [ ] REQ-DET-003: Configurable confidence threshold
- [ ] REQ-TRK-001: SORT algorithm implemented
- [ ] REQ-TRK-002: Kalman filter for prediction
- [ ] REQ-TRK-003: IOU matching for associations
- [ ] REQ-IO-001: Accepts video files and webcam
- [ ] REQ-IO-002: Outputs annotated video and metrics
- [ ] REQ-VIS-001: Draws bounding boxes
- [ ] REQ-VIS-002: Displays track IDs
- [ ] REQ-VIS-003: Shows FPS metrics

### Performance Requirements
Measure and verify:

```bash
# Test on GPU
python main.py --input test_video.mp4 --output output.mp4
# Record FPS: _____ (Target: ≥30)

# Test on CPU  
CUDA_VISIBLE_DEVICES="" python main.py --input test_video.mp4
# Record FPS: _____ (Target: ≥15)

# Test tracking metrics
python evaluation.py --predictions output.txt --ground_truth gt.txt
# MOTA: _____ (Target: > 0.80)
# ID Switches: _____ (Target: < 5/min)
```

- [ ] GPU FPS ≥30
- [ ] CPU FPS ≥15
- [ ] MOTA > 0.80
- [ ] ID switches < 5/min
- [ ] Memory usage reasonable

### Quality Requirements

```bash
# Run all tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term --cov-report=html
# Target: > 80%

# Check code style
flake8 src/ --max-line-length=100

# Check docstrings
pydocstyle src/
```

- [ ] All tests pass
- [ ] Coverage > 80%
- [ ] Code follows PEP 8
- [ ] All functions documented
- [ ] No critical linting errors

### Integration Requirements

```bash
# Test with different inputs
python main.py --input sample1.mp4 --output out1.mp4
python main.py --input sample2.mp4 --output out2.mp4  
python main.py --input 0  # Webcam

# Test with different parameters
python main.py --input video.mp4 --confidence 0.5
python main.py --input video.mp4 --no-display

# Test error handling
python main.py --input nonexistent.mp4  # Should fail gracefully
python main.py --input corrupted.mp4    # Should handle errors
```

- [ ] Works with multiple videos
- [ ] Works with webcam
- [ ] CLI parameters work correctly
- [ ] Error messages are helpful
- [ ] Handles edge cases gracefully

### Documentation Requirements

```bash
# Verify README
cat README.md
# Check: installation, usage, examples, troubleshooting

# Verify docstrings
python -c "import src.detector; help(src.detector.ObjectDetector)"
python -c "import src.tracker; help(src.tracker.Tracker)"

# Test examples from README
# Copy commands from README and verify they work
```

- [ ] README is complete and accurate
- [ ] Installation instructions work
- [ ] Usage examples work
- [ ] All modules documented
- [ ] Troubleshooting section helpful

---

## Performance Benchmarking

### FPS Measurement
```python
import time
import cv2

cap = cv2.VideoCapture('test_video.mp4')
fps_list = []

start = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Your processing here
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - start
        fps = 30 / elapsed
        fps_list.append(fps)
        start = time.time()

avg_fps = sum(fps_list) / len(fps_list)
print(f"Average FPS: {avg_fps:.2f}")
```

### Memory Profiling
```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler main.py --input video.mp4
```

### Tracking Metrics
```python
import motmetrics as mm

# Load ground truth and predictions
gt = load_ground_truth('annotations.txt')
predictions = load_predictions('output.txt')

# Compute metrics
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['mota', 'motp', 'num_switches'])

print(f"MOTA: {summary['mota'][0]:.3f}")
print(f"MOTP: {summary['motp'][0]:.3f}")
print(f"ID Switches: {summary['num_switches'][0]}")
```

---

## Validation Report Template

```markdown
# Feature Validation Report

**Feature:** feature-[X]  
**Date:** [Date]  
**Validator:** [Name]  
**Status:** [PASS | FAIL | NEEDS WORK]

## Functional Validation
- Detector: [PASS/FAIL]
- Tracker: [PASS/FAIL]
- Visualization: [PASS/FAIL]
- I/O: [PASS/FAIL]

## Performance Validation
- GPU FPS: [value] (Target: ≥30) [PASS/FAIL]
- CPU FPS: [value] (Target: ≥15) [PASS/FAIL]
- MOTA: [value] (Target: > 0.80) [PASS/FAIL]
- ID Switches: [value] (Target: < 5/min) [PASS/FAIL]

## Quality Validation
- Test Pass Rate: [X/Y] tests passed
- Code Coverage: [X]% (Target: > 80%) [PASS/FAIL]
- Style Compliance: [PASS/FAIL]
- Documentation: [PASS/FAIL]

## Issues Found
1. [Issue 1]: [Description] - [Severity: High/Med/Low]
2. [Issue 2]: [Description] - [Severity: High/Med/Low]

## Recommendation
[Ready for release | Needs fixes before release]

## Next Steps
- [Action item 1]
- [Action item 2]
```

---

## Common Issues and Fixes

### Issue: Tests pass but real-world performance poor
**Diagnosis:** Tests use synthetic data  
**Fix:** Add integration tests with real videos

### Issue: FPS below target
**Diagnosis:** Bottleneck in processing  
**Fix:** Profile code, optimize hot paths, reduce resolution

### Issue: High ID switches
**Diagnosis:** IOU threshold or Kalman tuning  
**Fix:** Experiment with parameters, consider Deep SORT

### Issue: Memory leak over time
**Diagnosis:** Not releasing resources  
**Fix:** Check object lifecycle, add cleanup

---

## Example Usage

```bash
# Step 1: Run comprehensive tests
pytest tests/ -v --cov=src --cov-report=html

# Step 2: Run performance benchmarks
python benchmark.py --input test_videos/*.mp4

# Step 3: Test integration
./test_integration.sh

# Step 4: Generate validation report
python generate_validation_report.py > validation_report.md

# Step 5: Review and decide
cat validation_report.md
# If all pass: Ready for release
# If issues found: Create GitHub issues and fix
```

---

## Integration with Workflow

```
Implementation Complete
    ↓
[USE THIS COMMAND]
    ↓
Validation Report
    ↓
Issues Found? ──Yes──> Fix Issues
    │                      │
    No                     │
    ↓                      │
Feature Complete <─────────┘
    ↓
Post-Development Analysis
```

---

**Last Updated:** 2025-01-27  
**Version:** 1.0
