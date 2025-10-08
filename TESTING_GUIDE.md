# Testing and Validation Guide

**Purpose:** How to test your code and validate it works correctly  
**Last Updated:** 2025-01-27

---

## Quick Start - Test Everything Now

```bash
# 1. Activate environment
conda activate object_tracking

# 2. Install testing tools
pip install pytest pytest-cov

# 3. Run all tests
pytest tests/ -v

# 4. Check code coverage
pytest tests/ --cov=src --cov-report=term --cov-report=html

# 5. Test the detector directly
python src/detector.py
```

---

## Step-by-Step Testing Guide

### Step 1: Activate Your Environment

```bash
# Make sure you're in the right environment
conda activate object_tracking

# Verify Python version
python --version  # Should be 3.10.x

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

**Expected Output:**
```
Python 3.10.x
PyTorch: 2.x.x
Ultralytics: OK
OpenCV: 4.x.x
```

### Step 2: Install Testing Tools

```bash
# Install pytest and coverage tools
pip install pytest pytest-cov

# Verify installation
pytest --version
```

**Expected Output:**
```
pytest 7.x.x
```

### Step 3: Test the Detector Module Directly

```bash
# Run the detector's built-in test
python src/detector.py
```

**Expected Output:**
```
ObjectDetector initialized on cuda  # or 'cpu'
Model: yolov5s.pt, Confidence threshold: 0.25
Target classes: {'person', 'car'}

Downloading yolov5s.pt...  # (first time only)
...
YOLOv5s model loaded successfully

Test detection completed.
Number of detections: 0
Detector module test completed successfully!
```

**What This Tests:**
- Environment is set up correctly
- PyTorch and CUDA work (if GPU available)
- YOLOv5s model downloads and loads
- Basic detection function works

### Step 4: Run Unit Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run just fast tests (skip slow ones)
pytest tests/ -v -m "not slow"

# Run with detailed output
pytest tests/ -v -s
```

**Expected Output:**
```
================================ test session starts =================================
collected 10 items

tests/test_detector.py::TestDetection::test_detection_creation PASSED         [ 10%]
tests/test_detector.py::TestDetection::test_detection_repr PASSED             [ 20%]
tests/test_detector.py::TestObjectDetector::test_detector_initialization PASSED [ 30%]
tests/test_detector.py::TestObjectDetector::test_detect_with_blank_frame PASSED [ 40%]
tests/test_detector.py::TestObjectDetector::test_detect_returns_list PASSED   [ 50%]
tests/test_detector.py::TestObjectDetector::test_detect_with_random_frame PASSED [ 60%]
tests/test_detector.py::TestObjectDetector::test_target_classes PASSED        [ 70%]
tests/test_detector.py::TestDetectorConfiguration::test_custom_confidence_threshold PASSED [ 80%]
tests/test_detector.py::TestDetectorConfiguration::test_custom_image_size PASSED [ 90%]
tests/test_detector.py::TestDetectorIntegration::test_full_detection_pipeline PASSED [100%]

================================ 10 passed in X.XXs ==================================
```

**What This Tests:**
- Detection class works correctly
- Detector initializes properly
- Detection function returns correct format
- Configuration options work
- Edge cases are handled

### Step 5: Check Code Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term --cov-report=html

# View coverage in terminal
# (automatically shown after command above)

# Open HTML coverage report in browser
firefox htmlcov/index.html  # or your browser
```

**Expected Output:**
```
---------- coverage: platform linux, python 3.10.x -----------
Name                Stmts   Miss  Cover
---------------------------------------
src/__init__.py         0      0   100%
src/detector.py        65      8    88%
---------------------------------------
TOTAL                  65      8    88%
```

**Coverage Goals:**
- ✅ Good: >80% coverage
- ⚠️ Warning: 60-80% coverage
- ❌ Poor: <60% coverage

---

## What Each Test Does

### Basic Tests (Fast)

1. **test_detection_creation**
   - Creates a Detection object
   - Verifies attributes are set correctly
   - Tests: Data structure works

2. **test_detection_repr**
   - Tests string representation
   - Verifies output format
   - Tests: Debugging output is useful

3. **test_detector_initialization**
   - Creates a detector
   - Checks all attributes
   - Tests: Detector sets up correctly

4. **test_detect_with_blank_frame**
   - Runs detection on empty image
   - Should return no detections
   - Tests: Handles empty cases

5. **test_detect_returns_list**
   - Verifies output type
   - Tests: API contract maintained

6. **test_target_classes**
   - Checks class filtering
   - Tests: Only person/car detected

### Configuration Tests

7. **test_custom_confidence_threshold**
   - Tests different threshold values
   - Verifies configuration works

8. **test_custom_image_size**
   - Tests different resolutions
   - Verifies size configuration

### Integration Tests (Slower)

9. **test_full_detection_pipeline**
   - Tests complete workflow
   - End-to-end validation

---

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'torch'"

**Problem:** Environment not activated or packages not installed

**Solution:**
```bash
# Activate environment
conda activate object_tracking

# Reinstall if needed
conda env update -f environment.yml
```

### Issue 2: "ModuleNotFoundError: No module named 'pytest'"

**Problem:** pytest not installed

**Solution:**
```bash
pip install pytest pytest-cov
```

### Issue 3: Tests fail with CUDA errors

**Problem:** GPU not available or CUDA mismatch

**Solution:**
```bash
# Force CPU mode
CUDA_VISIBLE_DEVICES="" pytest tests/ -v

# Or update detector to handle gracefully (already done)
```

### Issue 4: YOLOv5 model download fails

**Problem:** Network issues or firewall

**Solution:**
```bash
# Download manually
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

# Place in cache directory
mkdir -p ~/.cache/torch/hub/ultralytics/
mv yolov5s.pt ~/.cache/torch/hub/ultralytics/
```

### Issue 5: Some tests are slow

**Problem:** Model loading takes time

**Solution:**
```bash
# Skip slow tests during development
pytest tests/ -v -m "not slow"

# Run full suite before commit
pytest tests/ -v
```

---

## Testing Best Practices

### During Development

```bash
# Run relevant tests frequently
pytest tests/test_detector.py::TestDetection -v

# Use watch mode (requires pytest-watch)
pip install pytest-watch
ptw tests/ -- -v
```

### Before Committing

```bash
# Run full test suite
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term

# Verify no failures
# Fix any issues before committing
```

### Before Merging to Main

```bash
# Run complete validation
pytest tests/ -v --cov=src --cov-report=html

# Review coverage report
# Ensure >80% coverage

# Run linting (if configured)
flake8 src/ --max-line-length=100
```

---

## Manual Testing Scenarios

### Test 1: Basic Functionality

```bash
# Run detector standalone
python src/detector.py
```

**Expected:** No errors, model loads, test completes

### Test 2: Different Confidence Thresholds

```python
# In Python interpreter
from src.detector import ObjectDetector
import numpy as np

# High threshold
detector_high = ObjectDetector(conf_threshold=0.8)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
detections = detector_high.detect(frame)
print(f"High threshold: {len(detections)} detections")

# Low threshold
detector_low = ObjectDetector(conf_threshold=0.1)
detections = detector_low.detect(frame)
print(f"Low threshold: {len(detections)} detections")
```

### Test 3: GPU vs CPU Performance

```bash
# Test on GPU
python -c "from src.detector import ObjectDetector; d = ObjectDetector(); print(d.device)"

# Test on CPU
CUDA_VISIBLE_DEVICES="" python -c "from src.detector import ObjectDetector; d = ObjectDetector(); print(d.device)"
```

**Expected:** First shows 'cuda', second shows 'cpu'

---

## Next Steps After Testing

### If All Tests Pass ✅

1. **Commit the test files:**
   ```bash
   git add tests/ pytest.ini
   git commit -m "Add comprehensive test suite for detector module"
   git push origin main
   ```

2. **Update feature-1 analysis:**
   - Note that tests are now complete
   - Update metrics with coverage

3. **Ready for feature-2:**
   - Follow the workflow for SORT tracker
   - Write tests as you implement

### If Tests Fail ❌

1. **Read the error messages carefully**
   - Note which test failed
   - Read the assertion error

2. **Fix the issue:**
   - Update code in src/detector.py
   - Or fix the test if it's incorrect

3. **Re-run tests:**
   ```bash
   pytest tests/test_detector.py::test_that_failed -v
   ```

4. **Iterate until all pass**

---

## Continuous Integration (Future)

For feature-2, consider adding GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
      - run: conda env create -f environment.yml
      - run: conda activate object_tracking
      - run: pip install pytest pytest-cov
      - run: pytest tests/ -v --cov=src
```

---

## Summary: Your Testing Checklist

- [ ] Environment activated: `conda activate object_tracking`
- [ ] Testing tools installed: `pip install pytest pytest-cov`
- [ ] Direct test passes: `python src/detector.py`
- [ ] Unit tests pass: `pytest tests/ -v`
- [ ] Coverage >80%: `pytest tests/ --cov=src`
- [ ] All tests documented and understood
- [ ] Ready to move to feature-2

---

**Last Updated:** 2025-01-27  
**Next:** Start feature-2 with better testing from day one!
