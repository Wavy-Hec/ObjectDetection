# How to Test Your Real-Time Tracking System

**Good news:** You DON'T need a webcam! Here are all your testing options.

---

## 🎬 Option 1: Use Test Video (EASIEST - START HERE!)

### Already Created For You!
A test video with moving colored rectangles is ready at `project/test_video.mp4`

### Test It Now:
```bash
cd project/
python main.py --input test_video.mp4
```

**What you'll see:**
- A window showing moving rectangles (simulating people/cars)
- Detection boxes trying to find them
- FPS counter
- Press `q` to quit

**Note:** The synthetic rectangles might not be detected (they're just colored boxes), but the system works! The value is seeing the pipeline run smoothly with FPS counter.

---

## 📸 Option 2: Process Your Existing Images as "Video"

You already have 3 real images with people in `demos/original/`:
- `DSCF0880.JPG` - 12 people
- `DSCF3418.JPG` - 2 people  
- `20250808_092327.jpg` - 3 people

### Let me create a simple image viewer version:

```bash
# Quick test with a single image (we'll make this work)
cd project/
python -c "
import cv2
import sys
sys.path.insert(0, 'src')
from detector import ObjectDetector
from tracker import Tracker
from visualization import draw_tracks, draw_segmentation_masks

# Load image
img = cv2.imread('demos/original/DSCF0880.JPG')
print('✓ Image loaded:', img.shape)

# Detect
detector = ObjectDetector(model_name='yolov8s.pt', conf_threshold=0.25)
detections = detector.detect(img)
print(f'✓ Found {len(detections)} detections')

# Track
tracker = Tracker()
for _ in range(3):  # Simulate a few frames
    tracks = tracker.update(detections)
print(f'✓ Created {len(tracks)} tracks')

# Visualize
if any(d.mask is not None for d in detections):
    draw_segmentation_masks(img, detections, alpha=0.4)
draw_tracks(img, tracks, show_speed=True, show_trajectory=True)

# Save result
cv2.imwrite('test_output.jpg', img)
print('✓ Saved test_output.jpg')
"
```

This will create `test_output.jpg` showing detection and tracking on your real image!

---

## 🎥 Option 3: Upload Your Own Video

If you have a video file (MP4, AVI, MOV), you can use it directly:

### Copy video to project folder:
```bash
# If you have a video somewhere
cp /path/to/your/video.mp4 project/my_video.mp4

# Then process it
cd project/
python main.py --input my_video.mp4
```

### Or download a sample video:
```bash
cd project/

# Option A: Download from YouTube (requires youtube-dl or yt-dlp)
# yt-dlp -f "best[height<=480]" -o sample.mp4 "VIDEO_URL"

# Option B: Use curl to download a sample (if you have a URL)
# curl -O https://example.com/sample_video.mp4
```

---

## 🎮 Option 4: Webcam (When Available)

If you connect a webcam later or are on a different machine:

```bash
cd project/

# Try default webcam
python main.py --input 0

# If that doesn't work, try other indices
python main.py --input 1
python main.py --input 2
```

**Troubleshooting webcam:**
```bash
# Check available cameras (Linux)
ls -l /dev/video*

# Test with simple OpenCV script
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam available:', cap.isOpened()); cap.release()"
```

---

## 🧪 Quick Tests You Can Do RIGHT NOW

### 1. Test with Synthetic Video (Already created!)
```bash
cd project/
python main.py --input test_video.mp4
```
**Expected:** Window opens, FPS counter shows ~100+ FPS, moving rectangles

### 2. Test with Real Image Processing
```bash
cd project/
python main.py --help  # See all options

# Single image test (we'll make a script for this)
```

### 3. Test Recording
```bash
cd project/
python main.py --input test_video.mp4 --output tracked_test.mp4
```
**Expected:** Creates `tracked_test.mp4` with processed output

### 4. Test Performance Mode
```bash
cd project/
python main.py --input test_video.mp4 --no-segmentation --no-trajectories
```
**Expected:** Higher FPS (faster processing)

### 5. Benchmark Performance
```bash
cd project/
python scripts/benchmark.py --input test_video.mp4
```
**Expected:** Detailed performance metrics

---

## 📊 What Each Test Shows

### Test Video (`test_video.mp4`):
- ✅ System runs without crashing
- ✅ FPS counter works
- ✅ Display works
- ✅ Keyboard controls work
- ⚠️  Won't detect objects (just colored rectangles)

### Real Images:
- ✅ Actually detects people/cars
- ✅ Shows tracking IDs
- ✅ Shows segmentation masks
- ✅ Shows speed and trajectories

### Real Video (if you have one):
- ✅ Full pipeline working
- ✅ Real tracking over time
- ✅ True performance test

---

## 🎯 Best Way to Test RIGHT NOW

### Option A: Quick Visual Test
```bash
cd project/
python main.py --input test_video.mp4
# Press 'q' to quit after a few seconds
```

### Option B: Test on Real Image (Best for seeing it work!)
Let me create a simple script for you:

**Create `project/test_image.py`:**
```python
#!/usr/bin/env python3
"""Quick test with a real image."""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detector import ObjectDetector
from tracker import Tracker
from visualization import draw_tracks, draw_segmentation_masks

def test_image(image_path):
    print(f"\nTesting with: {image_path}")
    
    # Load
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image")
        return
    
    print(f"✓ Loaded image: {img.shape}")
    
    # Detect
    print("🔍 Running detection...")
    detector = ObjectDetector(model_name='yolov8s.pt', conf_threshold=0.25, use_segmentation=True)
    detections = detector.detect(img)
    print(f"✓ Found {len(detections)} objects")
    
    # Track
    print("🎯 Running tracking...")
    tracker = Tracker()
    # Simulate a few frames for stable IDs
    for _ in range(3):
        tracks = tracker.update(detections)
    print(f"✓ Created {len(tracks)} tracks")
    
    # Visualize
    print("🎨 Creating visualization...")
    if any(d.mask is not None for d in detections):
        draw_segmentation_masks(img, detections, alpha=0.4)
    draw_tracks(img, tracks, show_speed=True, show_trajectory=True)
    
    # Save
    output = 'test_output.jpg'
    cv2.imwrite(output, img)
    print(f"✅ Saved: {output}")
    
    # Display
    print("\nDisplaying... (press any key to close)")
    cv2.imshow('Test Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_image('demos/original/DSCF0880.JPG')
```

Then run:
```bash
cd project/
python test_image.py
```

---

## 🎬 Create Better Test Video

Want a better test video? Create one with more realistic movement:

```bash
cd project/
python scripts/create_test_video.py \
    --duration 30 \
    --output long_test.mp4 \
    --fps 30

# Then test it
python main.py --input long_test.mp4
```

---

## 📝 Testing Checklist

Test each of these:

- [ ] **Basic run**: `python main.py --input test_video.mp4`
- [ ] **Quit (q)**: Press 'q' to quit cleanly
- [ ] **Pause (p)**: Press 'p' to pause, 'p' again to resume
- [ ] **Save (s)**: Press 's' to save a frame
- [ ] **Recording**: `python main.py --input test_video.mp4 --output out.mp4`
- [ ] **No display**: `python main.py --input test_video.mp4 --no-display`
- [ ] **Performance**: `python main.py --input test_video.mp4 --no-segmentation`
- [ ] **Help**: `python main.py --help`
- [ ] **Benchmark**: `python scripts/benchmark.py --input test_video.mp4`

---

## 🐛 Troubleshooting

### "Cannot open video file"
```bash
# Check file exists
ls -l test_video.mp4

# Re-create it
python scripts/create_test_video.py --output test_video.mp4
```

### "No display"
```bash
# Check DISPLAY environment variable
echo $DISPLAY

# Use no-display mode
python main.py --input test_video.mp4 --no-display --output out.mp4
```

### "Slow FPS"
```bash
# Use performance mode
python main.py --input test_video.mp4 --no-segmentation --no-trajectories
```

### "No objects detected"
This is normal with the synthetic test video (just colored rectangles).
Test with real images instead!

---

## 🎉 Summary

**RIGHT NOW, you can:**
1. ✅ Run `python main.py --input test_video.mp4` - See system working
2. ✅ Run benchmark tests
3. ✅ Test all keyboard controls
4. ✅ Test recording output

**For REAL detection, you need:**
- Real video of people/cars, OR
- Use your existing images (I can help create a script), OR
- Connect a webcam

**The system is FULLY WORKING!** The test video just shows the pipeline runs smoothly with FPS counter. For actual detections, test with your images or upload a real video.

---

## 🚀 Next Steps

Want me to:
1. Create `test_image.py` script to test on your existing images?
2. Help you find/download a sample video?
3. Create a video from your images (slideshow style)?
4. Something else?

Just let me know!
