# Feature-3: Real-Time Webcam Usage Guide

## 🎥 You're Ready to Go!

Your real-time tracking system is implemented and ready to use!

---

## Quick Start

### 1. Webcam (Live Camera)

```bash
cd project/
python main.py --input 0
```

**Controls:**
- `q` or `ESC` - Quit
- `p` - Pause/Resume
- `s` - Save current frame

### 2. Video File

```bash
# Process existing video
python main.py --input demos/original/DSCF0880.JPG  # Wait, that's an image!

# First, create a test video
python scripts/create_test_video.py --output test.mp4

# Then process it
python main.py --input test.mp4
```

### 3. Record Output

```bash
# Record webcam to file
python main.py --input 0 --output webcam_tracked.mp4

# Process video and save
python main.py --input test.mp4 --output test_tracked.mp4
```

---

## Command-Line Options

### Input/Output
- `--input`, `-i` - **REQUIRED** - Input source
  - `0` for default webcam
  - `1` for second camera
  - `path/to/video.mp4` for video file

- `--output`, `-o` - Output file path (optional)
  - Saves processed video with all tracking

### Detection Settings
- `--confidence` - Detection threshold (default: 0.25)
  - Lower = more detections (0.1-0.5 typical)
  - Higher = fewer but more confident

- `--no-segmentation` - Disable instance segmentation
  - Faster performance, no colored masks

### Tracking Settings
- `--max-age` - Frames to keep lost tracks (default: 1)
- `--min-hits` - Hits needed to confirm track (default: 3)
- `--iou-threshold` - Matching threshold (default: 0.3)

### Visualization
- `--no-trajectories` - Disable trajectory lines
- `--no-speed` - Disable speed overlay

### Display
- `--no-display` - Headless mode (no window)
  - Useful for batch processing

---

## Examples

### Basic Webcam
```bash
python main.py --input 0
```

### High Confidence (Less Noise)
```bash
python main.py --input 0 --confidence 0.5
```

### Performance Mode (No Segmentation)
```bash
python main.py --input 0 --no-segmentation
```

### Minimal Visualization
```bash
python main.py --input 0 --no-segmentation --no-trajectories --no-speed
```

### Record Webcam Session
```bash
python main.py --input 0 --output my_tracking_demo.mp4
```

### Batch Process Video (No Display)
```bash
python main.py --input video.mp4 --output tracked.mp4 --no-display
```

---

## What You'll See

### On Screen Display:
```
┌─────────────────────────────────────┐
│ FPS: 28.5                          │  ← Green, top-left
│ Detections: 3                      │  ← White
│ Tracks: 2                          │  ← White
│                                    │
│  [Tracked objects with:]           │
│   • Bounding boxes                 │
│   • Track IDs                      │
│   • Segmentation masks (colored)   │
│   • Speed (px/s)                   │
│   • Trajectory trails              │
└─────────────────────────────────────┘
```

### Console Output:
```
==========================================================
  REAL-TIME OBJECT TRACKING
==========================================================

[1/4] Initializing video source...
✓ Webcam 0 opened: 640x480 @ 30 FPS

[2/4] Initializing detector...
ObjectDetector initialized on cuda
Model: yolov8s.pt, Confidence threshold: 0.25

[3/4] Initializing tracker...
Tracker initialized

[4/4] Setting up display and recording...

==========================================================
  KEYBOARD CONTROLS
==========================================================
  q / ESC  - Quit
  p        - Pause/Resume
  s        - Save current frame
==========================================================

Processing... Press 'q' to quit
```

---

## Troubleshooting

### "Cannot open webcam 0"
**Solutions:**
- Check camera is connected
- Try different index: `--input 1` or `--input 2`
- Close other apps using camera (Zoom, Skype, etc.)
- On Linux: Check permissions `ls -l /dev/video*`

### "Video file not found"
**Solutions:**
- Check file path is correct
- Use absolute path: `/full/path/to/video.mp4`
- Verify file exists: `ls -l video.mp4`

### Low FPS / Slow Performance
**Solutions:**
- Disable segmentation: `--no-segmentation`
- Increase confidence: `--confidence 0.5`
- Check GPU is being used (should see "cuda" in output)
- Lower resolution (webcam settings in code)

### No Objects Detected
**Solutions:**
- Lower confidence: `--confidence 0.1`
- Make sure you have people or cars in view
- Check lighting (better with good light)
- Try with test images first

---

## Performance Tips

### For Best FPS:
1. Use GPU (automatic if available)
2. Disable segmentation: `--no-segmentation`
3. Disable trajectories: `--no-trajectories`
4. Higher confidence: `--confidence 0.5`

### For Best Accuracy:
1. Lower confidence: `--confidence 0.25` or less
2. Enable all features (default)
3. Good lighting
4. Clear view of objects

---

## What Works Now

✅ **Webcam streaming** - Live camera feed  
✅ **Video file processing** - MP4, AVI, MOV  
✅ **Real-time detection** - YOLOv8s (people & cars)  
✅ **Object tracking** - SORT algorithm with persistent IDs  
✅ **Instance segmentation** - Colored masks  
✅ **Speed estimation** - From Kalman filter  
✅ **Trajectory visualization** - Path history  
✅ **FPS monitoring** - Real-time performance  
✅ **Interactive controls** - Pause, save, quit  
✅ **Video recording** - Save processed output

---

## Next Steps

### Try It Out:
1. **Test with webcam:**
   ```bash
   python main.py --input 0
   ```

2. **Create and process test video:**
   ```bash
   python scripts/create_test_video.py --output test.mp4
   python main.py --input test.mp4
   ```

3. **Record a demo:**
   ```bash
   python main.py --input 0 --output my_demo.mp4
   ```

### Performance Testing:
```bash
# Create longer test video
python scripts/create_test_video.py --duration 30 --output test_long.mp4

# Benchmark it
python scripts/benchmark.py --input test_long.mp4
```

---

## Code Structure

```
main.py                  # Main application
src/
  video_source.py        # NEW - Webcam/file abstraction
  detector.py            # Existing - YOLOv8 detection
  tracker.py             # Existing - SORT tracking
  visualization.py       # Existing - Drawing functions
```

---

## Have Fun! 🎉

You now have a fully functional real-time object tracking system!

**Try these:**
- Wave at the webcam and watch yourself get tracked
- Walk around and see the trajectory trail
- Count how many people/cars it detects
- Record a demo for your portfolio
- Test on different videos

**Questions?**
- Check `main.py --help` for all options
- Review `feature-3/design.md` for architecture
- See `feature-3/plan.md` for implementation details

---

**Status:** Feature-3 Core Implementation Complete! ✅
