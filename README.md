# Real-Time Multi-Object Detection & Tracking

Detects and tracks **all objects** in real time using YOLOv8 + SORT (Kalman filter tracking). Works with webcam or video files. Tracks people, phones, laptops, cars, bottles — anything in the 80 COCO classes.

## What It Does

- **Detects** objects in each frame using YOLOv8s on GPU (FP16 half precision)
- **Tracks** them across frames with unique IDs using a Kalman filter + Hungarian matching
- **Class-aware tracking** — a phone won't get confused with a person
- **Displays** bounding boxes, class labels, track IDs, speed, and trajectory trails
- **Records** output video (optional)

## Setup

```bash
# Create the conda environment
conda create -n ml python=3.10 -y
conda activate ml

# Install PyTorch (adjust cu128 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install ultralytics opencv-python scipy filterpy lap pyyaml matplotlib psutil
```

## Usage

```bash
conda activate ml

# Webcam (detect everything)
python main.py --input 0

# Webcam (detect only people and phones)
python main.py --input 0 --classes person "cell phone"

# Video file
python main.py --input video.mp4

# Record output
python main.py --input 0 --output tracked.mp4

# Enable segmentation masks
python main.py --input 0 --segmentation

# Adjust confidence threshold
python main.py --input 0 --confidence 0.4
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `p` | Pause / Resume |
| `s` | Save current frame |

## Project Structure

```
main.py              # Entry point — run this
config.yaml          # Default configuration
environment.yml      # Conda environment spec
src/
  detector.py        # YOLOv8 object detection (all 80 COCO classes)
  tracker.py         # SORT tracker with Kalman filters
  video_source.py    # Webcam and video file input
  visualization.py   # Drawing boxes, labels, trails, masks
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | required | `0` for webcam, or path to video file |
| `--output` / `-o` | none | Save output video to this path |
| `--confidence` | 0.25 | Detection confidence threshold |
| `--classes` | all | Space-separated class names to detect |
| `--max-age` | 30 | Frames to keep a track alive without detection |
| `--min-hits` | 3 | Minimum detections before track is confirmed |
| `--iou-threshold` | 0.3 | IOU threshold for matching |
| `--segmentation` | off | Enable segmentation mask overlay |
| `--no-trajectories` | off | Disable trajectory trails |
| `--no-speed` | off | Disable speed display |
| `--no-display` | off | Headless mode (no window) |
