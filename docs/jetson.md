# Running FlowCount on an NVIDIA Jetson

FlowCount runs on a Jetson Orin Nano (and NX / AGX). The tracker, analytics, and
the entire `flowcount/safety/` subsystem are pure aarch64-portable Python and
need no changes — `torch` is lazy-imported, so the dashboard and synthetic scene
even boot with **no ML stack installed**. The only real work is at the
install/packaging boundary, which is what this runbook covers.

**Target:** Orin Nano 8GB (Super), JetPack 6.2 (L4T r36.4, Ubuntu 22.04,
Python 3.10, CUDA 12.6, TensorRT 10.3). The legacy 4GB Nano is **not supported**
— it is stuck on JetPack 4.6 / Python 3.6 / CUDA 10.2, an EOL toolchain
incompatible with current ultralytics/torch.

---

## 0. Set the power mode (the #1 "why is my Jetson slow" mistake)

Worth 2–3× FPS. `Orin Nano Super` is a firmware/power unlock, not new silicon —
you still have to enable it:

```bash
sudo nvpmodel -m 0      # MAXN
sudo jetson_clocks       # lock clocks high
```

Boot headless (`sudo systemctl set-default multi-user.target`) to reclaim ~1 GB
of the shared CPU/GPU RAM, and add 4–8 GB of swap/zram (the TensorRT build step
is memory-hungry).

---

## 1. Get the ML stack — never `pip install torch`

PyPI has **no aarch64 + CUDA** torch wheel; a plain `pip install torch` gives you
a CPU-only build. Two supported paths:

### A. Container (recommended, shortest path)

```bash
docker build -f Dockerfile.jetson -t flowcount-jetson .
docker run --runtime nvidia --network host --restart unless-stopped flowcount-jetson
```

- The base image (`ultralytics/ultralytics:latest-jetson-jetpack6`) ships matched
  torch/torchvision + TensorRT and a GStreamer-enabled OpenCV.
- Use `--runtime nvidia`, **not** `--gpus all` (that is x86 Docker). Set nvidia as
  the default runtime in `/etc/docker/daemon.json` so it survives reboots.
- The image tag **must** match the board's L4T version: `cat /etc/nv_tegra_release`.

### B. Native install

```bash
sudo apt install libopenblas-dev libomp-dev
# NVIDIA aarch64 CUDA wheels, matched to JetPack 6 / CUDA 12.6:
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision
pip install ultralytics --no-deps       # --no-deps so it doesn't pull a 2nd CPU torch
pip install --no-deps .                  # install flowcount without deps
pip install "numpy<2" scipy filterpy pyyaml fastapi "uvicorn[standard]"
```

**Two pins that bite everyone:**

- **`numpy<2`** — NVIDIA's torch wheels are built against numpy 1.26 and throw ABI
  errors under numpy 2.x. Let torch dictate numpy.
- **Do not `pip install opencv-python`** — the aarch64 wheel has no GStreamer and
  silently shadows the JetPack CUDA/GStreamer `cv2`, so you lose NVDEC hardware
  decode with no error. Rely on the system/base `cv2`; verify with
  `python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer`
  → should say `GStreamer: YES`.

---

## 2. Export the model to TensorRT (FP16) — on the board

FP16 TensorRT is the sweet spot: ~3× the plain-PyTorch inference speed, roughly
**60–100 FPS full-pipeline** for `yolo11n` at 640 px on an Orin Nano Super in
MAXN. (Skip INT8 — it buys ~30% more for a calibration set you don't have and a
~3–4 mAP drop, the wrong trade for a safety system. Skip DeepStream — it means
rewriting the pipeline in GStreamer and abandoning the tracker/analytics/safety
code.)

```bash
flowcount-export --model yolo11n.pt --format engine   # -> yolo11n.engine (FP16)
```

> A TensorRT `.engine` is tuned to the exact GPU **and** TensorRT version and is
> **not portable** — build it on the deployment board, never commit it, and a
> JetPack upgrade invalidates it (rebuild). `.engine` is gitignored.

Then point the config at it:

```yaml
# config.yaml
detector:
  model_name: "yolo11n.engine"
  device: "cuda"
  use_half: true
  image_size: 640          # must match the --imgsz the engine was built for
```

`ObjectDetector` loads `.engine`/`.onnx` through the same `YOLO(...)` interface and
automatically skips the `.to(device)` / `half=` handling (device and precision are
baked into a serialized engine).

---

## 3. Run

```bash
# Live dashboard on an RTSP camera (the typical fixed-camera deployment):
flowcount-web --input rtsp://camera/stream --live

# CLI, headless, safety monitoring on a hazard zone with a webhook alert:
flowcount --input rtsp://camera/stream --live --no-display \
  --stall-zone 300,258,394,258,394,322,300,322 \
  --stabilize --alert-webhook https://example/alert
```

Start at `--detect-every 1` (you have the budget after TensorRT); raise it to 2–3
to buy thermal/power headroom or to afford the CPU-bound safety analytics — the
Kalman coasting between detections is what keeps the whole system real-time on the
6-core Arm CPU. Note that the CPU-bound safety stack (dual-rate background
subtraction, optical-flow stabilization), not YOLO, becomes the FPS ceiling once
TensorRT makes the detector cheap.

---

## Cameras

- **USB / UVC** cameras work today via `cv2.CAP_V4L2` (`--input 0`).
- **CSI ribbon** cameras (IMX219/477) are **not** V4L2 devices and won't open with
  the current video source — they need a GStreamer `nvarguscamerasrc` pipeline
  opened via `cv2.CAP_GSTREAMER`. That source is a planned addition; until then use
  a USB camera or an RTSP feed.
- **RTSP** works over FFmpeg today; an `nvv4l2decoder` GStreamer pipeline would add
  NVDEC hardware decode (future work).

## Biggest risks / gotchas

- torch↔JetPack version mapping and wheel URLs rotate constantly — pin the exact
  versions you validate.
- Install torch and torchvision as a matched pair from the same index, or you hit
  the notorious `torchvision::nms CUDA backend` error.
- Time steady-state (post-warmup) FPS, not the first frame — the first inference
  through a fresh engine includes a one-time build/load.
