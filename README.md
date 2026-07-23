# FlowCount — Real-Time Traffic Analytics

![CI](https://github.com/Wavy-Hec/ObjectDetection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Detector](https://img.shields.io/badge/detector-YOLO11%20%2F%20YOLO--World-blueviolet)

Point FlowCount at a webcam, video file, or IP camera and it detects vehicles,
tracks them with stable IDs, and turns the footage into traffic metrics:
counts, wrong-way alerts, zone occupancy, heatmaps, and exports — plus an
experimental [safety-monitoring mode](#safety-monitoring-experimental) (stalled
vehicles, intrusions, debris) that runs on a Jetson at the edge.

![FlowCount demo](assets/demo.gif)

**What you're watching:** a built-in synthetic traffic scene (cartoon vehicles,
no AI model or camera needed) running through the *real* detection → tracking →
analytics pipeline. Each vehicle gets a box and a persistent ID (`car #6`),
the yellow **count line** tallies vehicles as they cross, the blue **zone**
reports occupancy, and the red flash is the **WRONG WAY alert** catching the
car driving against traffic. On real footage, YOLO detections replace the
cartoon vehicles — everything else is identical.

| The activity heatmap: where vehicles spend their time — busy lanes glow |
|---|
| ![heatmap](assets/heatmap.jpg) |

---

## Features

- **Detection** — YOLO11 / YOLOv8, with automatic fallback to open-vocabulary
  YOLO-World for classes outside COCO.
- **Tracking** — ByteTrack-style two-stage matching on a custom SORT core
  (Kalman filter + Hungarian assignment). IDs survive occlusions, and a truck
  never inherits a car's ID.
- **Live mode** — background capture that drops stale frames (no latency
  build-up), detect-every-N frame skipping for CPU machines, and RTSP cameras
  with auto-reconnect.
- **Counting & alerts** — per-class in/out counts at virtual lines, wrong-way
  detection, polygon zones with occupancy + dwell time.
- **Safety monitoring** *(experimental)* — stalled-vehicle / intrusion /
  congestion detection in hazard zones, abandoned-object (debris) detection,
  camera-shake compensation, and learned zones-of-interest, with off-thread
  webhook alerts. See [Safety monitoring](#safety-monitoring-experimental).
- **Outputs** — activity heatmaps, event-triggered clips, CSV/SQLite export,
  annotated video.
- **Web dashboard** — live stream + stats in the browser (FastAPI + WebSocket),
  with a Docker image ready for a free public deploy.
- **Edge-ready** — lazy ML import, TensorRT export, and a Jetson runbook
  ([docs/jetson.md](docs/jetson.md)).

---

## Quickstart

```bash
pip install -e ".[yolo,web,demo]"
```

Weights download automatically on first real use (default: YOLO11n, needs
`ultralytics>=8.3`). GPU is used when available; everything below except real
footage also works with no ML dependencies at all.

**See it work instantly** (no model, no camera — regenerates the GIF above):

```bash
python scripts/demo.py
```

**Live webcam:**

```bash
flowcount --input 0 --live                # desktop window
flowcount-web --input 0 --live            # browser dashboard at :8000
```

**Real footage / IP cameras:**

```bash
# Count vehicles crossing a line, save a heatmap and a CSV
flowcount --input traffic.mp4 --count-line 640,0,640,720 --heatmap --export-csv runs/traffic.csv

# IP camera with wrong-way alerts and event clips
flowcount --input rtsp://camera/stream --live \
    --count-line 640,0,640,720 --expect-direction in --record-events runs/clips

# Slow CPU? Detect on every 4th frame — tracks coast in between with stable IDs
flowcount --input 0 --live --detect-every 4 --imgsz 480
```

Window controls: `q` quit · `p` pause · `s` save frame.
Defaults live in [config.yaml](config.yaml); any flag overrides them.

**Web dashboard** (also works with zero setup — it defaults to the synthetic
scene):

```bash
uvicorn flowcount.web.server:app          # open http://127.0.0.1:8000
flowcount-web --input traffic.mp4         # or real footage / webcam / RTSP
```

Live MJPEG stream, counts, per-class breakdown, zone occupancy, recent events
(including wrong-way alerts), and a refreshing heatmap. The page loads
instantly while the model warms up; `/healthz` supports deploy health checks.

---

## Safety monitoring (experimental)

Beyond "how many / how fast / which way", FlowCount can answer **"is something
wrong right now"** for a monitored zone — the job an AI grade-crossing monitor
does, generalized to any hazard area. The detectors live in
[`flowcount/safety/`](flowcount/safety/) and plug into the same pipeline through
the ordinary analyzer contract; nothing in the detection or tracking path knows
they exist.

```bash
flowcount --input rtsp://cam/stream --live --no-display \
  --stall-zone 300,258,394,258,394,322,300,322 \   # STALLED vehicle in a crossing
  --intrusion-zone 0,0,120,0,120,80,0,80 \          # person in a restricted area
  --debris --stabilize \                             # abandoned objects + camera-shake comp
  --alert-webhook https://example/alert              # POST each alert as JSON
```

Design choices worth a look: incidents are anchored to a **location**, not a
track ID (so SORT re-IDs can't silently reset a stall timer), stillness is
scale-invariant (not Kalman velocity), and thresholds follow PIARC's ~10 s
time-to-detect / low-false-alarm guidance. The browser dashboard's synthetic
scene ships a stalled car in a crossing, so you can watch a **STALLED** incident
fire with no model or camera:

```bash
uvicorn flowcount.web.server:app          # open http://127.0.0.1:8000
```

**Status:** the detection logic is fully implemented and tested (torch-free) and
wired into the CLI, dashboard, and synthetic scene. It has **not** yet been
validated against annotated real-world safety footage — treat the thresholds as
sensible defaults, not field-calibrated values.

---

## Put it online (free public demo)

The Dockerfile serves the synthetic dashboard with no GPU or weights:

```bash
docker build -t flowcount . && docker run -p 7860:7860 flowcount
```

To host it on **Hugging Face Spaces**:

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) — SDK: **Docker**, public.
2. `git remote add space https://huggingface.co/spaces/<user>/<space>`
3. `git push space main`

Your dashboard is now live at the Space URL, 24/7.

## Run on a Jetson (edge)

FlowCount runs on a Jetson Orin Nano. The tracker, analytics, and safety code
are pure aarch64-portable Python; the ML stack needs NVIDIA's matched
torch/TensorRT wheels (a plain `pip install torch` won't do), and `yolo11n`
exports to a FP16 TensorRT `.engine` for ~60–100 FPS full-pipeline:

```bash
flowcount-export --model yolo11n.pt --format engine   # on the board -> yolo11n.engine
docker build -f Dockerfile.jetson -t flowcount-jetson .
docker run --runtime nvidia --network host flowcount-jetson
```

Full runbook (power mode, wheel index, cameras, gotchas): [docs/jetson.md](docs/jetson.md).

---

## How it works

```mermaid
flowchart LR
    A[Camera / file / RTSP] --> B[YOLO11 detector]
    B --> C[Tracker<br/>Kalman + Hungarian,<br/>two-stage matching]
    C --> D[Analytics<br/>counts · zones · heatmap<br/>alerts · exports]
    D --> E[CLI window · dashboard · files]
```

One reusable [`Pipeline`](flowcount/pipeline.py) drives every front end — the
CLI, the dashboard, the demo, and the benchmarks share the exact same code
path. Architecture and tracker internals: [docs/DESIGN.md](docs/DESIGN.md).

**Performance:** tracking + analytics cost under 1 ms/frame (~360 FPS
end-to-end without a model); real-time behavior is set by YOLO inference,
which `--detect-every` and `--imgsz` trade against. Details:
[docs/benchmarks.md](docs/benchmarks.md).

---

## CLI reference

| Flag | Description |
|---|---|
| `--input`, `-i` | `0` = webcam, a file path, or an `rtsp://` URL |
| `--live` | Low-latency capture for cameras; implies `--detect-every 3` |
| `--detect-every N` | Run YOLO every Nth frame; tracks coast in between |
| `--model` / `--imgsz` / `--device` | Model name, inference size, `auto`/`cuda`/`cpu` |
| `--count-line x1,y1,x2,y2` | Line counter (repeatable) |
| `--expect-direction in\|out` | Wrong-way alerts against this direction |
| `--zone x1,y1,...` / `--dwell S` | Polygon zones, dwell events |
| `--heatmap` | Accumulate + save an activity heatmap |
| `--export-csv` / `--export-db PATH` | Export tracks + events |
| `--record-events DIR` | Save clips around each event |
| `--stall-zone` / `--intrusion-zone x1,y1,...` | Safety: stalled-vehicle / person-intrusion hazard zones |
| `--debris` / `--learn-zone` / `--stabilize` | Safety: debris detection · learn a zone · camera-shake comp |
| `--alert-webhook URL` / `--alert-min-severity` | POST safety alerts as JSON, above a severity |
| `--output`, `-o` | Write the annotated video |
| `--classes` / `--preset` | Class filter (non-COCO names auto-enable YOLO-World) |

A separate `flowcount-export` command builds a TensorRT/ONNX model for edge
deployment; `flowcount-web` serves the dashboard.

---

## Project structure

```
flowcount/           the package: detector, tracker, pipeline, video sources,
                     analytics/ (counting, zones, heatmap, export),
                     safety/ (stalled/intrusion/debris/stabilize/roi + alerts),
                     web/ dashboard, export.py (TensorRT/ONNX)
scripts/demo.py      regenerates the demo GIF + heatmap (synthetic or real footage)
scripts/bench.py     benchmark harness
docs/                DESIGN.md (architecture deep-dive), benchmarks.md, jetson.md
tests/               199 tests, no ML deps needed
```

## Roadmap

- [x] Detection, tracking, counting, zones, heatmaps, exports, clips
- [x] Web dashboard · live mode (webcam/RTSP, frame skipping) · wrong-way alerts
- [x] Packaging, CI, Docker, benchmarks
- [x] Safety monitoring (stalled/intrusion/congestion, debris, camera-shake, ROI learning, alerts)
- [x] Jetson/edge path: lazy ML import, TensorRT export, Jetson runbook
- [ ] Validate safety thresholds on annotated real-world footage
- [ ] Real-world km/h speed via camera calibration
- [ ] Draw count lines/zones in the browser · hosted live demo
- [ ] Fine-tune YOLO11 on a traffic dataset + MOT-metrics evaluation

## Development

```bash
pip install -e ".[web,demo,dev]"
pytest                                  # 199 tests, no torch needed
ruff check . && ruff format --check .
```

## License

MIT — see [LICENSE](LICENSE).
