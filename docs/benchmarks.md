# Benchmarks

Reproduce any table with [`scripts/bench.py`](../scripts/bench.py).

## Pipeline overhead (no ML model)

Mode A pushes the synthetic traffic scene (640x360, ~9 concurrent tracks)
through the **real** pipeline — ByteTrack-style tracker, line counter,
annotation — with a near-free synthetic detector, so it measures everything
*except* YOLO inference. This is the framework's floor: with a real model,
detection dominates and these costs are the fixed overhead on top.

Hardware: shared Linux server (x86_64, CPU only), Python 3.11 —
`python scripts/bench.py --frames 300`

| Configuration | FPS |
|---|---|
| full pipeline (annotate + analytics) | 358 |
| detect every 2nd frame | 506 |
| detect every 4th frame | 760 |
| no annotation | 1117 |
| no annotation, no analytics | 1154 |

Takeaway: tracking + analytics cost well under 1 ms/frame; annotation is the
largest non-model cost. The pipeline never bottlenecks a real-time feed —
model inference does, which is what `--detect-every` exists for.

## Real YOLO inference (mode B)

Run on your inference machine (requires the `[yolo]` extra):

```bash
python scripts/bench.py --models yolo11n.pt yolo11s.pt \
    --imgsz 640 416 --detect-every 1 2 4 --input your_traffic.mp4 \
    --out docs/benchmarks-gpu.md
```

| Model | imgsz | detect every | detect ms | pipeline FPS |
|---|---|---|---|---|
| _run mode B to fill this in_ | | | | |

With `--detect-every N`, the detector runs on every Nth frame and confirmed
tracks coast on Kalman prediction in between, so display/analytics FPS is
roughly N x the detection rate with stable IDs.
