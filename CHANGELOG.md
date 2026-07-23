# Changelog

All notable changes to FlowCount. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions track the
project's phase roadmap.

## [0.5.0] - 2026-07-23

### Added
- Safety-monitoring subpackage (`flowcount/safety/`) — "is something wrong right
  now" for a monitored zone, alongside the existing traffic analytics:
  - Stalled-vehicle / intrusion / congestion detection (`zone_incident.py`) via
    one rule-parameterised analyzer; thresholds follow PIARC's ~10 s
    time-to-detect and low-false-alarm guidance.
  - Incident lifecycle anchored to a **location**, not a track ID
    (`incidents.py`), so SORT re-IDs and class flips can't silently reset a
    stall timer; absence is not clearance (occlusion grace).
  - Scale-invariant stillness (`motion.py`): self-IoU at a lag + size-normalised
    radius, deliberately avoiding Kalman velocity (px/frame, non-zero while
    coasting).
  - Abandoned-object / debris detection (`static_objects.py`) by dual-rate
    background subtraction (Evangelio–Sikora FSM), tracker-masked.
  - Camera-motion compensation (`stabilize.py`): keyframe matching (not
    frame-to-frame chaining), 4-DOF partial affine; transforms geometry, not
    pixels.
  - Zone-of-interest learning (`roi.py`) from observed traffic (box footprints,
    convergence-gated).
  - Alert delivery (`alerts.py`): an `AlertDispatcher` that fans events to sinks
    on a background thread (never blocks the frame path), with `LogSink` and
    `WebhookSink` (stdlib only).
- CLI safety flags: `--stall-zone`, `--intrusion-zone`, `--debris`,
  `--learn-zone`, `--stabilize`, `--alert-webhook`, `--alert-min-severity`.
- Dashboard surfaces safety: `AnalyticsManager.stats()` is merged into the stats
  payload and a Safety card shows open incidents / camera status; the synthetic
  scene parks a stalled car in a crossing so a STALLED incident fires with no
  model or camera.
- Jetson/edge deployment: `ObjectDetector` loads serialized TensorRT `.engine` /
  ONNX models (skips device-move and half-precision, which are baked in at
  export); a `flowcount-export` command wraps the exporter; `Dockerfile.jetson`
  and `docs/jetson.md` document the full runbook.
- Pipeline plumbing for the above: `FrameContext` gains `detections`,
  `detection_ran`, `transform`, `prev_frame`; `Pipeline(stabilizer=...)` publishes
  a camera-motion transform without warping the frame; `AnalyticsManager` gains a
  `stats()` merge, per-analyzer exception isolation, and an `alert_dispatcher`
  hook; `Zone.set_polygon()` for moving zones; the heatmap splat is allocation-free.

### Fixed
- Camera-reposition detection was dead in the live pipeline: `process_frame`
  called the stabilizer's `estimate()` without a timestamp, so its time-based
  reposition timer never advanced and a real camera knock never suspended the
  incident detectors. The frame timestamp is now threaded through.

### Note
- The safety detection logic is fully tested (torch-free) and wired into the
  CLI, dashboard, and synthetic scene, but has not yet been validated against
  annotated real-world safety footage; thresholds are sensible defaults, not
  field-calibrated values.

## [0.4.0] - 2026-07-15

### Added
- ByteTrack-style two-stage association in the tracker: high-confidence
  detections match first and alone spawn tracks; low-confidence detections
  recover occluded tracks. Confirmed tracks are latched and coast through
  short misses (`output_coast`), and `predict_only(max_coast)` supports
  frame-skipping pipelines.
- Live mode: `Pipeline(detect_every=N)` runs the detector on every Nth frame
  with Kalman coasting in between; `StreamSource` adds RTSP/HTTP cameras with
  reconnect-and-backoff; `LatestFrameGrabber` drops stale frames so inference
  latency stays bounded. Exposed as `--live`, `--detect-every`, `--imgsz`,
  and `--device` on the CLI and dashboard.
- Dashboard upgrades: accumulated event history (crossings no longer lost
  between WebSocket pushes), async model warmup with an explicit "warming"
  state, sequence-gated MJPEG (no duplicate frames), source-paced engine FPS,
  and a `/healthz` endpoint.
- Wrong-way detection: `LineCrossingCounter(expected_direction=...)` emits
  `wrong_way` events and flashes the line red; the synthetic scene includes a
  wrong-way driver so demos show it firing.
- Packaging: the package is now `flowcount` with `pip install -e .`, extras
  (`[yolo]`, `[web]`, `[demo]`, `[dev]`), and `flowcount` / `flowcount-web`
  console scripts.
- Engineering: ruff lint+format, GitHub Actions CI (tests on 3.10–3.12,
  torch-free), pre-commit hooks, a Hugging-Face-Spaces-ready Dockerfile, a
  benchmark harness (`scripts/bench.py`), and `docs/benchmarks.md`.

### Changed
- Default detection model is YOLO11n (`ultralytics>=8.3`); YOLOv8 models
  remain supported. Default CLI preset is now `traffic`.
- Video files drive analytics with media time (`frame / fps`) instead of
  wall-clock time, so dwell and event timestamps no longer depend on
  processing speed. Speed overlays are labeled honestly as px/frame.

### Fixed
- Pausing the CLI no longer consumes (and skips) frames, and the PAUSED
  banner actually renders.
- Line crossings spanning a short detection dropout are counted (5-frame
  grace window).
- `config.yaml` no longer contains decorative knobs: everything left is
  wired, including device, image size, webcam capture settings, and
  `fps_limit`.
- `environment.yml` matches reality by delegating to `pyproject.toml`.

## [0.3.0] - 2026-06-04

### Added
- FastAPI + WebSocket web dashboard: MJPEG live stream, stats panel with
  per-class counts and zone occupancy, and a refreshing activity heatmap,
  all driven by the shared `Pipeline` on a background engine thread.
- Synthetic traffic scene so the dashboard and demo assets run with no
  model, camera, or GPU.
- FlowCount product narrative: traffic-analytics framing, `traffic` class
  preset, README demo GIF and heatmap generated by `scripts/demo.py`.

## [0.2.0] - 2026-06-04

### Added
- Analytics subpackage: line-crossing counter with per-class in/out buckets,
  polygon zones with occupancy and dwell events, activity heatmap
  accumulator, event-triggered clip recorder, and CSV/SQLite exporters — all
  consuming a shared `FrameContext` and emitting `Event` objects through
  `AnalyticsManager`.
- CLI flags for analytics: `--count-line`, `--zone`, `--dwell`, `--heatmap`,
  `--export-csv`, `--export-db`, `--record-events`.

## [0.1.0] - 2026-06-04

### Added
- Package layout with lazy exports, structured logging, and a typed
  `config.yaml` loader.
- Reusable `Pipeline` (`process_frame -> ProcessResult`) shared by every
  frontend; `main.py` reduced to a thin CLI driver.
- Test suite (48 tests at the time) running without any ML dependencies.
