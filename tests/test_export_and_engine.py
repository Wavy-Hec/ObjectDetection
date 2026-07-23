"""Jetson deployment path: model export + serialized-engine loading.

Both are exercised with a fake ``ultralytics`` (and ``torch``) injected into
sys.modules, so they run in the torch-free CI exactly like the rest of the
suite — no real weights, CUDA, or TensorRT needed.
"""

from __future__ import annotations

import sys
import types


def _fake_ultralytics(model_factory):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = model_factory
    return mod


# ── flowcount-export ────────────────────────────────────────────────────────
def test_export_model_requests_engine_fp16(monkeypatch):
    calls = {}

    class FakeModel:
        def __init__(self, name):
            calls["name"] = name

        def export(self, **kw):
            calls["kw"] = kw
            return "yolo11n.engine"

    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics(FakeModel))
    from flowcount.export import export_model

    out = export_model("yolo11n.pt", "engine", imgsz=640, half=True)
    assert out == "yolo11n.engine"
    assert calls["name"] == "yolo11n.pt"
    assert calls["kw"] == {"format": "engine", "imgsz": 640, "half": True}


def test_export_model_forwards_device_only_when_given(monkeypatch):
    seen = {}

    class FakeModel:
        def __init__(self, name):
            pass

        def export(self, **kw):
            seen.update(kw)
            return "m.onnx"

    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics(FakeModel))
    from flowcount.export import export_model

    export_model("m.pt", "onnx", half=False)
    assert "device" not in seen
    export_model("m.pt", "onnx", half=False, device="0")
    assert seen["device"] == "0"


# ── ObjectDetector serialized-engine awareness ──────────────────────────────
def _install_fakes(monkeypatch, cuda_available=True):
    to_calls = []
    predict_kwargs = []

    class FakeModel:
        names = {0: "person", 2: "car"}

        def to(self, device):
            to_calls.append(device)

        def __call__(self, *a, **k):
            predict_kwargs.append(k)
            return []

        def set_classes(self, classes):
            pass

    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics(lambda name: FakeModel()))
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return to_calls, predict_kwargs


def test_engine_model_skips_device_move_and_half(monkeypatch):
    to_calls, predict_kwargs = _install_fakes(monkeypatch, cuda_available=True)
    from flowcount.detector import ObjectDetector

    det = ObjectDetector(model_name="yolo11n.engine", use_half=True, device="auto")
    det._ensure_model()

    # A serialized engine bakes in device + precision: never move it, never half.
    assert to_calls == []
    assert det.use_half is False
    # device is still resolved (used elsewhere for logging), just not applied.
    assert det.device == "cuda"
    assert predict_kwargs and predict_kwargs[0].get("half") is False


def test_pt_model_still_moves_to_device_and_uses_half(monkeypatch):
    to_calls, predict_kwargs = _install_fakes(monkeypatch, cuda_available=True)
    from flowcount.detector import ObjectDetector

    det = ObjectDetector(model_name="yolo11n.pt", use_half=True, device="auto")
    det._ensure_model()

    assert to_calls == ["cuda"]  # .pt models are moved as before
    assert det.use_half is True
    assert predict_kwargs and predict_kwargs[0].get("half") is True
