"""Export a YOLO model to a deployment format (TensorRT engine / ONNX).

FlowCount's dev/deploy split: you develop on x86 against the ``.pt`` weights, but
on a Jetson you want a **TensorRT ``.engine``** for FP16 speed (roughly a 3x
inference win over the plain PyTorch model on an Orin Nano). This wraps
``ultralytics``' exporter so that step is a first-class, repeatable command
rather than tribal knowledge.

    flowcount-export --model yolo11n.pt --format engine        # -> yolo11n.engine (FP16)
    flowcount-export --model yolo11n.pt --format onnx --no-half # -> yolo11n.onnx (FP32)

Then point ``config.yaml``'s ``model_name`` (or ``--model``) at the result;
``ObjectDetector`` loads ``.engine``/``.onnx`` through the same ``YOLO(...)``
interface and skips the device-move / half-precision handling that a serialized
model bakes in at export time.

IMPORTANT: a TensorRT engine is tuned to the exact GPU **and** TensorRT version
and is **not portable** — an engine built on a laptop will not load on the Orin,
and a JetPack upgrade invalidates it. Run this ON the deployment board as a
first-run step; never commit the ``.engine`` artifact.
"""

from __future__ import annotations

import argparse
import sys


def export_model(
    model_name: str,
    fmt: str = "engine",
    imgsz: int = 640,
    half: bool = True,
    device: str | None = None,
):
    """Export ``model_name`` to ``fmt`` and return the written artifact path.

    ``ultralytics`` is imported lazily so this module imports without the ML
    stack (matching the rest of FlowCount).
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    kwargs: dict = {"format": fmt, "imgsz": imgsz, "half": half}
    if device is not None:
        kwargs["device"] = device
    return model.export(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a YOLO model for deployment (Jetson TensorRT / ONNX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="yolo11n.pt", help="Source .pt weights")
    parser.add_argument(
        "--format",
        "-f",
        default="engine",
        choices=["engine", "onnx"],
        help="engine = TensorRT (Jetson/CUDA); onnx = portable graph",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size the model is built for (must match the runtime imgsz)",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FP16 (default); pass --no-half for FP32",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Export device, e.g. 0 for the first CUDA GPU (default: ultralytics picks)",
    )
    args = parser.parse_args()

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print(
            "ultralytics is required to export. Install it with "
            "`pip install 'flowcount[yolo]'` on x86, or the matched NVIDIA "
            "wheels on a Jetson (see docs/jetson.md).",
            file=sys.stderr,
        )
        return 1

    out = export_model(args.model, args.format, args.imgsz, args.half, args.device)
    print(f"Exported: {out}")
    print(
        "Point config.yaml model_name (or --model) at this file. "
        "TensorRT engines are device- and version-locked; rebuild after a JetPack upgrade."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
