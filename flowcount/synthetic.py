"""Synthetic traffic scene.

A self-contained, dependency-light traffic simulator used by the demo asset
generator (``scripts/demo.py``) and the web dashboard. It produces ``Detection``
boxes for vehicles moving across lanes so the *real* detection→tracking→
analytics pipeline can run with no model, camera, or GPU.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .detector import Detection

WIDTH, HEIGHT = 640, 360
LANE_Y = (70, 180, 290)

# (start_cx, lane_y, speed_px, class_label, w, h)
VEHICLES = [
    (-40, LANE_Y[0], 6.0, "car", 60, 34),
    (-120, LANE_Y[1], 5.0, "truck", 92, 44),
    (-80, LANE_Y[2], 8.0, "car", 60, 34),
    (-200, LANE_Y[0], 7.0, "motorcycle", 34, 24),
    (-160, LANE_Y[1], 4.5, "bus", 112, 50),
    (-240, LANE_Y[2], 6.0, "car", 60, 34),
    (-300, LANE_Y[0], 9.0, "car", 58, 32),
    (-280, LANE_Y[2], 7.0, "motorcycle", 32, 22),
]

# How far past the right edge before a vehicle recycles to the left.
_RECYCLE_AT = WIDTH + 120
_RECYCLE_PERIOD = WIDTH + 200


def render_road() -> np.ndarray:
    """Draw a fresh asphalt background with dashed lane separators + title bar."""
    frame = np.full((HEIGHT, WIDTH, 3), 55, dtype=np.uint8)
    for y in (125, 235):  # lane separators
        for x in range(0, WIDTH, 40):
            cv2.line(frame, (x, y), (x + 22, y), (200, 200, 200), 2)
    cv2.rectangle(frame, (0, 18), (WIDTH, 40), (35, 35, 35), -1)
    cv2.putText(frame, "FlowCount - Traffic Analytics (synthetic)",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return frame


class SyntheticTrafficDetector:
    """Acts as an ObjectDetector: returns boxes for vehicles that advance each
    frame. Vehicles recycle off the right edge for continuous (looping) flow."""

    def __init__(self, loop: bool = True):
        self.loop = loop
        self.vehicles = [list(v) for v in VEHICLES]  # mutable copies

    def detect(self, frame) -> List[Detection]:
        dets = []
        for cx, cy, _speed, cls, w, h in self.vehicles:
            if -w < cx < WIDTH + w:
                dets.append(Detection(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], cls, 0.9))
        return dets

    def advance(self):
        for v in self.vehicles:
            v[0] += v[2]  # cx += speed
            if self.loop and v[0] > _RECYCLE_AT:
                v[0] -= _RECYCLE_PERIOD  # wrap back to the left as a "new" vehicle
