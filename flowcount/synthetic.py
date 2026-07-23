"""Synthetic traffic scene.

A self-contained, dependency-light traffic simulator used by the demo asset
generator (``scripts/demo.py``) and the web dashboard. It draws recognizable
vehicles moving across lanes AND produces matching ``Detection`` boxes, so the
*real* detection→tracking→analytics pipeline can run — and look like it's
watching traffic — with no model, camera, or GPU.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

WIDTH, HEIGHT = 640, 360
LANE_Y = (70, 180, 290)

# The hazard/crossing zone the safety monitor watches. A vehicle that stops
# inside it raises a STALLED incident — the grade-crossing headline case. Sized
# to contain STALLED_VEHICLE below.
HAZARD_ZONE = [(300, 258), (394, 258), (394, 322), (300, 322)]

# (start_cx, lane_y, speed_px, class_label, w, h, body_color_bgr). Negative
# speed = a wrong-way driver (fuels the WRONG WAY alert in demos/dashboard).
VEHICLES = [
    (-40, LANE_Y[0], 6.0, "car", 60, 34, (235, 235, 235)),  # white
    (-120, LANE_Y[1], 5.0, "truck", 92, 44, (190, 190, 190)),  # silver
    (-80, LANE_Y[2], 8.0, "car", 60, 34, (190, 120, 50)),  # blue
    (-200, LANE_Y[0], 7.0, "motorcycle", 34, 24, (70, 70, 70)),  # dark grey
    (-160, LANE_Y[1], 4.5, "bus", 112, 50, (40, 170, 250)),  # school-bus yellow
    (-240, LANE_Y[2], 6.0, "car", 60, 34, (90, 180, 90)),  # green
    (-300, LANE_Y[0], 9.0, "car", 58, 32, (200, 160, 60)),  # teal
    (-280, LANE_Y[2], 7.0, "motorcycle", 32, 22, (60, 60, 140)),  # maroon
    (WIDTH + 260, LANE_Y[1], -5.5, "car", 60, 34, (60, 60, 230)),  # red, wrong way
]

# A permanently stalled car sitting in the crossing (speed 0). Kept OUT of the
# base VEHICLES list so it is strictly opt-in: the safety integration tests
# drive that list and stall a vehicle of their own, and a second pre-stalled car
# would double their incident count. The dashboard opts in via
# ``SyntheticTrafficDetector(include_stalled=True)``.
STALLED_VEHICLE = (347, LANE_Y[2], 0.0, "car", 60, 34, (45, 45, 175))  # dark red

# How far past the edge before a vehicle recycles to the other side.
_RECYCLE_AT = WIDTH + 120
_RECYCLE_PERIOD = WIDTH + 200

_ASPHALT = (62, 62, 62)
_SHOULDER = (60, 105, 60)  # grass strips framing the road
_WINDOW = (90, 70, 50)  # dark glass, slightly blue
_TIRE = (25, 25, 25)


def render_road() -> np.ndarray:
    """Draw a fresh road: grass shoulders, asphalt, dashed lanes, title bar."""
    frame = np.full((HEIGHT, WIDTH, 3), _ASPHALT, dtype=np.uint8)
    frame[:46] = _SHOULDER
    frame[HEIGHT - 22 :] = _SHOULDER
    for y in (125, 235):  # dashed lane separators
        for x in range(0, WIDTH, 40):
            cv2.line(frame, (x, y), (x + 22, y), (210, 210, 210), 2)
    # Solid edge lines where asphalt meets grass.
    cv2.line(frame, (0, 48), (WIDTH, 48), (230, 230, 230), 2)
    cv2.line(frame, (0, HEIGHT - 24), (WIDTH, HEIGHT - 24), (230, 230, 230), 2)
    cv2.rectangle(frame, (0, 0), (WIDTH, 24), (35, 35, 35), -1)
    cv2.putText(
        frame,
        "FlowCount - synthetic traffic scene (no model, no camera)",
        (10, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
    )
    return frame


def _draw_vehicle(
    frame: np.ndarray, cx: float, cy: float, speed: float, cls: str, w: int, h: int, color
) -> None:
    """Draw a simple side-view vehicle sprite centered at (cx, cy)."""
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    wheel_y = y2 - max(2, h // 10)
    wheel_r = max(3, h // 5)
    body_bottom = wheel_y - wheel_r // 2

    if cls == "motorcycle":
        # Two wheels, a low frame, and a rider hump.
        cv2.circle(frame, (x1 + wheel_r, wheel_y), wheel_r, _TIRE, -1)
        cv2.circle(frame, (x2 - wheel_r, wheel_y), wheel_r, _TIRE, -1)
        cv2.line(frame, (x1 + wheel_r, wheel_y - 2), (x2 - wheel_r, wheel_y - 2), color, 3)
        cv2.circle(frame, (int(cx), y1 + h // 3), max(3, h // 4), color, -1)
        return

    # Body slab (cars/trucks/buses).
    cv2.rectangle(frame, (x1, y1 + h // 3), (x2, body_bottom), color, -1)

    if cls == "bus":
        # Full-height body with a row of windows.
        cv2.rectangle(frame, (x1, y1), (x2, body_bottom), color, -1)
        win_w = max(4, w // 9)
        for wx in range(x1 + 6, x2 - win_w - 2, win_w + 5):
            cv2.rectangle(frame, (wx, y1 + 5), (wx + win_w, y1 + h // 3), _WINDOW, -1)
    elif cls == "truck":
        # Trailer + a cab at the leading end.
        cab_w = w // 4
        if speed >= 0:
            cv2.rectangle(frame, (x2 - cab_w, y1 + h // 5), (x2, body_bottom), color, -1)
            cv2.rectangle(
                frame, (x2 - cab_w + 3, y1 + h // 5 + 3), (x2 - 4, y1 + h // 2), _WINDOW, -1
            )
            cv2.rectangle(frame, (x1, y1), (x2 - cab_w - 3, body_bottom), (150, 150, 150), -1)
        else:
            cv2.rectangle(frame, (x1, y1 + h // 5), (x1 + cab_w, body_bottom), color, -1)
            cv2.rectangle(
                frame, (x1 + 4, y1 + h // 5 + 3), (x1 + cab_w - 3, y1 + h // 2), _WINDOW, -1
            )
            cv2.rectangle(frame, (x1 + cab_w + 3, y1), (x2, body_bottom), (150, 150, 150), -1)
    else:  # car
        # Cabin with a windshield, biased toward the direction of travel.
        cabin_x1 = x1 + w // 5
        cabin_x2 = x2 - w // 3
        if speed < 0:
            cabin_x1, cabin_x2 = x1 + w // 3, x2 - w // 5
        cv2.rectangle(frame, (cabin_x1, y1), (cabin_x2, y1 + h // 3), color, -1)
        cv2.rectangle(frame, (cabin_x1 + 3, y1 + 2), (cabin_x2 - 3, y1 + h // 3), _WINDOW, -1)

    cv2.circle(frame, (x1 + w // 5, wheel_y), wheel_r, _TIRE, -1)
    cv2.circle(frame, (x2 - w // 5, wheel_y), wheel_r, _TIRE, -1)


class SyntheticTrafficDetector:
    """Acts as an ObjectDetector: returns boxes for vehicles that advance each
    frame. Vehicles recycle off the edge for continuous (looping) flow.
    ``render_frame()`` draws the matching scene, so the tracked objects are
    actually visible."""

    def __init__(self, loop: bool = True, *, include_stalled: bool = False):
        self.loop = loop
        source = [*VEHICLES, STALLED_VEHICLE] if include_stalled else VEHICLES
        self.vehicles = [list(v) for v in source]  # mutable copies

    def detect(self, frame) -> list[Detection]:
        dets = []
        for cx, cy, _speed, cls, w, h, _color in self.vehicles:
            if -w < cx < WIDTH + w:
                dets.append(Detection([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], cls, 0.9))
        return dets

    def render_frame(self) -> np.ndarray:
        """Road plus every vehicle at its current position."""
        frame = render_road()
        for cx, cy, speed, cls, w, h, color in self.vehicles:
            if -w < cx < WIDTH + w:
                _draw_vehicle(frame, cx, cy, speed, cls, w, h, color)
        return frame

    def advance(self):
        for v in self.vehicles:
            v[0] += v[2]  # cx += speed
            if not self.loop:
                continue
            if v[2] > 0 and v[0] > _RECYCLE_AT:
                v[0] -= _RECYCLE_PERIOD  # wrap back to the left as a "new" vehicle
            elif v[2] < 0 and v[0] < -120:
                v[0] += _RECYCLE_PERIOD  # wrong-way driver re-enters from the right
