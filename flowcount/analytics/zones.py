"""Zone / region-of-interest analytics: occupancy, enter/exit, and dwell time.

Dwell time is measured in wall-clock seconds (from ``FrameContext.timestamp``),
which is exact for a live camera and a good approximation when processing a
file near real time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from .base import Analyzer, Event, FrameContext, Point, polygon_to_contour


@dataclass(eq=False)  # auto __eq__ would compare _contour, a numpy array -> ValueError
class Zone:
    name: str
    polygon: Sequence[Point]
    classes: set[str] | None = None
    dwell_threshold_s: float | None = None  # emit a "dwell" event past this
    _contour: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._contour = polygon_to_contour(self.polygon)

    def set_polygon(self, polygon: Sequence[Point]) -> None:
        """Replace the polygon, rebuilding the cached contour.

        Always use this rather than assigning ``zone.polygon``: the contour is
        derived once in ``__post_init__``, so a bare assignment leaves
        ``contains()`` testing against the old shape. Learned and
        camera-drift-corrected zones move every frame.
        """
        self.polygon = polygon
        self._contour = polygon_to_contour(polygon)

    def contains(self, point: Point) -> bool:
        return cv2.pointPolygonTest(self._contour, (float(point[0]), float(point[1])), False) >= 0


class ZoneManager(Analyzer):
    def __init__(self, zones: Sequence[Zone]):
        self.zones: list[Zone] = list(zones)
        # per zone name: track_id -> entry timestamp
        self._entered: dict[str, dict[int, float]] = {z.name: {} for z in self.zones}
        # per zone name: track_ids that already fired a dwell event
        self._dwell_fired: dict[str, set[int]] = {z.name: set() for z in self.zones}

    def occupancy(self, zone_name: str) -> int:
        return len(self._entered.get(zone_name, {}))

    def update(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        by_id = {t.id: t for t in ctx.tracks}

        for zone in self.zones:
            entered = self._entered[zone.name]
            inside_now: set[int] = set()

            for track in ctx.tracks:
                if zone.classes and track.class_label not in zone.classes:
                    continue
                if zone.contains(track.get_center()):
                    inside_now.add(track.id)
                    if track.id not in entered:
                        entered[track.id] = ctx.timestamp
                        events.append(
                            Event(
                                "zone_enter",
                                track.id,
                                track.class_label,
                                ctx.frame_index,
                                ctx.timestamp,
                                {"zone": zone.name},
                            )
                        )
                    elif (
                        zone.dwell_threshold_s is not None
                        and track.id not in self._dwell_fired[zone.name]
                        and ctx.timestamp - entered[track.id] >= zone.dwell_threshold_s
                    ):
                        self._dwell_fired[zone.name].add(track.id)
                        events.append(
                            Event(
                                "dwell",
                                track.id,
                                track.class_label,
                                ctx.frame_index,
                                ctx.timestamp,
                                {
                                    "zone": zone.name,
                                    "dwell_s": round(ctx.timestamp - entered[track.id], 2),
                                },
                            )
                        )

            # Anyone previously inside but not inside now (or whose track vanished) -> exit.
            for tid in list(entered.keys()):
                if tid not in inside_now:
                    dwell_s = round(ctx.timestamp - entered.pop(tid), 2)
                    self._dwell_fired[zone.name].discard(tid)
                    label = by_id[tid].class_label if tid in by_id else "unknown"
                    events.append(
                        Event(
                            "zone_exit",
                            tid,
                            label,
                            ctx.frame_index,
                            ctx.timestamp,
                            {"zone": zone.name, "dwell_s": dwell_s},
                        )
                    )

        return events

    def draw(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        for zone in self.zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (255, 120, 0))
            cv2.polylines(frame, [pts], True, (255, 120, 0), 2)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            cv2.putText(
                frame,
                f"{zone.name}: {self.occupancy(zone.name)}",
                (int(cx) - 30, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 180),
                2,
            )
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
