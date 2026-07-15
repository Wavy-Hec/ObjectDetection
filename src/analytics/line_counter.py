"""Line-crossing counter.

Counts objects whose track center crosses a virtual line segment, bucketed by
class and direction ("in" / "out"). The canonical people/vehicle counting demo.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from .base import Analyzer, Event, FrameContext, Point, orient, segments_intersect


class LineCrossingCounter(Analyzer):
    # A track may briefly vanish (missed detection, detect-every-N gap) and
    # reappear with the same ID; keep its previous center for this many frames
    # so the crossing that happened during the gap is still counted.
    PRUNE_GRACE_FRAMES = 5

    def __init__(self, p1: Point, p2: Point, name: str = "line",
                 classes: Optional[Set[str]] = None):
        """
        Args:
            p1, p2: Endpoints of the counting line (image coordinates).
            name: Label for this line (used in events and overlay).
            classes: Restrict counting to these class labels (None = all).
        """
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.name = name
        self.classes = classes
        # counts[direction][class_label] = n
        self.counts: Dict[str, Dict[str, int]] = {"in": defaultdict(int), "out": defaultdict(int)}
        self._prev_center: Dict[int, Point] = {}
        self._last_seen: Dict[int, int] = {}  # track_id -> frame_index

    @property
    def total_in(self) -> int:
        return sum(self.counts["in"].values())

    @property
    def total_out(self) -> int:
        return sum(self.counts["out"].values())

    def update(self, ctx: FrameContext) -> List[Event]:
        events: List[Event] = []
        for track in ctx.tracks:
            if self.classes and track.class_label not in self.classes:
                continue
            cur = track.get_center()
            prev = self._prev_center.get(track.id)
            self._prev_center[track.id] = cur
            self._last_seen[track.id] = ctx.frame_index
            if prev is None:
                continue
            if segments_intersect(prev, cur, self.p1, self.p2):
                # Direction from sign change of the point's side of the line.
                side_cur = orient(self.p1, self.p2, cur)
                direction = "in" if side_cur >= 0 else "out"
                self.counts[direction][track.class_label] += 1
                events.append(Event(
                    kind="line_cross",
                    track_id=track.id,
                    class_label=track.class_label,
                    frame_index=ctx.frame_index,
                    timestamp=ctx.timestamp,
                    data={"line": self.name, "direction": direction},
                ))

        # Drop prev positions for ids unseen past the grace window, so a
        # retired id can't spuriously cross, but a short dropout still counts.
        stale = [tid for tid, seen in self._last_seen.items()
                 if ctx.frame_index - seen > self.PRUNE_GRACE_FRAMES]
        for tid in stale:
            self._prev_center.pop(tid, None)
            self._last_seen.pop(tid, None)
        return events

    def draw(self, frame: np.ndarray) -> None:
        p1 = (int(self.p1[0]), int(self.p1[1]))
        p2 = (int(self.p2[0]), int(self.p2[1]))
        cv2.line(frame, p1, p2, (0, 215, 255), 2)
        label = f"{self.name}  in:{self.total_in}  out:{self.total_out}"
        anchor = (min(p1[0], p2[0]), min(p1[1], p2[1]) - 8)
        cv2.putText(frame, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 215, 255), 2)
