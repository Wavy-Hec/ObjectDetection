"""Core analytics types and geometry helpers.

Every analyzer consumes a :class:`FrameContext` (built once per frame by the
pipeline) and returns a list of :class:`Event` objects describing anything
noteworthy that happened (a line crossing, a zone entry, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class FrameContext:
    """Everything an analyzer needs about the current frame."""

    tracks: List[Any]                 # List[Track] (avoid hard import / circularity)
    frame_index: int
    timestamp: float
    fps: float = 0.0
    frame: Optional[np.ndarray] = None


@dataclass
class Event:
    """A noteworthy occurrence produced by an analyzer."""

    kind: str                         # "line_cross" | "zone_enter" | "zone_exit" | "dwell"
    track_id: int
    class_label: str
    frame_index: int
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        extra = ", ".join(f"{k}={v}" for k, v in self.data.items())
        return f"{self.kind}[{self.class_label}#{self.track_id}] {extra}".strip()


class Analyzer(ABC):
    """Base class for per-frame analytics components."""

    @abstractmethod
    def update(self, ctx: FrameContext) -> List[Event]:
        """Process one frame's tracks and return any events produced."""
        raise NotImplementedError

    def draw(self, frame: np.ndarray) -> None:  # optional overlay
        """Draw this analyzer's overlay onto ``frame`` (in place). No-op by default."""
        return None

    def save(self, path_prefix: str) -> Optional[str]:
        """Optionally persist an artifact (e.g. a heatmap image). No-op by default."""
        return None


# ──────────────────────────── geometry helpers ────────────────────────────

def orient(a: Point, b: Point, c: Point) -> float:
    """Signed area * 2 of triangle (a, b, c). >0 = c left of a->b, <0 = right."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1-p2 properly intersects segment p3-p4.

    Collinear/touching edge cases are intentionally treated as no-intersection
    to avoid double counting at exact boundaries.
    """
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def polygon_to_contour(polygon: Sequence[Point]) -> np.ndarray:
    """Convert a list of points to an OpenCV contour (N,1,2) float32 array."""
    return np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
