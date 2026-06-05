"""Activity heatmap accumulator.

Accumulates a Gaussian-ish splat at every track center over time and renders a
colorized overlay. A single saved heatmap makes an excellent portfolio thumbnail.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .base import Analyzer, Event, FrameContext


class HeatmapAccumulator(Analyzer):
    def __init__(self, radius: int = 20, alpha: float = 0.5, decay: float = 1.0,
                 colormap: int = cv2.COLORMAP_JET):
        """
        Args:
            radius: Splat radius (pixels) added per track per frame.
            alpha: Overlay blend strength when drawing onto a frame.
            decay: Per-frame multiplier (<1.0 emphasizes recent activity).
            colormap: OpenCV colormap for rendering.
        """
        self.radius = radius
        self.alpha = alpha
        self.decay = decay
        self.colormap = colormap
        self.accumulator: Optional[np.ndarray] = None

    def update(self, ctx: FrameContext) -> List[Event]:
        if ctx.frame is None:
            return []
        h, w = ctx.frame.shape[:2]
        if self.accumulator is None or self.accumulator.shape != (h, w):
            self.accumulator = np.zeros((h, w), dtype=np.float32)
        if self.decay < 1.0:
            self.accumulator *= self.decay

        for track in ctx.tracks:
            cx, cy = track.get_center()
            cx, cy = int(cx), int(cy)
            if 0 <= cy < h and 0 <= cx < w:
                blob = np.zeros((h, w), dtype=np.float32)
                cv2.circle(blob, (cx, cy), self.radius, 1.0, -1)
                self.accumulator += blob
        return []  # heatmap produces no discrete events

    def render(self) -> Optional[np.ndarray]:
        """Return a BGR colorized heatmap image, or None if nothing accumulated."""
        if self.accumulator is None or float(self.accumulator.max()) <= 0:
            return None
        norm = (self.accumulator / self.accumulator.max() * 255.0).astype(np.uint8)
        norm = cv2.GaussianBlur(norm, (0, 0), sigmaX=max(1.0, self.radius / 2))
        return cv2.applyColorMap(norm, self.colormap)

    def draw(self, frame: np.ndarray) -> None:
        if self.accumulator is None or float(self.accumulator.max()) <= 0:
            return
        norm = (self.accumulator / self.accumulator.max() * 255.0).astype(np.uint8)
        norm = cv2.GaussianBlur(norm, (0, 0), sigmaX=max(1.0, self.radius / 2))
        colored = cv2.applyColorMap(norm, self.colormap)
        mask = norm > 0
        frame[mask] = (frame[mask] * (1 - self.alpha) + colored[mask] * self.alpha).astype(np.uint8)

    def save(self, path_prefix: str) -> Optional[str]:
        img = self.render()
        if img is None:
            return None
        path = f"{path_prefix}_heatmap.jpg"
        cv2.imwrite(path, img)
        return path
