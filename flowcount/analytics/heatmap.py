"""Activity heatmap accumulator.

Accumulates a Gaussian-ish splat at every track center over time and renders a
colorized overlay. A single saved heatmap makes an excellent portfolio thumbnail.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Analyzer, Event, FrameContext


class HeatmapAccumulator(Analyzer):
    def __init__(
        self,
        radius: int = 20,
        alpha: float = 0.5,
        decay: float = 1.0,
        colormap: int = cv2.COLORMAP_JET,
    ):
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
        self.accumulator: np.ndarray | None = None

    def update(self, ctx: FrameContext) -> list[Event]:
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
                # Add the splat through a view of just its bounding box. The
                # obvious version — allocate a full HxW array, draw one circle,
                # add it — costs ~8 MB of alloc + memset + a full-frame add per
                # track per frame at 1080p, which dominates the frame budget on
                # an edge device.
                r = self.radius
                x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                self.accumulator[y0:y1, x0:x1] += self._disc(r)[
                    y0 - (cy - r) : (y1 - (cy - r)), x0 - (cx - r) : (x1 - (cx - r))
                ]
        return []  # heatmap produces no discrete events

    def _disc(self, radius: int) -> np.ndarray:
        """A (2r+1)^2 float32 disc mask, built once per radius and reused."""
        cached = getattr(self, "_disc_cache", None)
        if cached is None or cached.shape[0] != 2 * radius + 1:
            disc = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
            cv2.circle(disc, (radius, radius), radius, 1.0, -1)
            self._disc_cache = disc
        return self._disc_cache

    def render(self) -> np.ndarray | None:
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

    def save(self, path: str) -> str | None:
        """Save the rendered heatmap.

        A ``path`` ending in an image extension is written verbatim; anything
        else is treated as a prefix and ``<path>_heatmap.jpg`` is written
        (the convention AnalyticsManager.save uses for all analyzers).
        """
        img = self.render()
        if img is None:
            return None
        if not path.lower().endswith((".jpg", ".jpeg", ".png")):
            path = f"{path}_heatmap.jpg"
        cv2.imwrite(path, img)
        return path
