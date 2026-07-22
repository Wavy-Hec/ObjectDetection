"""Learning the zone of interest from observed traffic.

Drawing a hazard polygon by hand means measuring pixel coordinates off a
screenshot for every camera, and redoing it whenever anything is remounted.
The scene already tells you where the drivable surface is — vehicles have been
driving over it — so this accumulates that evidence and proposes a polygon.

Two details make the difference between a usable proposal and a plausible-looking
wrong one:

**Accumulate box footprints, not track centres.** A vehicle's centre never
reaches the lane edge, so a centre-based density map describes a corridor
strictly narrower than the road. Measured against a ground-truth road quad,
footprint accumulation reached IoU 0.71 at 20 tracks where centre splats
reached 0.51.

**Gate on stability, not on elapsed time.** "Learn for five minutes" is not a
readiness signal — five minutes of one lane being used tells you nothing about
the other. Instead the polygon is re-derived every N qualifying tracks and
compared with the previous estimate; when consecutive estimates agree the
proposal has converged, which directly measures the thing you care about.

Contributing tracks are filtered the way trajectory-clustering work does — a
minimum length, a minimum total displacement, and directional consistency — so
parked cars and detector flicker do not vote on where the road is.

The result is a **proposal**, never a silent activation. A monitoring system
that quietly redraws its own alert zone is one that cannot be reasoned about;
an operator confirms it, and the geometry is written out ready to paste back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..analytics.base import Analyzer, Event, FrameContext
from .motion import transform_bbox


@dataclass
class ZoneLearnerConfig:
    """Tuning for :class:`ZoneLearner`."""

    #: Accumulate at 1/downscale resolution; the polygon is scaled back.
    downscale: int = 4
    #: A track must be seen this many times before it may vote.
    min_track_frames: int = 5
    #: ...and must have gone somewhere, so parked cars do not define the road.
    min_displacement_px: float = 50.0
    #: |sum of unit steps| / steps. Near 1 is purposeful travel; low values are
    #: detector flicker or an object milling about.
    direction_consistency: float = 0.8
    #: Density threshold as a fraction of the map's maximum.
    threshold_frac: float = 0.02
    close_kernel: int = 9
    open_kernel: int = 5
    #: approxPolyDP epsilon as a fraction of contour perimeter.
    approx_eps_frac: float = 0.01
    #: Reject a proposal covering less of the frame than this.
    min_area_frac: float = 0.01
    #: Re-derive the polygon every this many newly qualified tracks.
    window_tracks: int = 20
    #: Consecutive agreeing windows required before the proposal is offered.
    stable_windows: int = 3
    stable_iou: float = 0.85


@dataclass
class _TrackRecord:
    frames: int = 0
    start: tuple[float, float] = (0.0, 0.0)
    last: tuple[float, float] = (0.0, 0.0)
    direction_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))
    steps: int = 0
    qualified: bool = False

    def consistency(self) -> float:
        if self.steps == 0:
            return 0.0
        return float(np.linalg.norm(self.direction_sum) / self.steps)

    def displacement(self) -> float:
        return float(np.hypot(self.last[0] - self.start[0], self.last[1] - self.start[1]))


class ZoneLearner(Analyzer):
    """Proposes a zone-of-interest polygon from the traffic it has watched."""

    def __init__(self, config: ZoneLearnerConfig | None = None, *, frame_shape=None):
        self.cfg = config or ZoneLearnerConfig()
        self.density: np.ndarray | None = None
        self._shape = frame_shape
        self._records: dict[int, _TrackRecord] = {}
        self._qualified = 0
        self._last_window_at = 0
        self._previous_mask: np.ndarray | None = None
        self._agreements = 0
        self._polygon: list[tuple[float, float]] | None = None
        self._announced = False

    # ---- Analyzer -----------------------------------------------------------
    def update(self, ctx: FrameContext) -> list[Event]:
        if ctx.frame is not None and self.density is None:
            h, w = ctx.frame.shape[:2]
            d = max(1, self.cfg.downscale)
            self._shape = (h, w)
            self.density = np.zeros((h // d, w // d), dtype=np.float32)
        if self.density is None:
            return []

        for track in ctx.tracks:
            # Accumulate in the stabilizer's reference frame, so a swaying
            # camera does not smear the learned road across its own drift.
            box = transform_bbox(track.bbox, ctx.transform)
            record = self._records.get(track.id)
            if record is None:
                centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                record = _TrackRecord(start=centre, last=centre)
                self._records[track.id] = record

            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            step = np.array([centre[0] - record.last[0], centre[1] - record.last[1]])
            norm = float(np.linalg.norm(step))
            if norm > 1e-6:
                record.direction_sum += step / norm
                record.steps += 1
            record.last = centre
            record.frames += 1

            if not record.qualified and self._qualifies(record):
                record.qualified = True
                self._qualified += 1

            if record.qualified:
                self._accumulate(box)

        return self._maybe_close_window(ctx)

    def _qualifies(self, record: _TrackRecord) -> bool:
        return (
            record.frames >= self.cfg.min_track_frames
            and record.displacement() >= self.cfg.min_displacement_px
            and record.consistency() >= self.cfg.direction_consistency
        )

    def _accumulate(self, box) -> None:
        d = max(1, self.cfg.downscale)
        h, w = self.density.shape
        x1 = max(0, min(w, int(box[0] / d)))
        x2 = max(0, min(w, int(box[2] / d)))
        y1 = max(0, min(h, int(box[1] / d)))
        y2 = max(0, min(h, int(box[3] / d)))
        if x2 > x1 and y2 > y1:
            # The whole footprint, not just the centre: a vehicle's centre never
            # reaches the lane edge, so centres describe a corridor narrower
            # than the road itself.
            self.density[y1:y2, x1:x2] += 1.0

    def _maybe_close_window(self, ctx: FrameContext) -> list[Event]:
        if self._qualified - self._last_window_at < self.cfg.window_tracks:
            return []
        self._last_window_at = self._qualified

        mask = self._threshold_mask()
        if mask is None:
            return []

        if self._previous_mask is not None:
            inter = float(np.count_nonzero(mask & self._previous_mask))
            union = float(np.count_nonzero(mask | self._previous_mask))
            agreement = inter / union if union else 0.0
            self._agreements = self._agreements + 1 if agreement >= self.cfg.stable_iou else 0
        self._previous_mask = mask

        self._polygon = self._polygon_from(mask)
        if self.stable and not self._announced and self._polygon:
            self._announced = True
            return [
                Event(
                    kind="zone_proposal",
                    track_id=-1,
                    class_label="zone",
                    frame_index=ctx.frame_index,
                    timestamp=ctx.timestamp,
                    data={
                        "severity": "info",
                        "polygon": [[round(x, 1), round(y, 1)] for x, y in self._polygon],
                        "tracks_observed": self._qualified,
                        # A monitor that silently redraws its own alert zone
                        # cannot be reasoned about. Somebody confirms it.
                        "requires_human_review": True,
                    },
                )
            ]
        return []

    # ---- derivation ---------------------------------------------------------
    def _threshold_mask(self) -> np.ndarray | None:
        if self.density is None or float(self.density.max()) <= 0:
            return None
        level = float(self.density.max()) * self.cfg.threshold_frac
        mask = (self.density >= level).astype(np.uint8) * 255
        close = np.ones((self.cfg.close_kernel, self.cfg.close_kernel), np.uint8)
        opening = np.ones((self.cfg.open_kernel, self.cfg.open_kernel), np.uint8)
        # Close first to bridge the gaps between individual vehicles, then open
        # to shed the speckle from stray detections.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening)
        return mask

    def _polygon_from(self, mask: np.ndarray) -> list[tuple[float, float]] | None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count < 2:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest, cv2.CC_STAT_AREA] / mask.size < self.cfg.min_area_frac:
            return None

        component = (labels == largest).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        eps = self.cfg.approx_eps_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        d = max(1, self.cfg.downscale)
        return [(float(x * d), float(y * d)) for x, y in approx]

    # ---- accessors ----------------------------------------------------------
    @property
    def proposal(self) -> list[tuple[float, float]] | None:
        """The current polygon, or None if not enough evidence yet."""
        return self._polygon

    @property
    def stable(self) -> bool:
        """True once consecutive independent estimates agree."""
        return self._agreements >= self.cfg.stable_windows - 1

    def as_zone(self, name: str = "learned"):
        from ..analytics.zones import Zone

        if not self._polygon:
            return None
        return Zone(name, self._polygon)

    def cli_argument(self) -> str | None:
        """The proposal formatted for ``--zone``, ready to paste."""
        if not self._polygon:
            return None
        return ",".join(f"{int(round(x))},{int(round(y))}" for x, y in self._polygon)

    def draw(self, frame: np.ndarray) -> None:
        if not self._polygon:
            return
        pts = np.array(self._polygon, dtype=np.int32)
        colour = (0, 220, 120) if self.stable else (120, 120, 120)
        cv2.polylines(frame, [pts], True, colour, 2)
        label = "zone proposal" + ("" if self.stable else " (learning)")
        cv2.putText(
            frame,
            label,
            (int(pts[:, 0].min()), max(14, int(pts[:, 1].min()) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2,
        )

    def save(self, path_prefix: str) -> str | None:
        """Write the proposal as JSON, plus a density preview alongside it."""
        if not self._polygon:
            return None
        path = path_prefix if path_prefix.lower().endswith(".json") else f"{path_prefix}_zone.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "polygon": [[round(x, 1), round(y, 1)] for x, y in self._polygon],
                    "stable": self.stable,
                    "tracks_observed": self._qualified,
                    "cli": self.cli_argument(),
                },
                indent=2,
            )
        )
        if self.density is not None and float(self.density.max()) > 0:
            norm = (self.density / self.density.max() * 255).astype(np.uint8)
            preview = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            cv2.imwrite(path.rsplit(".", 1)[0] + "_density.jpg", preview)
        return path

    def stats(self) -> dict:
        return {
            "zone_learning": {
                "tracks_observed": self._qualified,
                "stable": self.stable,
                "has_proposal": self._polygon is not None,
            }
        }
