"""Debris / abandoned-object detection by dual-rate background subtraction.

COCO has no "debris" class, and open-vocabulary detection is a poor fit: the
published evaluations have YOLO-World largely missing road obstacles, and
FlowCount's own auto-switch would swap the *entire* detector for
``yolov8x-worldv2.pt`` — an X-size model, unusable on an Orin Nano. So this is
posed as an anomaly problem over pixels rather than a detection problem.

The classic formulation (Evangelio & Sikora's pixel finite-state machine) runs
two background models differing only in learning rate: a pixel that is
foreground to the *slow* model but background to the *fast* one has stopped
moving recently, which is the abandoned-object signature.

    fast    slow     meaning
    0       0        background
    1       1        moving foreground
    0       1        STATIC FOREGROUND  <- candidate
    1       0        uncovered background / ghost / lighting artifact

That structure is kept, but with the slow model replaced. Measured against
OpenCV's MOG2, a static object is promoted into the background after roughly
25 frames more or less regardless of ``history`` — its promotion schedule is
governed by internal weight dynamics, not by a knob. That is far too short to
hold an object across a 30-second persistence window, and it is not tunable.

So the long-term evidence comes from a **clean plate** whose timing is set
here explicitly, and which has a property the abandoned-object literature
mostly cannot assume: FlowCount has a tracker, so the plate is updated *only
where no track covers the pixel* and *only where staticness is not already
accumulating*. A stopped vehicle therefore can never be absorbed into the
reference — the failure mode where the object goes invisible before the alarm
fires becomes structurally impossible rather than merely unlikely. MOG2 is
kept for what it is genuinely good at: identifying what is moving right now,
and flagging scene-wide illumination steps.

    candidate = differs-from-plate  AND NOT moving  AND NOT covered-by-a-track

Everything runs at a few Hz on a downscaled frame. At 320px wide that is a
fraction of a millisecond, versus ~10 ms for the same work at 1080p — which
matters, because on an Orin Nano the detector's Python pre/post-processing
already dominates the frame budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..analytics.base import Analyzer, Event, FrameContext
from ..analytics.zones import Zone
from .incidents import IncidentTracker, Observation
from .motion import iou

#: MOG2 marks shadow pixels 127 and hard foreground 255. Thresholding above 200
#: drops shadows, which would otherwise become debris blobs on any sunny day.
_FG_THRESHOLD = 200


@dataclass
class DebrisConfig:
    """Tuning for :class:`StaticObjectMonitor`.

    Defaults target a fixed road camera at 30 fps and are stated in seconds, not
    frames, so they survive a change of frame rate or of ``detect_every``.
    """

    #: Width to downscale to before any pixel work. The cost driver.
    work_width: int = 320
    #: How often to run the background models, in Hz.
    rate_hz: float = 5.0
    #: MOG2 history for the "what is moving right now" model, in processed
    #: frames (~5 s at 5 Hz). The second model is used only for the
    #: illumination guard; long-term evidence comes from the clean plate.
    fast_history: int = 25
    slow_history: int = 500
    var_threshold: float = 16.0
    #: How long a region must stay static before it is reported.
    persist_s: float = 30.0
    #: Ignore the first stretch of video while the background models converge.
    warmup_s: float = 20.0
    #: Blob area as a fraction of frame area. The upper bound is what stops a
    #: whole-scene lighting change from ever being reported as one huge object.
    min_area_frac: float = 0.0004
    max_area_frac: float = 0.04
    #: Overlap with an active track above which a blob is that track, not
    #: debris. A stopped vehicle is a stall, and is reported as one.
    track_overlap: float = 0.15
    #: Clean-plate EMA rate per processed frame, once a pixel is learned.
    #: 0.002 at 5 Hz is a ~100 s time constant, comfortably longer than
    #: persist_s so an object cannot fade into the reference mid-alert.
    plate_alpha: float = 0.002
    #: Absolute pixel difference from the plate that counts as "changed".
    plate_diff_threshold: int = 25
    #: If more than this fraction of pixels flips to foreground in one step, the
    #: scene lighting changed (cloud front, auto-exposure step, IR-cut switch)
    #: rather than the world. Freeze and relearn instead of alarming.
    illumination_flip_frac: float = 0.28
    #: Seconds of suppression after a lighting transient.
    illumination_cooldown_s: float = 10.0
    #: A real object adds edges; a standing headlight pool is bright but
    #: featureless, and MOG2's shadow test is structurally blind to it because
    #: it only looks for pixels that got *darker*.
    min_edge_ratio: float = 1.15
    t_clear: float = 5.0
    cooldown_s: float = 60.0
    severity: str = "low"


class StaticObjectMonitor(Analyzer):
    """Reports objects that appear in the scene and then stop moving."""

    #: Times a pixel must have been observed unoccupied before its plate
    #: value is trusted as a reference.
    MIN_PLATE_SAMPLES = 10

    def __init__(
        self,
        zones: Sequence[Zone] = (),
        config: DebrisConfig | None = None,
        *,
        kind: str = "debris",
    ):
        self.zones = list(zones)
        self.cfg = config or DebrisConfig()
        self.kind = kind

        self._fast = None
        self._slow = None
        self._score: np.ndarray | None = None
        self._plate: np.ndarray | None = None
        self._plate_count: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._scale = 1.0

        self._incidents = IncidentTracker(
            kind,
            t_raise=0.0,  # persistence is enforced by the pixel score
            t_escalate=(),
            t_clear=self.cfg.t_clear,
            occlusion_grace_s=self.cfg.t_clear,
            cooldown_s=self.cfg.cooldown_s,
            # Debris does not move, so vanishing is the normal clear. This
            # guards the other case: a blob a full size away is a different
            # object, not this one having shifted. Zero here would make any
            # sub-pixel centroid jitter read as the object relocating.
            clear_displacement=1.0,
            confirmations=1,
        )

        self._first_ts: float | None = None
        self._last_processed_ts: float | None = None
        self._suppress_until_ts: float = float("-inf")
        self._now: float = 0.0
        self._blobs: list[tuple[float, float, float, float]] = []

    # ---- lifecycle ----------------------------------------------------------
    @property
    def score_on(self) -> float:
        """Pixel score at which a region is considered confirmed-static."""
        return max(1.0, self.cfg.persist_s * self.cfg.rate_hz)

    @property
    def warming(self) -> bool:
        if self._first_ts is None:
            return True
        return self._now - self._first_ts < self.cfg.warmup_s

    def _reset(self, shape: tuple[int, int]) -> None:
        self._shape = shape
        self._fast = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.fast_history, varThreshold=self.cfg.var_threshold, detectShadows=True
        )
        self._slow = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.slow_history, varThreshold=self.cfg.var_threshold, detectShadows=True
        )
        self._score = np.zeros(shape, dtype=np.float32)
        self._plate = None
        self._plate_count = None

    # ---- Analyzer -----------------------------------------------------------
    def update(self, ctx: FrameContext) -> list[Event]:
        if ctx.frame is None:
            return []
        self._now = ctx.timestamp
        if self._first_ts is None:
            self._first_ts = ctx.timestamp

        period = 1.0 / self.cfg.rate_hz if self.cfg.rate_hz > 0 else 0.0
        if self._last_processed_ts is not None and ctx.timestamp - self._last_processed_ts < period:
            return []
        self._last_processed_ts = ctx.timestamp

        small = self._downscale(ctx.frame)
        if self._shape != small.shape[:2]:
            self._reset(small.shape[:2])

        moving = cv2.threshold(self._fast.apply(small), _FG_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        slow = cv2.threshold(self._slow.apply(small), _FG_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

        if self._illumination_transient(slow, small, ctx.timestamp):
            return []

        occupancy = self._occupancy_mask(ctx)
        changed = self._plate_vote(small, occupancy)

        # Static foreground: differs from the learned clean background, is not
        # moving right now, and is not something the detector already explains.
        candidate = cv2.bitwise_and(changed, cv2.bitwise_not(moving))
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(occupancy))

        self._accumulate(candidate)

        if self.warming or ctx.timestamp < self._suppress_until_ts:
            return []

        blobs = self._blobs_from_score(ctx, small)
        self._blobs = [b[0] for b in blobs]
        observations = [
            Observation(
                track_id=self._cell_id(box),
                bbox=box,
                class_label=self.kind,
                still=True,
                zone=zone,
            )
            for box, zone in blobs
        ]
        return [
            self._to_event(update, ctx)
            for update in self._incidents.observe(ctx.timestamp, observations)
        ]

    # ---- pixel stages -------------------------------------------------------
    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._scale = w / float(self.cfg.work_width)
        if self._scale <= 1.0:
            self._scale = 1.0
            return frame
        # Halve with a Gaussian pyramid while more than a 2x reduction remains,
        # then finish with INTER_AREA. Same anti-aliasing, ~3x cheaper at 1080p
        # (3.5 ms -> 1.0 ms) because each pyramid step works on a quarter of the
        # pixels of the one before. Aliasing does matter here: a plain
        # INTER_LINEAR 6x reduction speckles the background model.
        #
        # Note INTER_AREA is passed by keyword — the third positional argument
        # of cv2.resize is `dst`, so passing the flag there silently selects the
        # default INTER_LINEAR instead.
        small = frame
        while small.shape[1] >= self.cfg.work_width * 2:
            small = cv2.pyrDown(small)
        return cv2.resize(
            small,
            (self.cfg.work_width, max(1, int(round(h / self._scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _illumination_transient(self, slow: np.ndarray, small: np.ndarray, now: float) -> bool:
        """Detect a scene-wide lighting step and relearn instead of alarming."""
        flipped = float(np.count_nonzero(slow)) / slow.size
        if flipped < self.cfg.illumination_flip_frac:
            return False
        # Force both models to adopt the new lighting quickly, and wipe the
        # staticness evidence gathered under the old lighting.
        self._fast.apply(small, learningRate=0.3)
        self._slow.apply(small, learningRate=0.3)
        self._score[:] = 0.0
        self._plate = small.astype(np.float32)
        # Count back to zero, not to the trusted threshold: this frame
        # contains whatever traffic happened to be passing, and baking that
        # into the reference would report it as debris the moment it left.
        # Pixels re-earn trust by being observed unoccupied.
        self._plate_count = np.zeros(self._shape, dtype=np.int32)
        self._suppress_until_ts = now + self.cfg.illumination_cooldown_s
        return True

    def _occupancy_mask(self, ctx: FrameContext) -> np.ndarray:
        """255 where a tracked object currently sits, in work resolution."""
        mask = np.zeros(self._shape, dtype=np.uint8)
        for track in ctx.tracks:
            x1, y1, x2, y2 = (int(v / self._scale) for v in track.bbox)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def _plate_vote(self, small: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        """Long-term change against a clean plate that tracks never contaminate.

        This carries the "has this region changed?" evidence rather than a slow
        MOG2, because MOG2's promotion of a static object into its background is
        governed by its own internals — measured at roughly 25 frames whatever
        ``history`` is set to — which is far too short to hold an object for a
        30 s persistence window and is not a knob. The plate's timing is set
        here, explicitly, by ``plate_alpha``.
        """
        if self._plate is None:
            self._plate = small.astype(np.float32)
            self._plate_count = np.zeros(self._shape, dtype=np.int32)

        # Update only where nothing is tracked AND nothing is accumulating
        # staticness. The first is the point of having a tracker: a stopped
        # vehicle can never be absorbed into the reference, which is the classic
        # way abandoned-object detectors go blind. The second stops confirmed
        # debris from healing into the plate and reporting itself cleared while
        # it is still lying there.
        free = (occupancy == 0) & (self._score < self.score_on * 0.5)
        counts = self._plate_count[free]
        # Running mean while a pixel is still being learned, slow EMA after —
        # so the plate converges in the first seconds instead of carrying
        # whatever happened to be in frame one for minutes.
        rates = np.maximum(self.cfg.plate_alpha, 1.0 / (counts + 1.0)).astype(np.float32)
        if small.ndim == 3:
            rates = rates[:, None]
        self._plate[free] += (small[free].astype(np.float32) - self._plate[free]) * rates
        self._plate_count[free] = counts + 1

        diff = cv2.absdiff(small, self._plate.astype(np.uint8))
        if diff.ndim == 3:
            diff = diff.max(axis=2)
        vote = cv2.threshold(diff, self.cfg.plate_diff_threshold, 255, cv2.THRESH_BINARY)[1]
        # A pixel never yet observed unoccupied has no trustworthy reference.
        vote[self._plate_count < self.MIN_PLATE_SAMPLES] = 0
        return vote

    def _accumulate(self, candidate: np.ndarray) -> None:
        """Leaky per-pixel integrator: evidence builds slowly, decays faster."""
        hits = candidate > 0
        self._score[hits] += 1.0
        self._score[~hits] -= 3.0
        np.clip(self._score, 0.0, self.score_on * 1.2, out=self._score)

    def _blobs_from_score(self, ctx: FrameContext, small: np.ndarray):
        """Confirmed-static regions that survive every gate, in full-res coords."""
        mask = (self._score >= self.score_on).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_area = float(mask.size)
        results = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            frac = area / frame_area
            if frac < self.cfg.min_area_frac or frac > self.cfg.max_area_frac:
                continue
            if not self._has_added_structure(small, x, y, w, h):
                continue

            box = (
                x * self._scale,
                y * self._scale,
                (x + w) * self._scale,
                (y + h) * self._scale,
            )
            if self._explained_by_a_track(box, ctx):
                continue
            cx, cy = centroids[i] * self._scale
            inside, zone = self._zone_for((cx, cy))
            if not inside:
                continue
            results.append((box, zone))
        return results

    def _has_added_structure(self, small: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """True if this region has more edge energy now than the plate remembers.

        A dropped object introduces contours. A pool of headlight glare does
        not, and MOG2's shadow suppression cannot reject it because it only
        recognises pixels that got darker.
        """
        if self._plate is None:
            return True
        patch = small[y : y + h, x : x + w]
        plate = self._plate[y : y + h, x : x + w].astype(np.uint8)
        if patch.size == 0 or plate.size == 0:
            return True
        return _edge_energy(patch) >= self.cfg.min_edge_ratio * _edge_energy(plate)

    def _explained_by_a_track(self, box, ctx: FrameContext) -> bool:
        return any(iou(box, t.bbox) > self.cfg.track_overlap for t in ctx.tracks)

    def _zone_for(self, point) -> tuple[bool, str | None]:
        if not self.zones:
            return True, None
        for zone in self.zones:
            if zone.contains(point):
                return True, zone.name
        return False, None

    @staticmethod
    def _cell_id(box) -> int:
        """Stable synthetic id from a quantised position.

        Debris does not move, so its location is a better identity than any
        frame-local index — and it keeps the incident engine's track-binding
        path working without inventing a fake tracker.
        """
        cx = int((box[0] + box[2]) / 2) // 16
        cy = int((box[1] + box[3]) / 2) // 16
        return -(1 + cy * 4096 + cx)

    def _to_event(self, update, ctx: FrameContext) -> Event:
        inc = update.incident
        kind = self.kind if update.action == "raise" else f"{self.kind}_clear"
        data: dict[str, Any] = {
            "incident_id": inc.id,
            "severity": self.cfg.severity if update.action == "raise" else "info",
            "zone": inc.zone,
            "duration_s": round(inc.duration_s(ctx.timestamp), 2),
            "bbox": [round(float(v), 1) for v in inc.bbox],
            # Debris is the noisiest channel in this system by a wide margin —
            # the road-anomaly literature is unanimous that it is harder than
            # stopped vehicles. It goes to a review queue, not to dispatch.
            "requires_human_review": True,
        }
        if update.action == "clear":
            data["reason"] = update.reason
        return Event(
            kind=kind,
            track_id=inc.id and min(inc.contributing_track_ids, default=-1),
            class_label=self.kind,
            frame_index=ctx.frame_index,
            timestamp=ctx.timestamp,
            data=data,
        )

    # ---- presentation -------------------------------------------------------
    def draw(self, frame: np.ndarray) -> None:
        for inc in self._incidents.open_incidents:
            x1, y1, x2, y2 = (int(v) for v in inc.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(
                frame,
                f"{self.kind.upper()}?",
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 215, 255),
                2,
            )

    def stats(self) -> dict:
        return {
            "debris": {
                "open": len(self._incidents.open_incidents),
                "warming": self.warming,
                "candidates": len(self._blobs),
            }
        }


def _edge_energy(patch: np.ndarray) -> float:
    """Mean Sobel gradient magnitude over a patch."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.abs(gx) + np.abs(gy))) + 1e-6
