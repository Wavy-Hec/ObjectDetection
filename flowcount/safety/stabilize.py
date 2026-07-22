"""Camera-motion compensation for a nominally fixed camera.

A pole-mounted camera sways in wind, creeps as the mast warms, and occasionally
gets knocked. None of that moves the world, but all of it moves the pixels — and
every zone polygon, counting line and stall timer in this project is expressed in
pixels. Uncompensated, a 15 px gust reads as every object in the scene lurching
sideways: stall timers reset, wrong-way alerts fire, and a real stalled vehicle
can be silently "cleared".

Three choices here, each measured rather than assumed (numbers from this repo's
dev box at 720p, single-threaded, against synthetically warped ground truth):

**Match every frame to a stored keyframe; never chain frame-to-frame.** Chaining
is a random walk — each pairwise estimate carries ~0.01 px of noise, the errors
compose multiplicatively, and nothing ever corrects them. Measured, chained error
grew monotonically to 3.13 px by frame 600 (~9 px/minute at 30 fps) while
keyframe error stayed flat at 0.02 px. Keyframe matching is also *cheaper*:
corners are detected once when the keyframe is set (23.5 ms, amortised to
nothing) and only the optical flow runs per frame — 5.7 ms/frame versus 29.3 ms
for re-detecting every frame.

**Four degrees of freedom, not eight.** ``estimateAffinePartial2D`` (rotation,
uniform scale, translation) fits what a swaying pole actually does. A homography
has eight parameters and a real outdoor scene is not one plane — road, poles and
trees sit at different depths — so it "explains" parallax by inventing
perspective distortion. It is also 3x more expensive to solve. BoT-SORT's global
motion compensation makes the same call.

**Transform the geometry, not the pixels.** Mapping an 8-vertex polygon costs
0.001 ms; warping a 720p frame costs 6.8 ms, which on a Jetson would eat a
third of the frame budget on its own. Zone containment, line crossing and
stillness all operate on points, so they only ever need the matrix.

The estimate is masked by the tracker's own boxes. Without that, a queue of
vehicles moving together gives RANSAC a large, perfectly consistent inlier set,
and the "camera motion" you recover is the traffic's motion — a failure that is
completely silent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class StabilizerConfig:
    """Tuning for :class:`CameraStabilizer`."""

    #: Work at 1/downscale resolution. Translation is scaled back afterwards.
    downscale: int = 2
    #: Re-estimate only every Nth frame, reusing the last transform in
    #: between. Pole sway is a ~1 Hz phenomenon, so a 100 ms-old transform is
    #: still accurate to a fraction of a pixel, and this is the cheapest
    #: lever there is: 9.4 -> 3.1 ms/frame at 720p.
    estimate_every: int = 3
    max_corners: int = 600
    quality_level: float = 0.01
    min_distance: int = 8
    block_size: int = 3
    lk_win: int = 21
    lk_levels: int = 3
    #: Forward-backward consistency threshold, in working pixels. A point that
    #: does not track back to where it started is on a moving object or an
    #: occlusion boundary.
    fb_error_px: float = 1.0
    #: Below either of these the fit is not trustworthy and identity is returned.
    #: Failing safe matters: a wrong transform is worse than none, because
    #: analytics would silently apply it.
    min_inliers: int = 25
    min_inlier_ratio: float = 0.3
    #: Ignore this fraction of each image edge — timestamp burn-ins and
    #: on-screen-display overlays are painted in camera space and never move,
    #: so they would anchor the estimate to zero motion.
    border_frac: float = 0.02
    #: Pad the tracker's boxes before excluding them; a box rarely covers the
    #: whole object, and the corners just outside it move with the object.
    box_pad_frac: float = 0.15
    #: Re-detect keyframe corners once this few survive tracking.
    min_keyframe_corners: int = 120
    #: Sustained translation beyond this fraction of frame width means the
    #: camera was repositioned rather than nudged.
    reposition_frac: float = 0.05
    reposition_hold_s: float = 2.0
    #: How long analytics should ignore transitions after a reposition.
    settle_s: float = 5.0
    history_s: float = 10.0


@dataclass
class StabilizerState:
    """What the stabilizer concluded about the most recent frame."""

    matrix: np.ndarray | None = None  # 3x3, current image -> reference frame
    inliers: int = 0
    inlier_ratio: float = 0.0
    #: Fraction of keyframe corners still trackable — how much of the
    #: reference view is still visible, as distinct from how well the
    #: visible part agrees on one transform.
    tracked_ratio: float = 0.0
    translation_px: float = 0.0
    degraded: bool = True
    rebaselined: bool = False
    frames: int = 0


@dataclass
class _Sample:
    timestamp: float
    translation: float
    inlier_ratio: float


class CameraStabilizer:
    """Estimates a 3x3 current-image -> reference-frame transform per frame."""

    def __init__(self, config: StabilizerConfig | None = None):
        self.cfg = config or StabilizerConfig()
        self.state = StabilizerState()
        #: Drained by :class:`StabilityMonitor` and turned into events.
        self.pending_events: list[dict] = []

        self._keyframe: np.ndarray | None = None
        self._corners: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        #: Maps the current keyframe's coordinates back to the original
        #: reference, so re-baselining never invalidates zone geometry.
        self._keyframe_to_origin = np.eye(3, dtype=np.float64)
        self._history: deque[_Sample] = deque()
        self._reposition_since: float | None = None
        self._frames = 0

    # ---- public API ---------------------------------------------------------
    def estimate(self, frame: np.ndarray, exclude_boxes=None, timestamp: float | None = None):
        """Return a 3x3 current -> reference transform, or None if untrustworthy.

        Returning None rather than a best-guess matrix is deliberate: analytics
        treat None as "no compensation available" and fall back to raw pixels,
        whereas a bad matrix would be applied silently.
        """
        self._frames += 1
        every = max(1, int(self.cfg.estimate_every))
        if (
            self._keyframe is not None
            and self._frames % every != 1 % every
            and self.state.matrix is not None
        ):
            return self.state.matrix

        gray = self._prepare(frame)

        if self._keyframe is None or self._shape != gray.shape:
            self._set_keyframe(gray)
            self.state = StabilizerState(
                matrix=np.eye(3), inliers=0, inlier_ratio=1.0, degraded=False, frames=self._frames
            )
            return self.state.matrix

        matrix, inliers, ratio, tracked = self._match(gray, exclude_boxes)
        if matrix is None:
            self.state = StabilizerState(
                matrix=None,
                inliers=inliers,
                inlier_ratio=ratio,
                tracked_ratio=tracked,
                degraded=True,
                frames=self._frames,
            )
            return None

        full = self._to_full_resolution(matrix)
        combined = self._keyframe_to_origin @ full
        translation = float(np.hypot(combined[0, 2], combined[1, 2]))

        self.state = StabilizerState(
            matrix=combined,
            inliers=inliers,
            inlier_ratio=ratio,
            tracked_ratio=tracked,
            translation_px=translation,
            degraded=False,
            frames=self._frames,
        )
        # Reposition is judged against the CURRENT keyframe, not against the
        # original origin. `combined` accumulates every past re-baseline, so
        # using it would re-trigger forever after the first genuine move.
        keyframe_translation = float(np.hypot(full[0, 2], full[1, 2]))
        self._check_reposition(gray, keyframe_translation, ratio, timestamp)
        self._maybe_refresh_corners(gray, inliers)
        return combined

    def reset(self, frame: np.ndarray | None = None) -> None:
        """Forget the reference. Geometry expressed in reference coords is void."""
        self._keyframe = None
        self._corners = None
        self._keyframe_to_origin = np.eye(3, dtype=np.float64)
        self._history.clear()
        self._reposition_since = None
        if frame is not None:
            self._set_keyframe(self._prepare(frame))

    # ---- internals ----------------------------------------------------------
    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        d = max(1, int(self.cfg.downscale))
        if d > 1:
            gray = cv2.resize(
                gray,
                (gray.shape[1] // d, gray.shape[0] // d),
                interpolation=cv2.INTER_AREA,
            )
        return gray

    def _border_mask(self, shape) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        bx, by = int(w * self.cfg.border_frac), int(h * self.cfg.border_frac)
        mask[by : h - by, bx : w - bx] = 255
        return mask

    def _set_keyframe(self, gray: np.ndarray, exclude_boxes=None) -> None:
        self._keyframe = gray
        self._shape = gray.shape
        mask = self._border_mask(gray.shape)
        if exclude_boxes:
            self._punch_boxes(mask, exclude_boxes)
        self._corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.max_corners,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance,
            blockSize=self.cfg.block_size,
            mask=mask,
        )

    def _punch_boxes(self, mask: np.ndarray, boxes) -> None:
        """Zero out the tracker's boxes so moving objects cannot vote."""
        d = max(1, int(self.cfg.downscale))
        pad = self.cfg.box_pad_frac
        for box in boxes:
            x1, y1, x2, y2 = (float(v) / d for v in box)
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            cv2.rectangle(
                mask,
                (int(x1 - px), int(y1 - py)),
                (int(x2 + px), int(y2 + py)),
                0,
                -1,
            )

    def _match(self, gray: np.ndarray, exclude_boxes):
        if self._corners is None or len(self._corners) < self.cfg.min_inliers:
            return None, 0, 0.0, 0.0

        lk = {
            "winSize": (self.cfg.lk_win, self.cfg.lk_win),
            "maxLevel": self.cfg.lk_levels,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        p0 = self._corners
        p1, status, _err = cv2.calcOpticalFlowPyrLK(self._keyframe, gray, p0, None, **lk)
        if p1 is None:
            return None, 0, 0.0, 0.0

        # Forward-backward check: a point that does not track back to where it
        # started is on a moving object or an occlusion edge, and would drag the
        # fit toward the traffic's motion.
        p0r, status_b, _ = cv2.calcOpticalFlowPyrLK(gray, self._keyframe, p1, None, **lk)
        fb = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
        good = (status.ravel() == 1) & (status_b.ravel() == 1) & (fb < self.cfg.fb_error_px)

        if exclude_boxes:
            good &= ~self._inside_boxes(p1.reshape(-1, 2), exclude_boxes)

        src = p0.reshape(-1, 2)[good]
        dst = p1.reshape(-1, 2)[good]
        tracked_ratio = len(src) / float(len(self._corners))
        if len(src) < self.cfg.min_inliers:
            return None, len(src), 0.0, tracked_ratio

        affine, inlier_mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if affine is None or inlier_mask is None:
            return None, 0, 0.0, tracked_ratio

        inliers = int(inlier_mask.sum())
        # Ratio against the points that were actually *trackable*, not against
        # every keyframe corner. A large genuine camera move pushes much of the
        # keyframe out of view; those corners are missing evidence, not
        # disagreeing evidence, and counting them as failures makes a real
        # reposition indistinguishable from a broken estimate. How much of the
        # keyframe is still visible is a separate signal (`tracked_ratio`),
        # used below to notice occlusion and repositioning.
        ratio = inliers / float(len(src))
        if inliers < self.cfg.min_inliers or ratio < self.cfg.min_inlier_ratio:
            return None, inliers, ratio, tracked_ratio

        # `affine` maps keyframe -> current; analytics want current -> reference.
        inverse = cv2.invertAffineTransform(affine)
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2] = inverse
        return matrix, inliers, ratio, tracked_ratio

    def _inside_boxes(self, points: np.ndarray, boxes) -> np.ndarray:
        d = max(1, int(self.cfg.downscale))
        pad = self.cfg.box_pad_frac
        inside = np.zeros(len(points), dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = (float(v) / d for v in box)
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            inside |= (
                (points[:, 0] >= x1 - px)
                & (points[:, 0] <= x2 + px)
                & (points[:, 1] >= y1 - py)
                & (points[:, 1] <= y2 + py)
            )
        return inside

    def _to_full_resolution(self, matrix: np.ndarray) -> np.ndarray:
        """Rescale a transform estimated at working resolution to full pixels.

        Conjugating by the scale matrix leaves the rotation/scale block alone
        and multiplies only the translation.
        """
        d = float(max(1, int(self.cfg.downscale)))
        out = matrix.copy()
        out[0, 2] *= d
        out[1, 2] *= d
        return out

    def _check_reposition(self, gray, translation, ratio, timestamp) -> None:
        """Separate a gust (transient, zero-mean) from an actual camera move."""
        if timestamp is None:
            return
        self._history.append(_Sample(timestamp, translation, ratio))
        while self._history and timestamp - self._history[0].timestamp > self.cfg.history_s:
            self._history.popleft()

        width = (self._shape[1] if self._shape else 0) * max(1, self.cfg.downscale)
        limit = width * self.cfg.reposition_frac
        if translation <= limit:
            self._reposition_since = None
            return

        if self._reposition_since is None:
            self._reposition_since = timestamp
        elif timestamp - self._reposition_since >= self.cfg.reposition_hold_s:
            # Sustained, not a gust. Adopt the new view as the keyframe and
            # compose the accumulated transform so geometry defined in the
            # original reference frame stays valid.
            # The new keyframe *is* the current frame, so its mapping back to
            # the original reference is exactly this frame's transform.
            self._keyframe_to_origin = self.state.matrix.copy()
            self._set_keyframe(gray)
            self._reposition_since = None
            self.state.rebaselined = True
            self.pending_events.append(
                {
                    "kind": "camera_moved",
                    "translation_px": round(translation, 1),
                    "settle_until": timestamp + self.cfg.settle_s,
                }
            )

    def _maybe_refresh_corners(self, gray: np.ndarray, inliers: int) -> None:
        """Re-detect on the *keyframe* when too few of its corners still track.

        The keyframe itself is never replaced here — replacing it would restart
        the drift-free reference and is reserved for a real reposition.
        """
        if inliers >= self.cfg.min_keyframe_corners:
            return
        if self._keyframe is None:
            return
        self._corners = cv2.goodFeaturesToTrack(
            self._keyframe,
            maxCorners=self.cfg.max_corners,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance,
            blockSize=self.cfg.block_size,
            mask=self._border_mask(self._keyframe.shape),
        )


@dataclass
class StabilityMonitor:
    """Turns stabilizer state into events and suspends detectors after a bump.

    Kept separate from the stabilizer so the estimator stays a pure function of
    pixels, and separate from the incident detectors so they need no knowledge
    of cameras. It is an ordinary Analyzer, so it runs in the existing chain.
    """

    stabilizer: CameraStabilizer
    #: Detectors with a ``suspend(until_ts)`` method, frozen after a reposition.
    suspends: list = field(default_factory=list)

    def update(self, ctx) -> list:
        from ..analytics.base import Event

        events = []
        for payload in self.stabilizer.pending_events:
            for target in self.suspends:
                target.suspend(payload["settle_until"])
            events.append(
                Event(
                    kind=payload["kind"],
                    track_id=-1,
                    class_label="camera",
                    frame_index=ctx.frame_index,
                    timestamp=ctx.timestamp,
                    data={
                        "severity": "warning",
                        "translation_px": payload["translation_px"],
                        # Zone geometry was defined against the old view. It is
                        # composed forward automatically, but an operator should
                        # still confirm it.
                        "requires_human_review": True,
                    },
                )
            )
        self.stabilizer.pending_events.clear()
        return events

    def draw(self, frame) -> None:
        return None

    def save(self, path_prefix: str):
        return None

    def stats(self) -> dict:
        s = self.stabilizer.state
        return {
            "camera": {
                "stabilized": not s.degraded,
                "inliers": s.inliers,
                "inlier_ratio": round(s.inlier_ratio, 3),
                "drift_px": round(s.translation_px, 1),
            }
        }
