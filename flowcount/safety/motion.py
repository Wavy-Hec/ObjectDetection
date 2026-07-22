"""Stillness metrics for tracked objects.

Deciding "has this object stopped?" from a tracked bounding box is harder than
it looks, and the obvious answers are all wrong:

- **Kalman velocity** (``Track.velocity`` / ``Track.get_speed()``) is the worst
  option. It is in px/*frame*, so its meaning changes with FPS and with
  ``detect_every``; it is non-zero by construction while a track coasts; and it
  is perspective-dependent.
- **Single-frame displacement** flips on detector jitter.
- **Raw pixel displacement over a window** is perspective-dependent: a vehicle
  far from the camera moves a handful of pixels for the same real speed that
  moves a near vehicle across the frame.

This module computes two independent, scale-invariant metrics over a sliding
time window and requires both. Their failure modes are largely uncorrelated:

1. **Self-IoU at a lag** — IoU of the current box against the box ``window_s``
   ago. A ratio of areas of the *same* object, so it is inherently scale- and
   perspective-invariant. Saturates for a very slow crawl, which is why it is
   paired with (2).
2. **Size-normalised radius** — the 90th-percentile distance of recent anchor
   points from their median, divided by the median object size. Apparent size
   goes as 1/depth and apparent displacement goes as real-displacement/depth,
   so the ratio is approximately ``real_displacement / real_size`` at any range
   in the frame. That cancels perspective with no calibration at all.

Anchors are the **bottom-centre** of the box, not the centroid: the
ground-contact point stays put when the top of the box is clipped or the object
is partially occluded, whereas the centroid moves whenever box height changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

BBox = tuple[float, float, float, float]


def iou(a, b) -> float:
    """Intersection over union of two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def anchor_of(bbox) -> tuple[float, float]:
    """Ground-contact point: bottom-centre of the box."""
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])


def size_of(bbox) -> float:
    """Characteristic size ``sqrt(w * h)`` — the normaliser for displacement."""
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return float(np.sqrt(w * h))


def transform_bbox(bbox, matrix) -> BBox:
    """Map a box through a 3x3 transform, returning the axis-aligned bounds.

    Used to express boxes in a stabilizer's reference frame so that camera sway
    does not read as object motion.
    """
    if matrix is None:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    corners = np.array(
        [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ],
        dtype=np.float64,
    )
    ones = np.ones((4, 1))
    mapped = (np.asarray(matrix, dtype=np.float64) @ np.hstack([corners, ones]).T).T
    w = mapped[:, 2:3]
    # A degenerate projective row would produce inf/nan; fall back to identity
    # rather than poisoning every downstream metric.
    if not np.all(np.isfinite(w)) or np.any(np.abs(w) < 1e-9):
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    pts = mapped[:, :2] / w
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


@dataclass(frozen=True)
class StillnessConfig:
    """Thresholds for the composite stillness predicate.

    Enter/exit pairs form a Schmitt trigger: once an object is judged still it
    takes a clearly larger movement to un-judge it. Without that hysteresis the
    state flaps on jitter and every downstream alert flaps with it.
    """

    window_s: float = 3.0
    iou_enter: float = 0.85
    iou_exit: float = 0.60
    radius_enter: float = 0.15
    radius_exit: float = 0.35
    #: Fraction of the window's *detection opportunities* that actually produced
    #: a detection for this track. Measured against frames where the detector
    #: ran, not against all frames — under ``detect_every=N`` only 1/N of frames
    #: carry a detection at all, so a whole-frame ratio could never exceed 1/N
    #: and would silently disable stall detection at exactly the frame-skipping
    #: settings an edge device needs.
    min_detection_support: float = 0.6
    min_height_px: float = 24.0
    min_confidence: float = 0.40
    #: Minimum detection-backed samples before the percentile statistics mean
    #: anything.
    min_samples: int = 5


@dataclass
class MotionSample:
    timestamp: float
    bbox: BBox
    confidence: float
    #: This track was matched to a real detection on this frame.
    detection_backed: bool
    #: The detector ran at all on this frame (False on frames skipped by
    #: ``detect_every``). Distinguishes "the detector looked and missed this
    #: object" from "the detector never looked".
    detector_ran: bool = True


@dataclass
class TrackMotionWindow:
    """Sliding time window of one track's boxes, in stabilized coordinates.

    Every metric is computed from **detection-backed samples only**. Coasted
    positions carry no evidence about whether the object moved — they are
    generated by assuming it kept moving — so feeding them into a stillness
    statistic measures the motion model rather than the world.
    """

    window_s: float = 3.0
    samples: deque = field(default_factory=deque)
    last_frame_index: int = 0

    def add(
        self,
        timestamp: float,
        bbox,
        confidence: float,
        detection_backed: bool,
        frame_index: int = 0,
        detector_ran: bool = True,
    ) -> None:
        self.samples.append(
            MotionSample(
                timestamp=timestamp,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                confidence=float(confidence),
                detection_backed=bool(detection_backed),
                detector_ran=bool(detector_ran),
            )
        )
        self.last_frame_index = frame_index
        cutoff = timestamp - self.window_s
        while len(self.samples) > 2 and self.samples[0].timestamp < cutoff:
            self.samples.popleft()

    # ---- derived quantities -------------------------------------------------
    @property
    def observed(self) -> list[MotionSample]:
        """Only the samples backed by an actual detection."""
        return [s for s in self.samples if s.detection_backed]

    @property
    def span_s(self) -> float:
        obs = self.observed
        if len(obs) < 2:
            return 0.0
        return obs[-1].timestamp - obs[0].timestamp

    @property
    def latest(self) -> MotionSample | None:
        return self.samples[-1] if self.samples else None

    @property
    def detection_support(self) -> float:
        """Observed samples over detection *opportunities* in the window."""
        opportunities = sum(s.detector_ran for s in self.samples)
        if not opportunities:
            return 0.0
        return sum(s.detection_backed for s in self.samples) / opportunities

    def self_iou(self) -> float:
        """IoU of the newest observed box against the oldest one in the window."""
        obs = self.observed
        if len(obs) < 2:
            return 1.0
        return iou(obs[-1].bbox, obs[0].bbox)

    def median_size(self) -> float:
        obs = self.observed
        if not obs:
            return 0.0
        return float(np.median([size_of(s.bbox) for s in obs]))

    def median_height(self) -> float:
        obs = self.observed
        if not obs:
            return 0.0
        return float(np.median([s.bbox[3] - s.bbox[1] for s in obs]))

    def mean_confidence(self) -> float:
        obs = self.observed
        if not obs:
            return 0.0
        return float(np.mean([s.confidence for s in obs]))

    def normalized_radius(self) -> float:
        """90th-percentile anchor spread divided by median object size.

        The percentile rather than the max, because a supremum statistic lets a
        single jitter frame veto a real stall; and rather than the mean, which
        is too permissive when an object drifts steadily.
        """
        obs = self.observed
        if len(obs) < 2:
            return 0.0
        pts = np.array([anchor_of(s.bbox) for s in obs], dtype=np.float64)
        centre = np.median(pts, axis=0)
        radii = np.linalg.norm(pts - centre, axis=1)
        size = self.median_size()
        if size <= 0:
            return float("inf")
        return float(np.percentile(radii, 90) / size)

    def displacement_from(self, point) -> float:
        """Distance of the newest observed anchor from ``point``, in size units."""
        obs = self.observed
        if not obs:
            return 0.0
        size = self.median_size()
        if size <= 0:
            return 0.0
        ax, ay = anchor_of(obs[-1].bbox)
        return float(np.hypot(ax - point[0], ay - point[1]) / size)


def evaluate_stillness(
    window: TrackMotionWindow, cfg: StillnessConfig, *, currently_still: bool
) -> bool | None:
    """Composite stillness predicate.

    Returns ``True`` (still), ``False`` (moving), or ``None`` meaning *not
    enough evidence to judge*. The three-valued result is the point: a caller
    must be able to freeze a stall timer during an occlusion rather than being
    forced to choose between "still" (which would let an alert fire off
    Kalman predictions) and "moving" (which would reset a nearly-mature timer).
    """
    if len(window.observed) < cfg.min_samples or window.span_s < cfg.window_s * 0.5:
        return None
    if window.detection_support < cfg.min_detection_support:
        return None

    self_iou = window.self_iou()
    radius = window.normalized_radius()

    if currently_still:
        # Hysteresis: stay still until clearly moving.
        return not (self_iou < cfg.iou_exit or radius > cfg.radius_exit)

    if window.median_height() < cfg.min_height_px:
        return None
    if window.mean_confidence() < cfg.min_confidence:
        return None
    return self_iou > cfg.iou_enter and radius < cfg.radius_enter
