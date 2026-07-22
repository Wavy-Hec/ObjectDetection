"""Safety-monitoring mode: stuck objects, intrusions, debris, and alerting.

FlowCount's traffic analytics answer "how many, how fast, which way". This
subpackage answers "is something wrong right now" for a monitored zone — the
job an AI grade-crossing monitor does, generalized to any hazard area.

Everything here plugs into the existing pipeline through the ordinary
:class:`~flowcount.analytics.base.Analyzer` contract; nothing in the detection or
tracking path knows this package exists.
"""

from .incidents import ACTIVE, CANDIDATE, CLEARED, CLEARING, Incident, IncidentTracker, Observation
from .motion import StillnessConfig, TrackMotionWindow, evaluate_stillness
from .static_objects import DebrisConfig, StaticObjectMonitor
from .zone_incident import (
    PEDESTRIAN_CLASSES,
    VEHICLE_CLASSES,
    VULNERABLE_CLASSES,
    IncidentRule,
    ZoneIncidentDetector,
    intrusion_rule,
    stalled_rule,
)

__all__ = [
    "StillnessConfig",
    "DebrisConfig",
    "StaticObjectMonitor",
    "TrackMotionWindow",
    "evaluate_stillness",
    "Incident",
    "IncidentTracker",
    "Observation",
    "CANDIDATE",
    "ACTIVE",
    "CLEARING",
    "CLEARED",
    "IncidentRule",
    "ZoneIncidentDetector",
    "stalled_rule",
    "intrusion_rule",
    "VEHICLE_CLASSES",
    "PEDESTRIAN_CLASSES",
    "VULNERABLE_CLASSES",
]
