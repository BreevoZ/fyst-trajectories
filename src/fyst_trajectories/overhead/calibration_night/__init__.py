"""Plan one night of solar-system calibration passes back to back.

A commissioning night visits whichever planets are up, in a chosen
order, planning each visit with the source-CES kernel across the
footprint (the instrument team's per-body scan tables give a reference
throw and dwell the policy can apply instead), checking every slew for
Sun safety, and reserving the detector operations between visits. The
entry point is :func:`plan_calibration_night`; the four step functions
it loops over (:func:`list_candidates`, :func:`plan_visit`,
:func:`commit_visit`, :func:`advance_idle`) are public so a person can
plan, inspect, discard and re-plan a visit interactively. The result is
an ordinary
:class:`~fyst_trajectories.overhead.ObservingTimeline`, read back by
:func:`summarize_calibration_night`, :func:`dispatch_sheet` and
:func:`dispatch_records`.
"""

from .night import plan_calibration_night
from .policy import (
    CalibrationNightMetadata,
    CalibrationNightPolicy,
    ScanOverrides,
    TuningPolicy,
    read_calibration_night_metadata,
)
from .reporting import (
    BodySummary,
    IdleRecord,
    NightRecord,
    NightSummary,
    OperationRecord,
    PassRecord,
    SlewRecord,
    VisitRecord,
    dispatch_records,
    dispatch_sheet,
    summarize_calibration_night,
)
from .selection import ScriptedSelection, SelectionRule, select_priority
from .state import BOOTSTRAP_POSE, NightContext, NightState, VisitPlanner
from .step import (
    Candidate,
    VisitPlan,
    advance_idle,
    commit_visit,
    list_candidates,
    plan_visit,
)
from .tables import (
    DEFAULT_SCAN_TABLES,
    ElevationBin,
    ScanParameterTable,
    load_scan_tables,
)

__all__ = [
    "BOOTSTRAP_POSE",
    "BodySummary",
    "CalibrationNightMetadata",
    "CalibrationNightPolicy",
    "Candidate",
    "DEFAULT_SCAN_TABLES",
    "ElevationBin",
    "IdleRecord",
    "NightContext",
    "NightRecord",
    "NightState",
    "NightSummary",
    "OperationRecord",
    "PassRecord",
    "ScanOverrides",
    "ScanParameterTable",
    "ScriptedSelection",
    "SelectionRule",
    "SlewRecord",
    "TuningPolicy",
    "VisitPlan",
    "VisitPlanner",
    "VisitRecord",
    "advance_idle",
    "commit_visit",
    "dispatch_records",
    "dispatch_sheet",
    "list_candidates",
    "load_scan_tables",
    "plan_calibration_night",
    "plan_visit",
    "read_calibration_night_metadata",
    "select_priority",
    "summarize_calibration_night",
]
