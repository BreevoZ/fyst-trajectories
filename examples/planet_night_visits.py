"""Write a planned calibration night as one JSON record per visit (offline planning).

:func:`~fyst_trajectories.overhead.plan_calibration_night` records a night as
timeline blocks, and :func:`~fyst_trajectories.overhead.dispatch_sheet` renders
them as text. This example reads the night back from the ECSV the planner
wrote and writes the same content as data: one record per visit, holding the
slews, the detector operations reserved at the target pose in execution order,
and each pass with the literal relative ``scan_params`` dict and the
``scheduled_t0_unix`` that the execution layer's source-scan task takes, plus
the slack the plan left in front of the pass. Nothing is re-planned, so the
records match the dispatch sheet pass for pass.

Run it from the repository root::

    python examples/planet_night.py 2026-09-10T22:00:00 2026-09-11T10:30:00 night
    python examples/planet_night_visits.py night/planet_night.ecsv night/visits.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fyst_trajectories.overhead import (
    BlockType,
    DeferralReason,
    ObservingTimeline,
    TimelineBlock,
    read_calibration_night_metadata,
    read_timeline,
)

# The pass-dict keys the dispatch sheet's note names: keys that shape a pass
# but that the execution layer's source-scan task, at the revision this
# library is checked against, drops without a message.
UNFORWARDED_KEYS = ("az_speed", "az_throw", "dwell", "eta_offset_deg", "footprint_margin")

LATENESS_RULE = (
    "the slack on a pass is what the plan left in front of it; "
    "once its start has gone by, re-plan the next block from now"
)


def _is_pass(block: TimelineBlock) -> bool:
    return (
        block.block_type == BlockType.CALIBRATION and block.metadata.get("cal_type") == "planet_cal"
    )


def _is_waiting(block: TimelineBlock) -> bool:
    return block.block_type == BlockType.IDLE and block.metadata.get("reason") == str(
        DeferralReason.WAITING_FOR_PASS
    )


def _when(block: TimelineBlock) -> dict[str, Any]:
    return {
        "t_start_utc": block.t_start.iso[:19],
        "t_start_unix": round(float(block.t_start.unix), 3),
        "duration_s": round(float(block.duration), 3),
    }


def _slew(block: TimelineBlock) -> dict[str, Any]:
    return {
        "name": block.patch_name,
        **_when(block),
        "az_from": float(block.az_start),
        "az_to": float(block.az_end),
        "el_to": float(block.elevation),
    }


def _operation(block: TimelineBlock) -> dict[str, Any]:
    name = block.metadata.get("operation") or block.metadata.get("cal_type") or block.scan_type
    return {
        "operation": str(name),
        **_when(block),
        "az": float(block.az_start),
        "el": float(block.elevation),
    }


def _pass(block: TimelineBlock, previous: TimelineBlock | None) -> dict[str, Any]:
    params = dict(block.metadata.get("scan_params", {}))
    transition = block.metadata.get("transition", {})
    slack = previous.duration if previous is not None and _is_waiting(previous) else 0.0
    return {
        "t_start_utc": block.t_start.iso[:19],
        "scheduled_t0_unix": round(float(block.t_start.unix), 3),
        "duration_s": round(float(block.duration), 3),
        "slack_s": round(float(slack), 3),
        "mode": params.get("mode"),
        "el_bore": params.get("el_bore"),
        "wrap_az": float(transition.get("wrap", block.az_start)),
        "n_legs": block.metadata.get("n_legs"),
        "science_fraction": block.metadata.get("science_fraction"),
        "scan_params": params,
        "unforwarded_keys": [key for key in UNFORWARDED_KEYS if key in params],
    }


def _new_visit() -> dict[str, Any]:
    return {"body": None, "slews": [], "detector_operations": [], "passes": []}


def _has_moves(visit: dict[str, Any]) -> bool:
    return bool(visit["slews"] or visit["detector_operations"])


def visits_from_timeline(timeline: ObservingTimeline, source: str = "") -> dict[str, Any]:
    """Group a planned night's blocks into one record per visit.

    A visit is the slews and detector operations that lead up to a pass,
    together with that pass and, for a multi-pass visit, the passes that
    follow it. An escape slew the planner makes while the night idles has
    no pass behind it and forms a record with no passes, since the telescope
    still makes the move. Idle periods are listed separately with their
    reasons; the idle that waits for a crossing to open is folded into the
    pass it precedes as its slack.

    Parameters
    ----------
    timeline : ObservingTimeline
        A night from :func:`~fyst_trajectories.overhead.plan_calibration_night`,
        in memory or read back from ECSV (times to the millisecond).
    source : str, optional
        Recorded in the output as the name of the file the night was read from.

    Returns
    -------
    dict
        ``targets``, ``sun_policy``, ``usable_interval_utc``, ``lateness_rule``,
        ``night_start_utc`` and ``night_end_utc`` describe the night. ``visits``
        holds, in time order, records of ``body``, ``slews`` (``name``, start,
        ``duration_s``, ``az_from``, ``az_to``, ``el_to``), ``detector_operations``
        (``operation``, start, ``duration_s``, ``az``, ``el``) and ``passes``
        (``scheduled_t0_unix``, ``duration_s``, ``slack_s``, ``mode``, ``el_bore``,
        ``wrap_az``, ``n_legs``, ``science_fraction``, ``scan_params`` and the
        ``unforwarded_keys`` the dispatch sheet's note names). ``idle`` lists
        the other idle periods with their ``reason``.

    Raises
    ------
    KeyError
        If the timeline carries no calibration-night metadata.
    ValueError
        If the metadata's schema version is not supported.
    """
    meta = read_calibration_night_metadata(timeline)
    blocks = sorted(timeline.blocks, key=lambda b: b.t_start.unix)
    visits: list[dict[str, Any]] = []
    idle: list[dict[str, Any]] = []
    current = _new_visit()  # the visit being filled; open once it holds a pass
    pending = _new_visit()  # moves and operations seen since the last pass
    previous: TimelineBlock | None = None
    for block in blocks:
        if block.block_type == BlockType.SLEW:
            pending["slews"].append(_slew(block))
        elif _is_pass(block):
            params = block.metadata.get("scan_params", {})
            if current["passes"] and int(params.get("pass_index", 0)) > 0:
                # A later pass of the open visit. The planner puts only the
                # waiting idle between passes, so this merge is a guard for
                # a timeline built by hand.
                current["slews"].extend(pending["slews"])
                current["detector_operations"].extend(pending["detector_operations"])
            else:
                if current["passes"]:
                    visits.append(current)
                current = pending
                current["body"] = params.get("body")
            current["passes"].append(_pass(block, previous))
            pending = _new_visit()
        elif block.block_type == BlockType.CALIBRATION:
            pending["detector_operations"].append(_operation(block))
        elif not _is_waiting(block):
            idle.append({**_when(block), "reason": str(block.metadata.get("reason", "unlabelled"))})
            if _has_moves(pending):
                # Moves with no pass behind them: close the open visit first
                # so the records stay in time order.
                if current["passes"]:
                    visits.append(current)
                visits.append(pending)
                current = _new_visit()
                pending = _new_visit()
        previous = block
    if current["passes"]:
        visits.append(current)
    if _has_moves(pending):
        visits.append(pending)
    return {
        "source": source,
        "targets": list(meta["targets"]),
        "sun_policy": meta["sun_safe"],
        "usable_interval_utc": meta["usable_interval"],
        "lateness_rule": LATENESS_RULE,
        "unforwarded_keys_note": (
            "confirm the execution layer forwards the keys a pass lists under unforwarded_keys"
        ),
        "night_start_utc": blocks[0].t_start.iso[:19] if blocks else None,
        "night_end_utc": blocks[-1].t_stop.iso[:19] if blocks else None,
        "visits": visits,
        "idle": idle,
    }


def main(argv: list[str] | None = None) -> int:
    """Read the night's ECSV, write the visits as JSON.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments: the ECSV to read and the JSON file to write.

    Returns
    -------
    int
        Process exit status (0 on success, 2 on a usage error).
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(
            "usage: python examples/planet_night_visits.py night.ecsv visits.json", file=sys.stderr
        )
        return 2
    timeline = read_timeline(argv[0])
    record = visits_from_timeline(timeline, source=Path(argv[0]).name)
    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    passes = sum(len(visit["passes"]) for visit in record["visits"])
    print(f"Visits: {len(record['visits'])} records, {passes} passes, wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
