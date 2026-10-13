"""The visits example turns a planned night's ECSV into one JSON record per visit.

It runs beside the hour-long ``short_night`` fixture rather than in
``tests/test_examples.py`` because that is where a planned night already
exists; ``test_examples.py`` still lists the example as exercised. The
grouping itself is pinned in process on a timeline built by hand, since the
planned night has only single-pass visits and no escape slew.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from astropy.time import Time, TimeDelta

from fyst_trajectories.overhead import (
    CalibrationPolicy,
    DeferralReason,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
    dispatch_sheet,
    write_timeline,
)
from fyst_trajectories.overhead.calibration_night.reporting import _UNFORWARDED_KEYS

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / "examples" / "planet_night_visits.py"

_PASS_ROW = re.compile(r"source_scan on \w+: .*; slack (\d+) s; elapsed")
_PASS_PAIR = re.compile(r"^\s+scan_params=(\{.*\})  scheduled_t0_unix=([\d.]+)$")
_PASS_NOTE = re.compile(r"^\s+note: confirm the execution layer forwards (.+)$")


def _load_example():
    spec = importlib.util.spec_from_file_location("planet_night_visits", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def visits_record(short_night, example_env, tmp_path_factory):
    """Run the example on the short night's ECSV and load what it wrote."""
    _, timeline = short_night
    directory = tmp_path_factory.mktemp("visits")
    ecsv = directory / "night.ecsv"
    out = directory / "visits.json"
    write_timeline(timeline, ecsv)
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), str(ecsv), str(out)],
        cwd=REPO_ROOT,
        env=example_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Visits:" in result.stdout
    return json.loads(out.read_text(encoding="utf-8"))


def _sheet_passes(sheet):
    """Return the sheet's pass rows in order: (slack, params json, t0 text, note keys)."""
    lines = sheet.splitlines()
    rows = []
    for n, line in enumerate(lines):
        row = _PASS_ROW.search(line)
        if row is None:
            continue
        pair = _PASS_PAIR.match(lines[n + 1])
        assert pair is not None, lines[n + 1]
        note = _PASS_NOTE.match(lines[n + 2]) if n + 2 < len(lines) else None
        keys = note.group(1).split(", ") if note else []
        rows.append((int(row.group(1)), pair.group(1), pair.group(2), keys))
    return rows


def test_every_pass_matches_its_sheet_row(short_night, visits_record):
    """Each pass record repeats what the sheet prints for it, in order."""
    _, timeline = short_night
    passes = [p for visit in visits_record["visits"] for p in visit["passes"]]
    rows = _sheet_passes(dispatch_sheet(timeline))
    assert len(rows) >= 1
    for record, (slack, params_json, t0_text, keys) in zip(passes, rows, strict=True):
        assert json.dumps(record["scan_params"], sort_keys=True) == params_json
        assert f"{record['scheduled_t0_unix']:.3f}" == t0_text
        assert f"{record['slack_s']:.0f}" == str(slack)
        assert record["unforwarded_keys"] == keys == ["az_speed", "eta_offset_deg"]


def test_visits_group_the_operations_before_their_pass(visits_record):
    """A visit with a pass names its body and finishes its operations before the pass."""
    with_passes = [visit for visit in visits_record["visits"] if visit["passes"]]
    assert with_passes
    for visit in with_passes:
        assert visit["body"] == visit["passes"][0]["scan_params"]["body"]
        operations = [op["operation"] for op in visit["detector_operations"]]
        assert "retune" in operations
        last_operation = visit["detector_operations"][-1]
        first_pass = visit["passes"][0]
        assert (
            last_operation["t_start_unix"] + last_operation["duration_s"]
            <= first_pass["scheduled_t0_unix"]
        )
    first = with_passes[0]
    assert [op["operation"] for op in first["detector_operations"]][:2] == [
        "find_detectors",
        "skydip",
    ]
    assert visits_record["targets"] == ["saturn", "uranus"]
    assert visits_record["usable_interval_utc"] is not None
    assert visits_record["lateness_rule"].startswith("the slack on a pass")


def test_unforwarded_keys_match_the_sheet_copy():
    """The example's key tuple is the sheet's set, which the contract test pins."""
    assert set(_load_example().UNFORWARDED_KEYS) == _UNFORWARDED_KEYS


def _hand_built_night(short_night):
    """Build a night the planner would not produce: a two-pass visit and two escape slews."""
    site, planned = short_night
    t0 = Time("2026-09-11T06:30:00", scale="utc")

    def at(seconds):
        return t0 + TimeDelta(seconds, format="sec")

    def pass_block(start, body, index, n_passes, scan_index):
        params = {"body": body, "el_bore": 50.0, "mode": "rising", "az_speed": 1.5}
        params.update(pass_index=index, n_passes=n_passes)
        return TimelineBlock.calibration(
            "planet_cal",
            at(start),
            600.0,
            330.0,
            50.0,
            site,
            scan_index,
            target=body,
            az_end=338.0,
            scan_params=params,
            t0_scan=at(start).isot,
            extra_metadata={"transition": {"wrap": 334.0}, "n_legs": 10, "science_fraction": 0.4},
        )

    waiting = str(DeferralReason.WAITING_FOR_PASS)
    blocks = [
        TimelineBlock.slew(at(0), 60.0, 180.0, 334.0, 50.0, site, 0, patch_name="slew_to_saturn"),
        TimelineBlock.calibration(
            "retune",
            at(60),
            300.0,
            334.0,
            50.0,
            site,
            0,
            extra_metadata={"operation": "find_detectors"},
        ),
        TimelineBlock.calibration("retune", at(360), 300.0, 334.0, 50.0, site, 0),
        TimelineBlock.idle(at(660), 30.0, 334.0, 50.0, site, 0, reason=waiting),
        pass_block(690, "saturn", 0, 2, 1),
        TimelineBlock.idle(at(1290), 120.0, 334.0, 50.0, site, 1, reason=waiting),
        pass_block(1410, "saturn", 1, 2, 1),
        TimelineBlock.idle(at(2010), 300.0, 334.0, 50.0, site, 2, reason="nothing_available"),
        TimelineBlock.slew(at(2310), 40.0, 334.0, 200.0, 40.0, site, 2, patch_name="sun_escape"),
        TimelineBlock.idle(at(2350), 300.0, 200.0, 40.0, site, 2, reason="nothing_available"),
        TimelineBlock.slew(
            at(2650), 60.0, 200.0, 100.0, 50.0, site, 3, patch_name="slew_to_uranus"
        ),
        TimelineBlock.calibration("retune", at(2710), 300.0, 100.0, 50.0, site, 3),
        pass_block(3010, "uranus", 0, 1, 3),
        TimelineBlock.slew(at(3610), 40.0, 100.0, 250.0, 45.0, site, 4, patch_name="sun_escape"),
    ]
    return ObservingTimeline(
        blocks,
        site,
        t0,
        at(3650),
        OverheadModel(),
        CalibrationPolicy(),
        metadata=dict(planned.metadata),
    )


def test_grouping_handles_multi_pass_visits_and_escapes(short_night):
    """A second pass joins its visit; escape slews with no pass form their own records."""
    record = _load_example().visits_from_timeline(_hand_built_night(short_night), source="hand")
    visits = record["visits"]
    assert [(v["body"], len(v["passes"])) for v in visits] == [
        ("saturn", 2),
        (None, 0),
        ("uranus", 1),
        (None, 0),
    ]
    saturn, escape, uranus, trailing = visits
    assert [s["name"] for s in saturn["slews"]] == ["slew_to_saturn"]
    assert [op["operation"] for op in saturn["detector_operations"]] == [
        "find_detectors",
        "retune",
    ]
    assert [p["slack_s"] for p in saturn["passes"]] == [30.0, 120.0]
    assert [p["scan_params"]["pass_index"] for p in saturn["passes"]] == [0, 1]
    assert saturn["passes"][0]["unforwarded_keys"] == ["az_speed"]
    assert saturn["passes"][0]["wrap_az"] == 334.0
    assert [s["name"] for s in escape["slews"]] == ["sun_escape"]
    assert escape["slews"][0]["el_to"] == 40.0
    assert escape["detector_operations"] == []
    assert uranus["passes"][0]["slack_s"] == 0.0
    assert [s["name"] for s in trailing["slews"]] == ["sun_escape"]
    assert [i["reason"] for i in record["idle"]] == ["nothing_available", "nothing_available"]
    assert record["source"] == "hand"
    assert record["night_start_utc"] == "2026-09-11 06:30:00"
    json.dumps(record)
