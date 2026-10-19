"""UTC-offset serialization of timeline times (no TOAST needed).

TOAST's GroundSchedule parses ``start_time`` / ``stop_time`` with
``datetime.fromisoformat(...).timestamp()``; without an offset that is read
in the host's local timezone.
"""

import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from astropy.table import QTable
from astropy.time import Time, TimeDelta

from fyst_trajectories import get_fyst_site
from fyst_trajectories.overhead import (
    CalibrationPolicy,
    ObservingTimeline,
    OverheadModel,
    TimelineBlock,
    read_timeline,
    write_timeline,
)


def _timeline(scale="utc"):
    start = getattr(Time("2026-01-15T23:59:59.123456", scale="utc", precision=3), scale)
    stop = start + TimeDelta(2.345678, format="sec")
    block = TimelineBlock(
        t_start=start,
        t_stop=stop,
        block_type="science",
        patch_name="clock test",
        az_start=-5.0,
        az_end=5.0,
        elevation=50.0,
        scan_index=7,
        subscan_index=2,
    )
    return ObservingTimeline(
        blocks=[block],
        site=get_fyst_site(),
        start_time=start,
        end_time=stop,
        overhead_model=OverheadModel(),
        calibration_policy=CalibrationPolicy(),
    )


def _written_times(timeline, path):
    write_timeline(timeline, path)
    table = QTable.read(path)
    return table, [
        (table[0]["start_time"], timeline.start_time),
        (table[0]["stop_time"], timeline.end_time),
        (table.meta["timeline_start_time"], timeline.start_time),
        (table.meta["timeline_end_time"], timeline.end_time),
    ]


@pytest.mark.parametrize("scale", ["utc", "tai", "tt"])
def test_export_times_are_millisecond_utc_with_offset(tmp_path, scale):
    timeline = _timeline(scale)
    _, pairs = _written_times(timeline, tmp_path / "time.ecsv")
    for text, expected in pairs:
        # The digits are astropy's millisecond rendering, plus an explicit offset.
        assert text == Time(expected.utc, precision=3).isot + "+00:00"
        parsed = datetime.fromisoformat(text)
        assert parsed.utcoffset() == timedelta(0)
        assert abs(parsed.timestamp() - expected.utc.unix) <= 5e-4
    assert timeline.start_time.precision == 3  # input display precision untouched
    assert timeline.start_time.scale == scale
    restored = read_timeline(tmp_path / "time.ecsv")
    np.testing.assert_allclose(
        [
            restored.blocks[0].t_start.unix,
            restored.blocks[0].t_stop.unix,
            restored.start_time.unix,
            restored.end_time.unix,
        ],
        [timeline.start_time.unix, timeline.end_time.unix] * 2,
        rtol=0,
        atol=5e-4,
    )


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")
@pytest.mark.parametrize("host_tz", ["UTC", "America/New_York", "Asia/Shanghai"])
def test_written_times_parse_to_the_same_instant_in_any_host_timezone(
    tmp_path, monkeypatch, host_tz
):
    timeline = _timeline()
    _, pairs = _written_times(timeline, tmp_path / "time.ecsv")
    monkeypatch.setenv("TZ", host_tz)
    time.tzset()
    try:
        for text, expected in pairs:
            # What TOAST does with the string.
            assert abs(datetime.fromisoformat(text).timestamp() - expected.utc.unix) <= 5e-4
    finally:
        monkeypatch.undo()
        time.tzset()


@pytest.mark.parametrize("with_window", [True, False], ids=["window", "block_extents"])
@pytest.mark.parametrize(
    "style", ["legacy", "legacy_microseconds", "Z", "+00:00", "+05:30", "-04:00"]
)
def test_reader_accepts_old_and_offset_iso_times(tmp_path, style, with_window):
    timeline = _timeline()
    path = tmp_path / "external.ecsv"
    write_timeline(timeline, path)
    table = QTable.read(path)

    def render(time_):
        if style == "legacy":
            return time_.iso  # older files: a space, milliseconds, no offset
        instant = time_.to_datetime(timezone=timezone.utc)
        if style == "legacy_microseconds":
            return instant.replace(tzinfo=None).isoformat(sep=" ", timespec="microseconds")
        if style == "Z":
            return instant.isoformat(timespec="microseconds").replace("+00:00", "Z")
        offset = {
            "+00:00": timedelta(0),
            "+05:30": timedelta(hours=5, minutes=30),
            "-04:00": timedelta(hours=-4),
        }[style]
        return instant.astimezone(timezone(offset)).isoformat(timespec="microseconds")

    # Replace columns so longer offset strings cannot be truncated by the old dtype width.
    table["start_time"] = [render(timeline.start_time)]
    table["stop_time"] = [render(timeline.end_time)]
    if with_window:
        table.meta["timeline_start_time"] = render(timeline.start_time)
        table.meta["timeline_end_time"] = render(timeline.end_time)
    else:
        del table.meta["timeline_start_time"]
        del table.meta["timeline_end_time"]
    table.write(path, overwrite=True)
    restored = read_timeline(path)
    # Millisecond input can only preserve the precision present in the file.
    expected = [timeline.start_time, timeline.end_time]
    if style == "legacy":
        expected = [Time(t.iso, scale="utc") for t in expected]
    actual = [
        restored.blocks[0].t_start,
        restored.blocks[0].t_stop,
        restored.start_time,
        restored.end_time,
    ]
    assert all(t.scale == "utc" for t in actual)
    np.testing.assert_allclose(
        [t.unix for t in actual], [t.unix for t in expected] * 2, rtol=0, atol=1e-6
    )


def test_utc_suffix_reads_like_the_offset_free_string(tmp_path):
    """``+00:00`` takes the same astropy parse as before, so read-backs are unchanged."""
    timeline = _timeline()
    path = tmp_path / "time.ecsv"
    table, _ = _written_times(timeline, path)
    with_offset = read_timeline(path)
    table["start_time"] = [str(table[0]["start_time"]).removesuffix("+00:00")]
    table["stop_time"] = [str(table[0]["stop_time"]).removesuffix("+00:00")]
    table.write(path, overwrite=True)
    without = read_timeline(path)
    for a, b in (
        (with_offset.blocks[0].t_start, without.blocks[0].t_start),
        (with_offset.blocks[0].t_stop, without.blocks[0].t_stop),
    ):
        assert (a.jd1, a.jd2) == (b.jd1, b.jd2)


def test_empty_timeline_placeholder_and_window_carry_the_offset(tmp_path):
    timeline = replace(_timeline("tai"), blocks=[])
    path = tmp_path / "empty.ecsv"
    write_timeline(timeline, path)
    table = QTable.read(path)
    assert len(table) == 1
    assert table.meta["timeline_is_empty"] is True
    for column in ("start_time", "stop_time"):
        assert table[column][0] == "2000-01-01T00:00:00.000+00:00"
    for key, expected in (
        ("timeline_start_time", timeline.start_time),
        ("timeline_end_time", timeline.end_time),
    ):
        assert table.meta[key] == Time(expected.utc, precision=3).isot + "+00:00"
    restored = read_timeline(path)
    assert restored.blocks == []
    np.testing.assert_allclose(
        [restored.start_time.unix, restored.end_time.unix],
        [timeline.start_time.unix, timeline.end_time.unix],
        rtol=0,
        atol=5e-4,
    )
