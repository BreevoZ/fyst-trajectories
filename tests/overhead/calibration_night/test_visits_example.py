"""The visits example writes the dispatch records of a planned night's ECSV as JSON.

It runs beside the hour-long ``short_night`` fixture rather than in
``tests/test_examples.py`` because that is where a planned night already
exists; ``test_examples.py`` still lists the example as exercised. What the
records hold is pinned in ``test_dispatch_records.py``.
"""

import json
import subprocess
import sys
from pathlib import Path

from fyst_trajectories.overhead import dispatch_records, read_timeline, write_timeline

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / "examples" / "planet_night_visits.py"


def test_the_example_writes_the_records_of_the_night_it_reads(short_night, example_env, tmp_path):
    """The command exits 0, reports its counts and writes ``dispatch_records`` as JSON."""
    _, timeline = short_night
    ecsv = tmp_path / "night.ecsv"
    out = tmp_path / "records" / "visits.json"
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
    want = dispatch_records(read_timeline(ecsv), source=ecsv.name)
    passes = sum(len(visit["passes"]) for visit in want["visits"])
    assert passes == 3
    assert f"Visits: {len(want['visits'])} records, {passes} passes, wrote {out}" in result.stdout
    assert json.loads(out.read_text(encoding="utf-8")) == want
