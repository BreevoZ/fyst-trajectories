"""Write a planned calibration night as one JSON record per visit (offline planning).

Reads the ECSV that ``examples/planet_night.py`` writes and writes
:func:`~fyst_trajectories.overhead.dispatch_records` of that night as JSON;
the function's documentation and the record types it returns state what
each key holds.

Run it from the repository root::

    python examples/planet_night.py 2026-09-10T22:00:00 2026-09-11T10:30:00 night
    python examples/planet_night_visits.py night/planet_night.ecsv night/visits.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fyst_trajectories.overhead import dispatch_records, read_timeline


def main(argv: list[str] | None = None) -> int:
    """Read the night's ECSV, write its dispatch records as JSON.

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
    record = dispatch_records(read_timeline(argv[0]), source=Path(argv[0]).name)
    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    passes = sum(len(visit["passes"]) for visit in record["visits"])
    print(f"Visits: {len(record['visits'])} records, {passes} passes, wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
