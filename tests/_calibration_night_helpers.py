"""Shared helpers of the calibration-night tests."""

from __future__ import annotations

import dataclasses

from fyst_trajectories.overhead import read_calibration_night_metadata
from fyst_trajectories.overhead.calibration_night.policy import (
    encode_calibration_night_metadata,
)


def _with_policy(timeline, *, drop=(), **changes):
    """Return the timeline with keys of its recorded policy changed or dropped."""
    meta = read_calibration_night_metadata(timeline)
    policy = {k: v for k, v in {**meta["policy"], **changes}.items() if k not in drop}
    return dataclasses.replace(
        timeline, metadata=encode_calibration_night_metadata({**meta, "policy": policy})
    )
