"""Backfills a float's history one cycle at a time from its real Argo trajectory (Rtraj.nc).

Run by hand, one cycle per run:  uv run python posthumous.py 3902607 [--debug]
"""
import json
import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr

from helpers import ControlAction, Location
from main import initialise_float, setup_logging, surface_trigger
from records import Records

logger = logging.getLogger(__name__)

GROUNDED_DEPTH = 10_000.0  # metres, deep enough to always hit the bottom
GROUNDED_MIN_HOURS = 3.0  # 3902607: parks on the bottom every cycle, its park pressure is ~2 m above the seabed

ASCENT_END_CODE = 600
SURFACE_FIX_CODE = 703
PARKING_CODE = 290


def decode(values):
    return np.array([v.decode().strip() if isinstance(v, bytes) else str(v).strip() for v in values])


def estimate_cycles(rtraj_path) -> dict[int, dict]:
    """Argo cycle number -> {"time", "location", "park_depth", "ascent_end"} of that cycle's surfacing and park."""
    with xr.open_dataset(rtraj_path) as ds:
        df = pd.DataFrame({
            "cycle": ds.CYCLE_NUMBER.values,
            "code": ds.MEASUREMENT_CODE.values,
            "time": ds.JULD.values,
            "pres": ds.PRES.values,
            "lat": ds.LATITUDE.values,
            "lon": ds.LONGITUDE.values,
            "qc": decode(ds.POSITION_QC.values),
            "accuracy": decode(ds.POSITION_ACCURACY.values),
        })

    cycles = {}
    for cycle_number, rows in df[df.cycle >= 0].groupby("cycle"):
        # First good GPS fix, the Iridium ('I') positions are km-scale
        fixes = rows[(rows.code == SURFACE_FIX_CODE) & (rows.accuracy == "G") & (rows.qc == "1")].dropna(subset=["time", "lat", "lon"])
        if fixes.empty:
            logger.warning("Cycle %d has no good GPS fix, skipping", cycle_number)
            continue
        fix = fixes.sort_values("time").iloc[0]

        park_pres = rows[rows.code == PARKING_CODE].pres.dropna()
        park_depth = park_pres.mean() if not park_pres.empty else rows.pres.max()

        ascent_end = rows[rows.code == ASCENT_END_CODE].time.dropna()

        cycles[int(cycle_number)] = {
            "time": fix.time.floor("s").to_pydatetime(),
            "ascent_end": ascent_end.min().floor("s").to_pydatetime() if not ascent_end.empty else None,
            "location": Location(float(fix.lat), float(fix.lon)),
            "park_depth": 0.0 if np.isnan(park_depth) else float(park_depth),
        }
    return cycles


def estimate_action(cycles, cycle_number) -> ControlAction:
    """The action carried out after surfacing cycle_number, from Argo cycle cycle_number + 1.

    It lasts until the float is back on the surface, surface_trigger then drifts it on the surface until the GPS fix.
    """
    this_cycle, next_cycle = cycles[cycle_number], cycles[cycle_number + 1]
    surfaced = next_cycle["ascent_end"]
    if surfaced is None:
        logger.warning("Cycle %d has no ascent end time, using its GPS fix time", cycle_number + 1)
        surfaced = next_cycle["time"]
    duration_hours = (surfaced - this_cycle["time"]).total_seconds() / 3600

    durations = [(cycles[c + 1]["time"] - cycles[c]["time"]).total_seconds() / 3600 for c in cycles if c + 1 in cycles]
    if duration_hours > 2 * np.median(durations):
        logger.warning("Cycle %d lasted %.1f h, more than twice the usual %.1f h, possibly a missed cycle", cycle_number, duration_hours, np.median(durations))

    drifting_depth = GROUNDED_DEPTH if duration_hours > GROUNDED_MIN_HOURS else next_cycle["park_depth"]
    return ControlAction(drifting_depth=drifting_depth, duration_hours=duration_hours)


def backfill_next_cycle(float_id, debug=False):
    records = Records(float_id)
    records.update_real_trajectory()
    cycles = estimate_cycles(records.real_trajectory_path)

    last = records.get_last_cycle_number()
    if last is None:
        first = min(cycles)
        with open(records.float_dir / "config.json") as f:
            config = json.load(f)
        initialise_float(float_id, config, cycles[first]["location"], cycles[first]["time"], estimate_action(cycles, first), first, rewrite=True)
        return

    if last + 1 not in cycles:
        logger.info("Float %s is caught up at cycle %d", float_id, last)
        return

    # surface_trigger reads the action from the log, so it is saved even in debug
    if not records.has_action(last):
        records.save_action(estimate_action(cycles, last), last, source="rtraj")

    next_cycle = cycles[last + 1]
    logger.info("Backfilling float %s cycle %d, surfaced at %s %s", float_id, last + 1, next_cycle["time"], next_cycle["location"])
    surface_trigger(float_id, next_cycle["location"], next_cycle["time"], last + 1, debug)


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    setup_logging(debug)
    backfill_next_cycle(int(sys.argv[1]), debug)
