"""Onboards the real Baltic floats tracked by the sibling model_tracker project into this
server's EKF/MPC pipeline, each with a single possible_action mirroring the real dive cycle
model_tracker already estimated for it (park mode, depth, timing, speeds).

Run from anywhere; re-running is safe — floats that already exist under floats/<id>/ are skipped.
"""
import logging
import os
import sys
from pathlib import Path

import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent.parent
MODEL_TRACKER_DIR = SERVER_DIR.parent / "model_tracker"
FLOATS_META_PATH = MODEL_TRACKER_DIR / "data" / "store" / "floats_meta.parquet"
BATHYMETRY_PATH = MODEL_TRACKER_DIR / "data" / "store" / "D6_2024.nc"

sys.path.insert(0, str(SERVER_DIR))
os.chdir(SERVER_DIR)  # data_handler resolves floats/<id> relative to the cwd

from api import initialise_float_core
from data_handler import check_if_float_exists

logger = logging.getLogger(__name__)

# MPC tuning knobs that don't exist in model_tracker's data (it has no MPC/EKF concept at
# all — see model_tracker/CLAUDE.md). These barely matter here since every float has exactly
# one possible_action, so there's nothing to choose between; reused from the existing
# single-action floats (floats/3902607/config.json, floats/6990657/config.json) for consistency.
DEFAULT_MPC_WEIGHTS = dict(
    radius_std=10.0,
    flow_time_horizon_hours=6.0,
    flow_weight=1.0,
    distance_weight=1.0,
    science_weight=1.0,
    variance_weight=1.0,
    vertical_dt=60.0,
    parking_dt=600.0,
    max_drift=20.0,
    model_type="CMEMS",
    model_id="cmems_mod_bal_phy_anfc_PT1H-i",
    process_noise_diagonal=[4.0, 4.0, 6.944444444444444e-05, 6.944444444444444e-05],
)


def _build_config(row: pd.Series) -> dict:
    # model_tracker's cycle_hours is descent+parking only; the server's duration_hours
    # spans the whole action (descent+park+ascent), so add the ascent time back in.
    duration_hours = float(row.cycle_hours) + (float(row.target_depth) / float(row.ascent_speed_ms)) / 3600.0
    action_name = f"Park on bottom for {duration_hours:.1f} hours"

    return {
        "float_id": int(row.float_id),
        # model_tracker has no notion of a mission target — target_lat/lon here is really
        # the flat-earth coordinate origin the EKF works in, so use the float's own last
        # known real position to keep it centered on where the float actually operates.
        "target_lat": float(row.last_lat),
        "target_lon": float(row.last_lon),
        "estimated_tranmission_time_hours": float(row.transmission_duration_minutes) / 60.0,
        "ascent_speed_m_per_s": float(row.ascent_speed_ms),
        "descent_speed_m_per_s": float(row.descent_speed_ms),
        "bathymetry_file_name": "bathymetry.nc",
        "possible_actions": [{
            "name": action_name,
            "duration_hours": duration_hours,
            "depth_m": float(row.target_depth),
            "action_type": "park",
            "science_cost": 1.0,
        }],
        **DEFAULT_MPC_WEIGHTS,
    }


def _build_starting_state_and_action(row: pd.Series, action_name: str) -> dict:
    return {
        "time": row.last_time.isoformat(),
        "location": {"latitude": float(row.last_lat), "longitude": float(row.last_lon)},
        "depth": 0.0,
        "phase": "descending",
        "action_name": action_name,
    }


def onboard_all() -> None:
    floats_meta = pd.read_parquet(FLOATS_META_PATH)
    bathymetry_bytes = BATHYMETRY_PATH.read_bytes()

    for _, row in floats_meta.iterrows():
        float_id = int(row.float_id)
        if check_if_float_exists(float_id):
            logger.info("Float %s already exists, skipping.", float_id)
            continue

        config_json = _build_config(row)
        action_name = config_json["possible_actions"][0]["name"]
        starting_contents = _build_starting_state_and_action(row, action_name)

        logger.info("Onboarding float %s...", float_id)
        try:
            initialise_float_core(config_json, bathymetry_bytes, starting_contents)
            logger.info("Float %s onboarded successfully.", float_id)
        except Exception:
            logger.exception("Failed to onboard float %s, continuing with the rest.", float_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    onboard_all()
