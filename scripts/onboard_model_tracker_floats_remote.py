"""Initialises the real Baltic floats tracked by model_tracker against a LIVE deployed
instance of this server (e.g. the Render-hosted API), via real HTTP POST /initialise_float
calls. Companion to onboard_model_tracker_floats.py, which does the same thing in-process
against the local floats/ directory — this script reuses its config/starting-state builders
so both stay in sync, but drives them over HTTP with curl (multipart file upload) instead.

Each request uploads the ~267MB shared bathymetry file and triggers a live CMEMS download +
full simulation server-side before responding, so requests can take a long time — pass
--timeout generously and consider onboarding one float first with --float-id.
"""
import argparse
import json
import logging
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from onboard_model_tracker_floats import (
    BATHYMETRY_PATH,
    FLOATS_META_PATH,
    _build_config,
    _build_starting_state_and_action,
)

logger = logging.getLogger(__name__)


def _remote_float_exists(api_base: str, float_id: int) -> bool:
    try:
        with urllib.request.urlopen(f"{api_base}/visualize/{float_id}", timeout=30) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def _initialise_remote(api_base: str, config_json: dict, starting_contents: dict, bathymetry_path: Path, timeout_s: int) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "config.json"
        config_path.write_text(json.dumps(config_json))
        start_path = Path(tmp) / "starting_state_and_action.json"
        start_path.write_text(json.dumps(starting_contents))

        cmd = [
            "curl", "-sS", "-w", "\n__HTTP_STATUS__:%{http_code}",
            "--max-time", str(timeout_s),
            "-X", "POST", f"{api_base}/initialise_float",
            "-F", f"config_file=@{config_path};type=application/json;filename=config.json",
            "-F", f"bathymetry_file=@{bathymetry_path};type=application/octet-stream;filename=bathymetry.nc",
            "-F", f"starting_state_and_action=@{start_path};type=application/json;filename=starting_state_and_action.json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        if "__HTTP_STATUS__:" in output:
            body, status_str = output.rsplit("__HTTP_STATUS__:", 1)
            status = int(status_str.strip())
        else:
            body, status = result.stderr, -1
        return status, body.strip()


def onboard_remote(api_base: str, float_ids: list[int] | None, timeout_s: int) -> None:
    floats_meta = pd.read_parquet(FLOATS_META_PATH)
    if float_ids is not None:
        floats_meta = floats_meta[floats_meta["float_id"].astype(int).isin(float_ids)]

    bathymetry_mb = BATHYMETRY_PATH.stat().st_size / 1e6

    for _, row in floats_meta.iterrows():
        float_id = int(row.float_id)
        if _remote_float_exists(api_base, float_id):
            logger.info("Float %s already exists on %s, skipping.", float_id, api_base)
            continue

        config_json = _build_config(row)
        action_name = config_json["possible_actions"][0]["name"]
        starting_contents = _build_starting_state_and_action(row, action_name)

        logger.info(
            "Initialising float %s on %s (uploading %.0f MB bathymetry + live CMEMS run, this can take a while)...",
            float_id, api_base, bathymetry_mb,
        )
        status, body = _initialise_remote(api_base, config_json, starting_contents, BATHYMETRY_PATH, timeout_s)
        if status == 200:
            logger.info("Float %s initialised successfully on %s.", float_id, api_base)
        else:
            logger.error("Failed to initialise float %s on %s: HTTP %s - %s", float_id, api_base, status, body[:800])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="https://ekf-mpc-profiling-float-control.onrender.com")
    parser.add_argument("--float-id", type=int, action="append", dest="float_ids", default=None,
                         help="Limit to specific float_id(s); repeatable. Defaults to all floats in floats_meta.parquet.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-request curl timeout in seconds.")
    args = parser.parse_args()
    onboard_remote(args.api_base, args.float_ids, args.timeout)
