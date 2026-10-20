from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path

from argopy import ArgoFloat
import numpy as np
import pandas as pd

from helpers import Config, ControlAction, EstimatedState, Location, OU, latlon_to_xy

logger = logging.getLogger(__name__)


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "floats"))
LOG_FILES = ["action_log.json", "surfacing_log.json", "error_log.csv", "estimated_trajectory.csv", "predicted_surfacings.csv"]


def config_from_dict(config_json: dict, float_dir: Path) -> Config:
    """Builds a Config from config.json contents, raises KeyError if a required field is missing."""
    return Config(
        float_id=config_json["float_id"],
        target_lat=config_json["target_lat"],
        target_lon=config_json["target_lon"],
        vertical_dt=config_json["vertical_dt"],
        parking_dt=config_json["parking_dt"],
        bathymetry_file_name=config_json["bathymetry_file_name"],
        delay_before_diving_hours=config_json.get("delay_before_diving_mins", 0) / 60,
        ascent_speed_m_per_s=config_json["ascent_speed_m_per_s"],
        descent_speed_m_per_s=config_json["descent_speed_m_per_s"],
        possible_actions=[
            ControlAction(drifting_depth=action["depth_m"], duration_hours=action["duration_hours"])
            for action in config_json["possible_actions"]
        ],
        data_dir=float_dir,
        model_type=config_json.get("model_type", "CMEMS"),
        dataset_id=config_json.get("model_id", "cmems_mod_bal_phy_anfc_PT1H-i_202411"),
        max_drift=config_json.get("max_drift", 30.0),
        buffer_time=config_json.get("buffer_time", 6.0),
        max_time_difference=config_json.get("max_time_difference", 30.0),
    )


def ou_from_json(ou_json: dict) -> OU:
    """ou.json keeps tau in hours, OU works in seconds."""
    return OU(ou_json["tau_e"] * 3600, ou_json["tau_n"] * 3600, ou_json["q_e"], ou_json["q_n"])


def ou_to_json(ou: OU) -> dict:
    return {"tau_e": ou.tau_e / 3600, "tau_n": ou.tau_n / 3600, "q_e": ou.q_e, "q_n": ou.q_n}


class Records:
    def __init__(self, float_id):
        self.float_id = float_id
        self.float_dir = DATA_ROOT / str(float_id)
        if not self.float_dir.exists():
            raise FileNotFoundError(f"Float {float_id} not initialised yet, no folder at {self.float_dir}")
        self.action_log_path = self.float_dir / "action_log.json"
        self.surfacing_log_path = self.float_dir / "surfacing_log.json"
        self.ou_path = self.float_dir / "ou.json"
        self.ou_log_path = self.float_dir / "ou.log"
        self.error_log_path = self.float_dir / "error_log.csv"
        self.trajectory_path = self.float_dir / "estimated_trajectory.csv"
        self.prediction_log_path = self.float_dir / "predicted_surfacings.csv"
        self.real_trajectory_path = self.float_dir / f"{float_id}_Rtraj.nc"
        self.config = self.read_config()

    @classmethod
    def create(cls, float_id, config: dict, rewrite=False) -> "Records":
        """Creates the float folder and config.json. With rewrite, an existing float's logs are deleted."""
        float_dir = DATA_ROOT / str(float_id)
        config_from_dict(config, float_dir)
        if float_dir.exists():
            if not rewrite:
                raise FileExistsError(f"Float {float_id} already exists, pass rewrite=True to start it again")
            for file_name in LOG_FILES:
                (float_dir / file_name).unlink(missing_ok=True)
            logger.warning("Rewriting float %s, previous logs deleted", float_id)

        float_dir.mkdir(parents=True, exist_ok=True)
        with open(float_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        return cls(float_id)

    def _read_json(self, file_path):
        with open(file_path) as f:
            return json.load(f)

    def read_config(self) -> Config:
        return config_from_dict(self._read_json(self.float_dir / "config.json"), self.float_dir)

    def save_config(self, config: dict):
        """Replaces config.json, only after checking the new config can be read."""
        if config.get("float_id") != self.float_id:
            raise ValueError(f"Config float_id {config.get('float_id')} does not match float {self.float_id}")
        self.config = config_from_dict(config, self.float_dir)
        with open(self.float_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        logger.info("Updated config for float %s", self.float_id)

    # Action log: every action sent to the float, keyed by the surfacing cycle it is executed after

    def get_action(self, cycle_number) -> ControlAction:
        # Latest entry wins, so a re-run of get_action or a manual override replaces earlier choices
        for entry in reversed(self._read_log(self.action_log_path)):
            if entry["cycle_number"] == cycle_number:
                return ControlAction(entry["drifting_depth"], entry["duration_hours"])
        raise ValueError(f"No action found for float {self.float_id} cycle {cycle_number}")

    def has_action(self, cycle_number) -> bool:
        return any(entry["cycle_number"] == cycle_number for entry in self._read_log(self.action_log_path))

    def save_action(self, action, cycle_number, source="mpc"):
        entry = {"cycle_number": cycle_number, "time_sent": datetime.now().isoformat(), "source": source, **asdict(action)}
        self._append_log(self.action_log_path, entry)
        logger.info("Saved action for float %s cycle %s: %s", self.float_id, cycle_number, action)

    # Surfacing log: real GPS surfacings, bias and P are taken from the trajectory history

    def get_last_cycle_number(self) -> int | None:
        return max((entry["cycle_number"] for entry in self._read_log(self.surfacing_log_path)), default=None)

    def get_surfacing_entry(self, cycle_number) -> dict:
        return next(e for e in self._read_log(self.surfacing_log_path) if e["cycle_number"] == cycle_number)

    def get_surfacing(self, cycle_number) -> EstimatedState:
        entry = self.get_surfacing_entry(cycle_number)
        state = EstimatedState(
            time=datetime.fromisoformat(entry["time"]),
            location=Location(entry["latitude"], entry["longitude"]),
            depth=0.0,
            phase="communicating",
        )

        if self.trajectory_path.exists():
            trajectory = pd.read_csv(self.trajectory_path)
            cycle_rows = trajectory[trajectory["cycle_number"] == cycle_number]
            if not cycle_rows.empty:
                last = cycle_rows.iloc[-1]
                state.bx, state.by = last["bx"], last["by"]
                state.P = np.array(json.loads(last["P"]))

        return state

    def save_surfacing(self, state, cycle_number):
        entry = {
            "cycle_number": cycle_number,
            "time": state.time.isoformat(),
            "latitude": state.location.latitude,
            "longitude": state.location.longitude,
        }
        self._append_log(self.surfacing_log_path, entry)
        logger.info("Saved surfacing for float %s cycle %s", self.float_id, cycle_number)

    # Estimated trajectory: one copy per cycle, the latest prediction or the real replay ending with its KF updated state

    def save_trajectory(self, states, cycle_number):
        rows = [{
            "cycle_number": cycle_number,
            "time": state.time.isoformat(),
            "latitude": state.location.latitude,
            "longitude": state.location.longitude,
            "depth": state.depth,
            "phase": state.phase,
            "bx": state.bx,
            "by": state.by,
            "P": json.dumps(state.P.tolist()),
        } for state in states]
        trajectory = pd.DataFrame(rows)
        if self.trajectory_path.exists():
            previous = pd.read_csv(self.trajectory_path)
            trajectory = pd.concat([previous[previous["cycle_number"] != cycle_number], trajectory])
        trajectory.to_csv(self.trajectory_path, index=False)

    # Prediction log: every predicted surfacing, to see how it changes over a cycle

    def save_prediction(self, state, cycle_number):
        row = {
            "run_time": datetime.now().isoformat(),
            "cycle_number": cycle_number,
            "predicted_time": state.time.isoformat(),
            "latitude": state.location.latitude,
            "longitude": state.location.longitude,
            "position_std_m": math.sqrt((state.P[0, 0] + state.P[1, 1]) / 2),
        }
        self._append_csv(self.prediction_log_path, [row])
        logger.info("Cycle %s predicted surfacing at %s %s, std %.0f m", cycle_number, state.time, state.location, row["position_std_m"])

    # Error log: one row per cycle comparing the estimated and real surfacing

    def update_errors(self, cycle_number, estimated, real, nis, ou: OU):
        error_east, error_north = latlon_to_xy(real.location.latitude, real.location.longitude, estimated.location.latitude, estimated.location.longitude)
        row = {
            "cycle_number": cycle_number,
            "real_time": real.time.isoformat(),
            "real_latitude": real.location.latitude,
            "real_longitude": real.location.longitude,
            "estimated_time": estimated.time.isoformat(),
            "estimated_latitude": estimated.location.latitude,
            "estimated_longitude": estimated.location.longitude,
            "error_east_m": error_east,
            "error_north_m": error_north,
            "error_m": math.hypot(error_east, error_north),
            "nis": nis,
            "bx": estimated.bx,
            "by": estimated.by,
            "P": json.dumps(estimated.P.tolist()),
            **asdict(ou),
        }
        self._append_csv(self.error_log_path, [row])
        logger.info("Cycle %s surfacing error %.0f m", cycle_number, row["error_m"])

    # Real trajectory: the float's Rtraj.nc from the Argo GDAC

    def update_real_trajectory(self):
        ds = ArgoFloat(self.float_id).open_dataset("Rtraj", cast=False)
        ds.to_netcdf(self.real_trajectory_path)
        logger.info("Downloaded real trajectory for float %s", self.float_id)

    def get_errors_and_bias(self) -> str:
        """CSV of each cycle's surfacing error (east, north) and bias right before and after the KF update."""
        if not self.error_log_path.exists():
            return ""
        errors = pd.read_csv(self.error_log_path)
        errors = errors[["cycle_number", "real_time", "error_east_m", "error_north_m", "bx", "by"]]
        errors = errors.rename(columns={"real_time": "time", "bx": "bx_before", "by": "by_before"})

        # The last trajectory row of each cycle is its KF updated state
        trajectory = pd.read_csv(self.trajectory_path, usecols=["cycle_number", "bx", "by"])
        updated = trajectory.groupby("cycle_number").last().rename(columns={"bx": "bx_after", "by": "by_after"})
        return errors.merge(updated, on="cycle_number", how="left").to_csv(index=False)

    def _read_log(self, path) -> list[dict]:
        return self._read_json(path) if path.exists() else []

    def _append_log(self, path, entry):
        log = self._read_log(path)
        log.append(entry)
        with open(path, "w") as f:
            json.dump(log, f, indent=4)

    def _append_csv(self, path, rows):
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)

    # OU: ou.json holds the bias model parameters in use, with tau in hours, ou.log records every update

    def get_ou(self) -> OU:
        """Reads ou.json, converting tau from hours to the seconds OU works in."""
        return ou_from_json(self._read_json(self.ou_path))

    def write_ou(self, ou: OU, cycle_number):
        """Replaces ou.json and appends the new values to ou.log."""
        ou_json = ou_to_json(ou)
        with open(self.ou_path, "w") as f:
            json.dump(ou_json, f, indent=4)
        values = f"tau_e {ou_json['tau_e']} h, tau_n {ou_json['tau_n']} h, q_e {ou_json['q_e']}, q_n {ou_json['q_n']}"
        with open(self.ou_log_path, "a") as f:
            f.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} cycle {cycle_number}: {values}\n")
        logger.info("Updated OU for float %s: %s", self.float_id, values)

