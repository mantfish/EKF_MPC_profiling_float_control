import json
import logging
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine
from scipy.interpolate import RegularGridInterpolator

from helpers import *

CMEMS_DATASET_ID = "cmems_mod_bal_phy_anfc_PT1H-i_202411"
CMEMS_DEPTH_MAX  = 200.0
CMEMS_TIME_PRIOR = 12
KM_PER_DEG_LAT = 111.32

logger = logging.getLogger(__name__)

MAX_DRIFT = 100 # max drift in km expected float can undergo


def _float_dir(float_id: int) -> Path:
    return Path(f"float_{float_id}")


def read_json(file_path) -> dict:
    "Reads and parses an arbitrary JSON file"
    with open(file_path, "r") as f:
        return json.load(f)


def read_config(float_id: int) -> Config:
    "Opens config file"

    config_path = _float_dir(float_id) / "config.json"

    try:
        config_json = read_json(config_path)
    except Exception as e:
        logger.error("Error reading config file: %s", e)
        raise

    config = Config(
        float_id=config_json["float_id"],
        target_lat=config_json["target_lat"],
        target_lon=config_json["target_lon"],
        radius_std=config_json["radius_std"],
        flow_time_horizon_hours=config_json["flow_time_horizon_hours"],
        flow_weight=config_json["flow_weight"],
        distance_weight=config_json["distance_weight"],
        science_weight=config_json["science_weight"],
        variance_weight=config_json["variance_weight"],
        vertical_dt=config_json["vertical_dt"],
        parking_dt=config_json["parking_dt"],
        bathymetry_file_name=config_json["bathymetry_file_name"],
        estimated_tranmission_time_hours=config_json["estimated_tranmission_time_hours"],
        ascent_speed_m_per_s=config_json["ascent_speed_m_per_s"],
        descent_speed_m_per_s=config_json["descent_speed_m_per_s"],
        possible_actions=config_json["possible_actions"],
        data_dir=_float_dir(float_id),
        model=config_json.get("model", "CMEMS"),
        Q=np.array(config_json["process_noise_diagonal"]),
    )
    return config


def download_cmems_data_around_float(location: Location, earliest_time, latest_time) -> xr.Dataset:

    bbox = define_region_aroung_float(location)
    start_datetime = pd.Timestamp(earliest_time) - pd.Timedelta(hours=CMEMS_TIME_PRIOR)

    kwargs: dict = dict(
        dataset_id=CMEMS_DATASET_ID,
        variables=["uo", "vo"],
        minimum_latitude=bbox.latitude_min,
        maximum_latitude=bbox.latitude_max,
        minimum_longitude=bbox.longitude_min,
        maximum_longitude=bbox.longitude_max,
        minimum_depth=0.52,
        maximum_depth=CMEMS_DEPTH_MAX,
        start_datetime=start_datetime.isoformat(),
        end_datetime=pd.Timestamp(latest_time).isoformat(),
        output_filename="cmems_subset.nc",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        kwargs["output_directory"] = tmp_dir
        tmp_path = os.path.join(tmp_dir, "cmems_subset.nc")

        copernicusmarine.subset(**kwargs)
        logger.info("CMEMS subset downloaded on %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        ds = xr.open_dataset(tmp_path)
        ds.load()  # materialize into memory before tmp_dir (and the file in it) is deleted

    first_time = ds.time[0].values
    last_time = ds.time[-1].values

    logger.info("CMEMS subset time range: %s to %s", first_time, last_time)
    logger.info("CMEMS subset lat range: %s to %s", ds.latitude.min().values, ds.latitude.max().values)

    return ds


def read_last_surfacing_and_action(float_id: int) -> dict:
    surfacing_and_action_log_path = _float_dir(float_id) / "surfacing_action_log.json"

    if not surfacing_and_action_log_path.exists():
        raise FileNotFoundError(f"Surfacing and action log file does not exist for float ID: {float_id}")

    surfacing_and_action_log = read_json(surfacing_and_action_log_path)

    return surfacing_and_action_log[-1]  # Get the last surfacing


def _serialize_state(state: EstimatedState) -> dict:
    return {
        "time": state.time.isoformat() if hasattr(state.time, "isoformat") else state.time,
        "location": {
            "latitude": state.location.latitude,
            "longitude": state.location.longitude,
        },
        "depth": state.depth,
        "phase": state.phase,
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "bx": state.bx,
        "by": state.by,
        "P": state.P.tolist(),
    }


def write_surfacing_action_log(
    float_id: int,
    action_name: str,
    surfaced_timestamp,
    surfaced_location: Location,
    estimated_state: EstimatedState,
    real_state: EstimatedState,
    nis: float,
    actions_cost: dict,
) -> None:
    """Appends one surfacing/action entry to surfacing_log.json.

    Schema matches float_123456/surfacing_action_log.json: estimated_state is the
    predicted state before the Kalman innovation step, real_state is the corrected
    state after it, actions_cost is the {action_name: cost} map evaluated when
    choosing action_sent.
    """
    log_path = _float_dir(float_id) / "surfacing_action_log.json"

    if log_path.exists():
        surfacing_and_action_log = read_json(log_path)
    else:
        surfacing_and_action_log = []

    surfacing_and_action_log.append({
        "surfaced_timestamp": surfaced_timestamp.isoformat() if hasattr(surfaced_timestamp, "isoformat") else surfaced_timestamp,
        "surfaced_location": {
            "latitude": surfaced_location.latitude,
            "longitude": surfaced_location.longitude,
        },
        "action_sent": action_name,
        "estimated_state": _serialize_state(estimated_state),
        "real_state": _serialize_state(real_state),
        "nis": nis,
        "actions_cost": actions_cost,
    })

    with open(log_path, "w") as f:
        json.dump(surfacing_and_action_log, f, indent=4)

    logger.info("Surfacing action log updated for float ID: %s", float_id)


def define_region_aroung_float(location: Location) -> Region:
    """
    Define a region around the float's location based on the maximum drift expected."""
    start_location = location
    lat_min = start_location.latitude - MAX_DRIFT / KM_PER_DEG_LAT
    lon_min = start_location.longitude - MAX_DRIFT / (KM_PER_DEG_LAT * np.cos(np.radians(start_location.latitude)))
    lat_max = start_location.latitude + MAX_DRIFT / KM_PER_DEG_LAT
    lon_max = start_location.longitude + MAX_DRIFT / (KM_PER_DEG_LAT * np.cos(np.radians(start_location.latitude)))
    return Region(
        latitude_min=lat_min,
        latitude_max=lat_max,
        longitude_min=lon_min,
        longitude_max=lon_max
    )

def read_state(float_id: int) -> pd.DataFrame:

    state_file_path = _float_dir(float_id) / "estimated_state.parquet"
    if not state_file_path.exists():
        logger.error("State file does not exist for float ID: %s", float_id)
        raise FileNotFoundError(f"State file does not exist for float ID: {float_id}")

    return pd.read_parquet(state_file_path)

def write_state(float_id: int, state_df: pd.DataFrame) -> None:
    state_file_path = _float_dir(float_id) / "estimated_state.parquet"
    state_df.to_parquet(state_file_path, index=False)
    logger.info("Estimated state history written to file for float ID: %s", float_id)

def get_action(action_name: str, config: Config) -> ControlAction:
    for action in config.possible_actions:
        if action["name"] == action_name:
            return ControlAction(
                parking_depth=action["depth_m"],
                duration_hours=action["duration_hours"],
                science_cost=action["science_cost"],
            )
    raise ValueError(f"Action '{action_name}' not found in possible actions.")

def load_bathymetry(bathymetry_path) -> xr.Dataset:
    "Loads a GEBCO-style bathymetry NetCDF file (see build_bathymetry_interpolator)."
    return xr.open_dataset(bathymetry_path)

def build_bathymetry_interpolator(bathy_ds: xr.Dataset):
    """Build a fast scipy interpolator for seabed depth from the GEBCO dataset.

    Returns a callable ``f(lat, lon) -> depth_m`` that is orders of magnitude
    faster than calling xarray interp on every timestep.

    Parameters
    ----------
    bathy_ds:
        Dataset from :func:`load_bathymetry`.

    Returns
    -------
    callable
        ``f(lat, lon) -> float`` — seabed depth in metres, positive down.
        Returns ``0.0`` for land (positive GEBCO elevation).
    """
    lats = bathy_ds.lat.values.astype(np.float64)
    lons = bathy_ds.lon.values.astype(np.float64)
    elev = bathy_ds["elevation"].values.astype(np.float64)  # (lat, lon)

    interp = RegularGridInterpolator(
        (lats, lons), elev, method="linear", bounds_error=False, fill_value=np.nan,
    )

    def query(lat: float, lon: float) -> float:
        depth = float(interp([[lat, lon]])[0]) * -1.0
        return max(depth, 0.0)

    return query

def xy_to_latlon(x: float, y: float, start_lat: float, start_lon: float) -> tuple[float, float]:
    lat = start_lat + y / 111_000.0
    lon = start_lon + x / (111_000.0 * math.cos(math.radians(start_lat)))
    return lat, lon

def latlon_to_xy(lat: float, lon: float, start_lat: float, start_lon: float) -> tuple[float, float]:
    x = (lon - start_lon) * 111_000.0 * math.cos(math.radians(start_lat))
    y = (lat - start_lat) * 111_000.0
    return x, y


def build_uv_interpolators(model_ds):
    """
    Builds (u_interp, v_interp, bounds) from a CMEMS-style xarray Dataset
    with dims (time, depth, latitude, longitude).
    """
    times = model_ds["time"].values.astype("datetime64[s]").astype(np.float64)
    depths = model_ds["depth"].values
    lats = model_ds["latitude"].values
    lons = model_ds["longitude"].values

    u_interp = RegularGridInterpolator(
        (times, depths, lats, lons),
        model_ds["uo"].values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    v_interp = RegularGridInterpolator(
        (times, depths, lats, lons),
        model_ds["vo"].values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    bounds = {
        "t_min": times.min(), "t_max": times.max(),
        "z_min": depths.min(), "z_max": depths.max(),
        "lat_min": lats.min(), "lat_max": lats.max(),
        "lon_min": lons.min(), "lon_max": lons.max(),
    }

    return u_interp, v_interp, bounds


def query_uv(u_interp, v_interp, bounds, t, z, lat, lon):
    """
    Queries (u, v) at a single (t, z, lat, lon) point via linear
    interpolation, clamping to grid bounds and zeroing NaNs.
    """
    t_s = np.datetime64(t, "s").astype(np.float64)

    t_c = np.clip(t_s, bounds["t_min"], bounds["t_max"])
    z_c = np.clip(z, bounds["z_min"], bounds["z_max"])
    lat_c = np.clip(lat, bounds["lat_min"], bounds["lat_max"])
    lon_c = np.clip(lon, bounds["lon_min"], bounds["lon_max"])

    point = [[t_c, z_c, lat_c, lon_c]]
    u = float(u_interp(point)[0])
    v = float(v_interp(point)[0])

    if math.isnan(u):
        u = 0.0
    if math.isnan(v):
        v = 0.0

    return u, v
