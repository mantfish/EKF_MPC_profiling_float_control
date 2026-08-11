import ast
import json
import logging
import math
import os
import resource
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine
from scipy.interpolate import RegularGridInterpolator

from helpers import Config, ControlAction, EstimatedState, Location, Region

import os

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "floats"))


CMEMS_DEPTH_MAX  = 200.0
CMEMS_TIME_PRIOR = 12
KM_PER_DEG_LAT = 111.32

logger = logging.getLogger(__name__)

MAX_DRIFT = 100 # max drift in km expected float can undergo


def _log_memory(stage: str) -> None:
    "Logs the process's peak resident set size so far (Linux: ru_maxrss is in KB), to locate where memory goes on memory-capped instances."
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info("[memory] %s: peak RSS so far = %.1f MB", stage, peak_rss_kb / 1024)


def _float_dir(float_id: int) -> Path:
    return DATA_ROOT / str(float_id)

def check_if_float_exists(float_id: int) -> bool:
    return _float_dir(float_id).is_dir()


def read_json(file_path: str | Path) -> Any:
    "Reads and parses an arbitrary JSON file"
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            raise TypeError("The file is not in .json format")


def _resolve_within_data_root(file_name: str) -> Path:
    "Resolves file_name under DATA_ROOT, rejecting any path that escapes it (e.g. via '..')."
    data_root = DATA_ROOT.resolve()
    resolved = (data_root / file_name).resolve()
    if resolved != data_root and data_root not in resolved.parents:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid file_name {file_name}.",
        )
    return resolved

def config_from_dict(config_json: dict, float_id: int) -> Config:
    return Config(
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
        model_type=config_json.get("model_type", "CMEMS"),
        dataset_id=config_json.get("model_id", "cmems_mod_bal_phy_anfc_PT1H-i_202411"),
        max_drift=config_json.get("max_drift", MAX_DRIFT),
        Q=np.array(config_json["process_noise_diagonal"]),
    )


def read_config(float_id: int) -> Config:
    "Opens config file"
    config_path = _float_dir(float_id) / "config.json"
    try:
        config_json = read_json(config_path)
    except Exception as e:
        logger.error("Error reading config file: %s", e)
        raise
    return config_from_dict(config_json, float_id)


def _cmems_credentials() -> tuple[str, str]:
    username = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Copernicus Marine credentials are not configured. Set the "
            "COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD "
            "environment variables (register for free at "
            "https://data.marine.copernicus.eu/register)."
        )
    return username, password


def download_cmems_data_around_float(
    location: Location, earliest_time: datetime, latest_time: datetime, config: Config
) -> xr.Dataset:

    username, password = _cmems_credentials()
    if config.max_drift:
        bbox = define_region_aroung_float(location, config.max_drift)
    else:
        bbox = define_region_aroung_float(location, MAX_DRIFT)

    start_datetime = pd.Timestamp(earliest_time) - pd.Timedelta(hours=CMEMS_TIME_PRIOR)

    kwargs: dict = dict(
        dataset_id=config.dataset_id,
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
        username=username,
        password=password,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        kwargs["output_directory"] = tmp_dir
        tmp_path = os.path.join(tmp_dir, "cmems_subset.nc")

        _log_memory("before copernicusmarine.subset()")
        copernicusmarine.subset(**kwargs)
        _log_memory("after copernicusmarine.subset() (output file written, not yet opened)")

        if not os.path.exists(tmp_path):
            raise RuntimeError(
                "CMEMS subset download did not produce an output file. Check that "
                "the Copernicus Marine credentials are valid and the toolbox logs "
                "above for the underlying error."
            )

        logger.info("CMEMS subset downloaded on %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("CMEMS subset file size: %.1f MB", os.path.getsize(tmp_path) / (1024 * 1024))

        ds = xr.open_dataset(tmp_path)
        ds.load()  # materialize into memory before tmp_dir (and the file in it) is deleted
        _log_memory("after ds.load() into memory")

        # CMEMS's packed-int16-with-scale-factor storage gets CF-decoded to float64
        # by default, which is far more precision than an ocean current velocity
        # needs. Downcast to float32 to roughly halve this dataset's memory footprint.
        for var in ("uo", "vo"):
            ds[var] = ds[var].astype(np.float32, copy=False)
        _log_memory("after downcasting uo/vo to float32")

    first_time = ds.time[0].values
    last_time = ds.time[-1].values

    logger.info("CMEMS subset time range: %s to %s", first_time, last_time)
    logger.info("CMEMS subset lat range: %s to %s", ds.latitude.min().values, ds.latitude.max().values)

    return ds


def read_surfacing_and_action(float_id: int) -> dict:
    surfacing_and_action_log_path = _float_dir(float_id) / "surfacing_action_log.json"

    if not surfacing_and_action_log_path.exists():
        raise FileNotFoundError(f"Surfacing and action log file does not exist for float ID: {float_id}")

    surfacing_and_action_log = read_json(surfacing_and_action_log_path)

    return surfacing_and_action_log  # Get the last surfacing


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
    surfaced_timestamp: datetime,
    surfaced_location: Location,
    estimated_state: EstimatedState,
    real_state: EstimatedState,
    nis: float,
    innovation: np.array,
    actions_cost: dict,
) -> None:
    """Appends one surfacing/action entry to surfacing_log.json.

    Schema matches floats/123456/surfacing_action_log.json: estimated_state is the
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
        "innovation": innovation.tolist(),
        "actions_cost": actions_cost,
    })

    with open(log_path, "w") as f:
        json.dump(surfacing_and_action_log, f, indent=4)

    logger.info("Surfacing action log updated for float ID: %s", float_id)


def define_region_aroung_float(location: Location, max_drift: float) -> Region:
    """
    Define a region around the float's location based on the maximum drift expected."""
    start_location = location
    lat_min = start_location.latitude - max_drift / KM_PER_DEG_LAT
    lon_min = start_location.longitude - max_drift / (KM_PER_DEG_LAT * np.cos(np.radians(start_location.latitude)))
    lat_max = start_location.latitude + max_drift / KM_PER_DEG_LAT
    lon_max = start_location.longitude + max_drift / (KM_PER_DEG_LAT * np.cos(np.radians(start_location.latitude)))
    return Region(
        latitude_min=lat_min,
        latitude_max=lat_max,
        longitude_min=lon_min,
        longitude_max=lon_max
    )

def read_state(float_id: int) -> pd.DataFrame:

    state_file_path = _float_dir(float_id) / "estimated_state.csv"
    if not state_file_path.exists():
        logger.info("No state file yet for float ID: %s. Returning empty history.", float_id)
        return pd.DataFrame()

    state_df = pd.read_csv(state_file_path, parse_dates=["time"])
    state_df["P"] = state_df["P"].apply(ast.literal_eval)
    return state_df

def write_state(float_id: int, state_df: pd.DataFrame) -> None:
    state_file_path = _float_dir(float_id) / "estimated_state.csv"
    state_df.to_csv(state_file_path, index=False)
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

def load_bathymetry(bathymetry_path: str | Path) -> xr.Dataset:
    "Loads a GEBCO-style bathymetry NetCDF file (see build_bathymetry_interpolator)."
    return xr.open_dataset(bathymetry_path)

def build_bathymetry_interpolator(bathy_ds: xr.Dataset, bbox: Region | None = None) -> Callable[[float, float], float]:
    """Build a fast scipy interpolator for seabed depth from the GEBCO dataset.

    Returns a callable ``f(lat, lon) -> depth_m`` that is orders of magnitude
    faster than calling xarray interp on every timestep.

    Parameters
    ----------
    bathy_ds:
        Dataset from :func:`load_bathymetry`.
    bbox:
        If given, crops to this region before materializing. Bathymetry files
        are basin-wide (tens of millions of grid points); without this, the
        whole grid gets pulled into memory even though only a small area
        around the float is ever queried.

    Returns
    -------
    callable
        ``f(lat, lon) -> float`` — seabed depth in metres, positive down.
        Returns ``0.0`` for land (positive GEBCO elevation).
    """
    if bbox is not None:
        bathy_ds = bathy_ds.sel(
            lat=slice(bbox.latitude_min, bbox.latitude_max),
            lon=slice(bbox.longitude_min, bbox.longitude_max),
        )

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


def build_uv_interpolators(
    model_ds: xr.Dataset,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator, dict[str, float]]:
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


def query_uv(
    u_interp: RegularGridInterpolator,
    v_interp: RegularGridInterpolator,
    bounds: dict[str, float],
    t: datetime,
    z: float,
    lat: float,
    lon: float,
) -> tuple[float, float]:
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

def state_to_row(state: EstimatedState) -> dict:
    "Flattens an EstimatedState into a parquet-friendly row (see read_state/write_state)."
    return {
        "time": state.time,
        "latitude": state.location.latitude,
        "longitude": state.location.longitude,
        "depth": state.depth,
        "phase": state.phase,
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "bx": state.bx,
        "by": state.by,
        "Q_x": state.Q[0],
        "Q_y": state.Q[1],
        "Q_bx": state.Q[2],
        "Q_by": state.Q[3],
        "P": state.P.tolist(),
    }

def row_to_state(row: pd.Series) -> EstimatedState:
    "Inverse of _state_row: builds an EstimatedState from a single state-history row."
    return EstimatedState(
        time=row["time"],
        location=Location(latitude=row["latitude"], longitude=row["longitude"]),
        depth=row["depth"],
        phase=row["phase"],
        x=row["x"],
        y=row["y"],
        z=row["z"],
        bx=row["bx"],
        by=row["by"],
        # Older estimated_state.csv files predate the Q_x/Q_y/Q_bx/Q_by columns entirely;
        # fall back to zeros (matching EstimatedState.Q's own default) rather than KeyError.
        Q=np.array([row.get("Q_x", 0.0), row.get("Q_y", 0.0), row.get("Q_bx", 0.0), row.get("Q_by", 0.0)], dtype=float),
        P=np.array(list(row["P"]), dtype=float),
    )

def dict_to_state(state_dict: dict) -> EstimatedState:
    "Inverse of _serialize_state: builds an EstimatedState from a surfacing_action_log 'estimated_state'/'real_state' entry."
    return EstimatedState(
        time=pd.Timestamp(state_dict["time"]).tz_localize(None),
        location=Location(
            latitude=state_dict["location"]["latitude"],
            longitude=state_dict["location"]["longitude"],
        ),
        depth=state_dict["depth"],
        phase=state_dict["phase"],
        x=state_dict["x"],
        y=state_dict["y"],
        z=state_dict["z"],
        bx=state_dict["bx"],
        by=state_dict["by"],
        P=np.array(state_dict["P"], dtype=float),
    )

def read_last_surfacing(float_id: int) -> dict:
    """Returns the last surfacing/action entry from surfacing_action_log.json."""
    log_path = _float_dir(float_id) / "surfacing_action_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Surfacing and action log file does not exist for float ID: {float_id}")

    surfacing_and_action_log = read_json(log_path)
    if not surfacing_and_action_log:
        raise ValueError(f"Surfacing and action log is empty for float ID: {float_id}")

    return surfacing_and_action_log[-1]