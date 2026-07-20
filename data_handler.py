import pathlib as path
from typing import *
import jsons
import copernicusmarine
import tempfile
import logging
import os
from datetime import datetime
from helpers import *
import numpy as np
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

CMEMS_DATASET_ID = "cmems_mod_bal_phy_anfc_PT1H-i_202411"
CMEMS_DEPTH_MAX  = 200.0   
CMEMS_TIME_PRIOR = 12
KM_PER_DEG_LAT = 111.32

logger = logging.getLogger(__name__)

MAX_DRIFT = 100 # max drift in km expected float can undergo


def read_config(float_id: int) -> dict:
    "Opens config file"

    config_path = path.join(path.dirname(float_id), "config.json")

    try:
        config_json = read_json(config_path)
    except Exception as e:
        logger.error("Error reading config file: %s", e)
        raise
    # Todo parse control actions into actions
    config.Q = np.diagonal(config_json["process_noise"])
    return config

def download_cmems_data_around_float(location: Location) -> xr.Dataset: 
    
    bbox = define_region_aroung_float(location)

    kwargs: dict = dict(
        dataset_id=CMEMS_DATASET_ID,
        variables=["uo", "vo"],
        minimum_latitude=bbox.lat_min,
        maximum_latitude=bbox.lat_max,
        minimum_longitude=bbox.lon_min,
        maximum_longitude=bbox.lon_max,
        minimum_depth=0.52,
        maximum_depth=CMEMS_DEPTH_MAX,
        output_filename="cmems_subset.nc",
        # no start_datetime / end_datetime -> full available range
    )

    #TODO probably need to thrown in an end datetime too. 

    with tempfile.TemporaryDirectory() as tmp_dir:
        kwargs["output_directory"] = tmp_dir
    tmp_path = os.path.join(tmp_dir, "cmems_subset.nc")

    copernicusmarine.subset(**kwargs)
    logger.info("CMEMS subset downloaded on %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    ds = xr.open_dataset(tmp_path)

    first_time = ds.time[0].values
    last_time = ds.time[-1].values

    logger.info("CMEMS subset time range: %s to %s", first_time, last_time)
    logger.info("CMEMS subset lat range: %s to %s", ds.latitude.min().values, ds.latitude.max().values)
    
    return ds

def read_last_surfacing_and_action(float_id: int) -> dict:
    surfacing_and_action_log_path = path.join(path.dirname(float_id), "surfacing_log.json")

    if not path.exists(surfacing_log_path):
        raise FileNotFoundError(f"Surfacing and action log file does not exist for float ID: {float_id}")

    with open(surfacing_and_action_log_path, "r") as f:
        surfacing_and_action_log = jsons.load(f)

    last_surfacing = surfacing_and_action_log[-1]  # Get the last surfacing

    return [last_surfacing["location"], last_surfacing["action"]]


def write_last_surfacing_and_action(float_id: int, location: Location, action: ControlAction, estimated_state, innovated_state, nis) -> None:
    surfacing_and_action_log_path = path.join(path.dirname(float_id), "surfacing_log.json")

    if path.exists(surfacing_and_action_log_path):
        with open(surfacing_and_action_log_path, "r") as f:
            surfacing_and_action_log = jsons.load(f)
    else:
        surfacing_and_action_log = []

    new_entry = {
        "time": datetime.now().isoformat(),
        "location": {
            "latitude": location.latitude,
            "longitude": location.longitude
        },
        "action": {
            "parking_depth": action.parking_depth,
            "duration_hours": action.duration_hours,
            "science_cost": action.science_cost
        },
        "estimated_state": {
            "x": estimated_state.x,
            "y": estimated_state.y,
            "bx": estimated_state.bx,
            "by": estimated_state.by
        },
        "innovated_state": {
            "x": innovated_state.x,
            "y": innovated_state.y,
            "bx": innovated_state.bx,
            "by": innovated_state.by
        },
        "nis": nis
    }

    surfacing_and_action_log.append(new_entry)

    with open(surfacing_and_action_log_path, "w") as f:
        jsons.dump(surfacing_and_action_log, f, indent=4)

    logger.info("Surfacing and action log updated for float ID: %s", float_id)


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

def read_state(float_id: int) -> dict:
    
    state_file_path = path.join(path.dirname(float_id), "estimated_state.parquet")
    if not path.exists(state_file_path):
        logger.error("State file does not exist for float ID: %s", float_id)
        raise FileNotFoundError(f"State file does not exist for float ID: {float_id}")

    state_df = pd.read_parquet(state_file_path)
    
    return state_df

def write_state(float_id: int, state_df: pd.DataFrame) -> None:
    state_file_path = path.join(path.dirname(float_id), "estimated_state.parquet")
    state_df.to_parquet(state_file_path, index=False)
    logger.info("Estimated state history written to file for float ID: %s", float_id)
    return True

def read_surfacings(float_id: int) -> pd.DataFrame:
    surfacing_file_path = path.join(path.dirname(float_id), "surfacing_log.json")
    if not path.exists(surfacing_file_path):
        logger.error("Surfacing log file does not exist for float ID: %s", float_id)
        raise FileNotFoundError(f"Surfacing log file does not exist for float ID: {float_id}")

    surfacing_df = jsons.load(open(surfacing_file_path, "r"))

    last_surfacing
    
    return surfacing_df

def get_action(action_name: str, config) -> ControlAction:
    possible_actions = config["possible_actions"]
    if action_name not in possible_actions:
        raise ValueError(f"Action '{action_name}' not found in possible actions.")
    else:
        action_config = possible_actions[action_name]
        return ControlAction(
            parking_depth=action_config["parking_depth"],
            duration_hours=action_config["duration_hours"],
            science_cost=action_config["science_cost"]
        )

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

def write_surfacings(float_id: int, estiamed_state, innovated_state):
    surfacing_log_path = path.join(path.dirname(float_id), "surfacing_log.json")
    surfacing_entry = {
        "time": innovated_state.time.isoformat(),
        "location": {
            "latitude": innovated_state.location.latitude,
            "longitude": innovated_state.location.longitude
        },
        "daction": {
            "parking_depth": estiamed_state.parking_depth,
            "duration_hours": estiamed_state.duration_hours,
            "science_cost": estiamed_state.science_cost
        }
    }

    if path.exists(surfacing_log_path):
        with open(surfacing_log_path, "r") as f:
            surfacing_log = jsons.load(f)
    else:
        surfacing_log = []

    surfacing_log.append(surfacing_entry)

    with open(surfacing_log_path, "w") as f:
        jsons.dump(surfacing_log, f, indent=4)

    logger.info("Surfacing log updated for float ID: %s", float_id)