from datetime import datetime
import logging
import math
import os
import tempfile
import time

import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from helpers import Config, Location, region_around

logger = logging.getLogger(__name__)

CMEMS_DEPTH_MAX = 200.0


def to_seconds(t) -> float:
    return np.datetime64(pd.Timestamp(t), "s").astype(np.float64)


class Forecast:
    def __init__(self, config: Config, location: Location, time_range: tuple[datetime, datetime]):
        if config.model_type != "CMEMS":
            raise ValueError(f"Unknown model type in config: {config.model_type}")

        # Same 2 * max_drift box as Bathymetry, the MPC simulations start up to max_drift away
        region = region_around(location, 2 * config.max_drift)
        started = time.monotonic()
        ds = self.download_cmems(config.dataset_id, region, time_range)
        logger.info("CMEMS download took %.1f s", time.monotonic() - started)

        times = ds["time"].values.astype("datetime64[s]").astype(np.float64)
        depths = ds["depth"].values
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        grid = (times, depths, lats, lons)
        self.u_interp = RegularGridInterpolator(grid, ds["uo"].values, bounds_error=False, fill_value=np.nan)
        self.v_interp = RegularGridInterpolator(grid, ds["vo"].values, bounds_error=False, fill_value=np.nan)
        self.bounds = [(axis.min(), axis.max()) for axis in grid]
        self.out_of_bounds_logged = False

    def download_cmems(self, dataset_id, region, time_range) -> xr.Dataset:
        # Credentials come from the COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD env vars or the toolbox login file
        with tempfile.TemporaryDirectory() as tmp_dir:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=["uo", "vo"],
                minimum_latitude=region.latitude_min,
                maximum_latitude=region.latitude_max,
                minimum_longitude=region.longitude_min,
                maximum_longitude=region.longitude_max,
                minimum_depth=0.0,
                maximum_depth=CMEMS_DEPTH_MAX,
                start_datetime=pd.Timestamp(time_range[0]).isoformat(),
                end_datetime=pd.Timestamp(time_range[1]).isoformat(),
                output_directory=tmp_dir,
                output_filename="cmems_subset.nc",
                disable_progress_bar=True,
            )
            # Load into memory before tmp_dir is deleted, float32 is plenty for currents
            ds = xr.open_dataset(os.path.join(tmp_dir, "cmems_subset.nc")).load()
            for var in ("uo", "vo"):
                ds[var] = ds[var].astype(np.float32)

        logger.info("Downloaded CMEMS forecast %s to %s, lat [%.3f, %.3f] lon [%.3f, %.3f]",
                    ds.time.values[0], ds.time.values[-1], ds.latitude.min(), ds.latitude.max(), ds.longitude.min(), ds.longitude.max())
        return ds

    def query_uv(self, t: datetime, z: float, location: Location, bias=None) -> tuple[float, float]:
        """(u, v) in m/s at a point, clamped to the forecast grid, NaN (land/below seabed) as 0, plus optional bias."""
        point = [to_seconds(t), z, location.latitude, location.longitude]
        clamped = [np.clip(value, low, high) for value, (low, high) in zip(point, self.bounds)]
        # Depth is clamped quietly, the top CMEMS layer starts at ~0.5 m
        if not self.out_of_bounds_logged and [clamped[0], *clamped[2:]] != [point[0], *point[2:]]:
            logger.warning("Forecast query (t=%s, z=%.1f, %s) outside forecast grid, clamping to edge (only logged once).", t, z, location)
            self.out_of_bounds_logged = True

        u = float(self.u_interp([clamped])[0])
        v = float(self.v_interp([clamped])[0])
        u = 0.0 if math.isnan(u) else u
        v = 0.0 if math.isnan(v) else v

        if bias is not None:
            u += bias[0]
            v += bias[1]
        return u, v

    def check_forecast_coverage(self, time_range: tuple[datetime, datetime]) -> bool:
        t_min, t_max = self.bounds[0]
        return t_min <= to_seconds(time_range[0]) and to_seconds(time_range[1]) <= t_max
