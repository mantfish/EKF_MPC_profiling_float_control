import logging
import math

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from helpers import Config, Location, region_around

logger = logging.getLogger(__name__)


class Bathymetry:
    def __init__(self, config: Config, location: Location):
        # Crop to twice max_drift: the MPC simulations start from an estimated
        # surfacing that can itself be up to max_drift from this location
        region = region_around(location, 2 * config.max_drift)

        with xr.open_dataset(config.data_dir / config.bathymetry_file_name) as ds:
            elevation = ds["elevation"].sel(
                lat=slice(region.latitude_min, region.latitude_max),
                lon=slice(region.longitude_min, region.longitude_max),
            ).load()

        self.lats = elevation.lat.values.astype(np.float64)
        self.lons = elevation.lon.values.astype(np.float64)
        self.interpolator = RegularGridInterpolator((self.lats, self.lons), elevation.values.astype(np.float64), bounds_error=False, fill_value=np.nan)
        self.out_of_bounds_logged = False
        logger.info("Loaded bathymetry lat [%.3f, %.3f] lon [%.3f, %.3f]", self.lats.min(), self.lats.max(), self.lons.min(), self.lons.max())

    def get_depth(self, location: Location) -> float:
        """Seabed depth in metres, positive down, 0 on land. Clamps to the grid edge outside the loaded region."""
        lat = np.clip(location.latitude, self.lats.min(), self.lats.max())
        lon = np.clip(location.longitude, self.lons.min(), self.lons.max())
        if not self.out_of_bounds_logged and (lat != location.latitude or lon != location.longitude):
            logger.warning("Bathymetry query %s outside loaded region, clamping to edge (only logged once).", location)
            self.out_of_bounds_logged = True

        depth = -float(self.interpolator([[lat, lon]])[0])
        if math.isnan(depth):
            return 0.0
        return max(depth, 0.0)

    def check_bathymetry_coverage(self, location: Location, max_drift: float) -> bool:
        """True if the loaded grid covers max_drift km around location."""
        region = region_around(location, max_drift)
        return (self.lats.min() <= region.latitude_min and region.latitude_max <= self.lats.max()
                and self.lons.min() <= region.longitude_min and region.longitude_max <= self.lons.max())
