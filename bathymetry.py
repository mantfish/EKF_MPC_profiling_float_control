class Bathymetry:
    def __init__(self, config: dict, target_location: Location):
        self.bathy_file = config.get("bathy_file")
        self.bathy_ds = self.load_bathymetry()
        self.target_location = location
        self._bathy_interpolator = self.build_bathymetry_interpolator()

    def load_bathymetry(self) -> xr.Dataset:
        """
        Loads the bathymetry dataset from the specified file.
        """
        if not self.bathy_file:
            raise ValueError("Bathymetry file path is not specified in the config.")
        
        try:
            bathy_ds = xr.open_dataset(self.bathy_file)
            return bathy_ds
        except Exception as e:
            logger.error("Failed to load bathymetry dataset: %s", e)
            raise

    def define_region_aroung_float() -> Region:
        """
        Define a region around the float's location based on the maximum drift expected."""
        start_location = self.target_location
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

    def get_depth(self, location: Location) -> float:
        """
        Returns the bathymetry depth (positive down) at a given (lat, lon) location.
        """
        lat = location.latitude
        lon = location.longitude

        # Use nearest-neighbor interpolation to get the depth at the specified location
        depth = self.bathy_ds.sel(latitude=lat, longitude=lon, method="nearest")["depth"].values.item()

        return float(depth)


    def build_bathymetry_interpolator(self) -> Callable[[float, float], float]:
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
        self.bbox = self.define_region_aroung_float() if self.config.max_drift else None
        if self.bbox is not None:
            bathy_ds = bathy_ds.sel(
                lat=slice(self.bbox.latitude_min, self.bbox.latitude_max),
                lon=slice(self.bbox.longitude_min, self.bbox.longitude_max),
            )

        lats = self.bathy_ds.lat.values.astype(np.float64)
        lons = self.bathy_ds.lon.values.astype(np.float64)
        elev = self.bathy_ds["elevation"].values.astype(np.float64)  # (lat, lon)
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()

        interp = RegularGridInterpolator(
            (lats, lons), elev, method="linear", bounds_error=False, fill_value=np.nan,
        )

        logged_out_of_bounds = False

        def query(lat: float, lon: float) -> float:
            nonlocal logged_out_of_bounds
            lat_c = np.clip(lat, lat_min, lat_max)
            lon_c = np.clip(lon, lon_min, lon_max)
            if not logged_out_of_bounds and (lat_c != lat or lon_c != lon):
                logger.warning(
                    "Bathymetry query (%.4f, %.4f) outside grid bounds lat[%.4f, %.4f] "
                    "lon[%.4f, %.4f]; clamping to nearest edge value (further out-of-bounds "
                    "queries in this run will not be logged).",
                    lat, lon, lat_min, lat_max, lon_min, lon_max,
                )
                logged_out_of_bounds = True
            depth = float(interp([[lat_c, lon_c]])[0]) * -1.0
            if math.isnan(depth):
                # max(nan, 0.0) would silently return nan here (any comparison
                # against NaN is False, so Python's max() keeps the first arg).
                depth = 0.0
            return max(depth, 0.0)

        return query

    def get_depth(self, location: Location) -> float:
        """
        Returns the bathymetry depth (positive down) at a given (lat, lon) location.
        """
        lat = location.latitude
        lon = location.longitude

        # Use the interpolator to get the depth at the specified location
        depth = self._bathy_interpolator(lat, lon)

        return float(depth)

    def check_bathymetry_coverage(self, location: Location, max_drift: float) -> bool:
        """
        Checks if the bathymetry data covers the region around the float's location
        based on the maximum drift expected.

        Parameters
        ----------
        location : Location
            The current location of the float.
        max_drift : float
            The maximum drift distance in meters.

        Returns
        -------
        bool
            True if the bathymetry data covers the region, False otherwise.
        """
        region = self.define_region_aroung_float()
        
        # Check if the bathymetry dataset covers the defined region
        lat_min, lat_max = self.bathy_ds.lat.min().item(), self.bathy_ds.lat.max().item()
        lon_min, lon_max = self.bathy_ds.lon.min().item(), self.bathy_ds.lon.max().item()

        return (region.latitude_min >= lat_min and region.latitude_max <= lat_max and
                region.longitude_min >= lon_min and region.longitude_max <= lon_max)