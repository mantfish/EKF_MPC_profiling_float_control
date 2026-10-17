

class Forecast:
    def __init__(self, config: dict, location: location, time_range: tuple[datetime, datetime]):

        self.target_location = location
        self.model_type = config.model_type
        self.time_range = time_range
        self.forecast_ds = self.get_forecast()
        self.uv_interpolators = self.build_uv_interpolators()


    def cmems_credentials(self) -> tuple[str, str]:
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

    def define_region_aroung_float(self, max_drift: float) -> Region:
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


    def get_forecast(
        ) -> xr.Dataset:

        username, password = self.cmems_credentials()
        if config.max_drift:
            bbox = self.define_region_aroung_float(self.target_location, self.config.max_drift)
        else:
            bbox = self.define_region_aroung_float(self.target_location, MAX_DRIFT)

        start_datetime = pd.Timestamp(min(self.time_range)) - pd.Timedelta(hours=CMEMS_TIME_PRIOR)

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

            copernicusmarine.subset(**kwargs)

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

            # CMEMS's packed-int16-with-scale-factor storage gets CF-decoded to float64
            # by default, which is far more precision than an ocean current velocity
            # needs. Downcast to float32 to roughly halve this dataset's memory footprint.
            for var in ("uo", "vo"):
                ds[var] = ds[var].astype(np.float32, copy=False)

        first_time = ds.time[0].values
        last_time = ds.time[-1].values

        logger.info("CMEMS subset time range: %s to %s", first_time, last_time)
        logger.info("CMEMS subset lat range: %s to %s", ds.latitude.min().values, ds.latitude.max().values)

        return ds

    def build_uv_interpolators(
    ) -> tuple[RegularGridInterpolator, RegularGridInterpolator, dict[str, float]]:
        """
        Builds (u_interp, v_interp, bounds) from a CMEMS-style xarray Dataset
        with dims (time, depth, latitude, longitude).
        """
        if not self.forecast_ds:
            raise RuntimeError("Forecast dataset is not loaded. Call get_forecast() first.")
        
        times = self.forecast_ds["time"].values.astype("datetime64[s]").astype(np.float64)
        depths = self.forecast_ds["depth"].values
        lats = self.forecast_ds["latitude"].values
        lons = self.forecast_ds["longitude"].values

        u_interp = RegularGridInterpolator(
            (times, depths, lats, lons),
            self.forecast_ds["uo"].values,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        v_interp = RegularGridInterpolator(
            (times, depths, lats, lons),
            self.forecast_ds["vo"].values,
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

        t: datetime,
        z: float,
        location: Location,
        bias = None,
    
    ) -> tuple[float, float]:
        """
        Queries (u, v) at a single (t, z, lat, lon) point via linear
        interpolation, clamping to grid bounds and zeroing NaNs.
        """



        t_s = np.datetime64(t, "s").astype(np.float64)

        t_c = np.clip(t_s, self.bounds["t_min"], self.bounds["t_max"])
        z_c = np.clip(z, self.bounds["z_min"], self.bounds["z_max"])
        lat_c = np.clip(location.lat, self.bounds["lat_min"], self.bounds["lat_max"])
        lon_c = np.clip(location.lon, self.bounds["lon_min"], self.bounds["lon_max"])

        if not self.bounds.get("_out_of_bounds_logged") and (
            t_c != t_s or z_c != z or lat_c != location.lat or lon_c != location.lon
        ):
            logger.warning(
                "UV query (t=%s, z=%.1f, lat=%.4f, lon=%.4f) outside forecast bounds "
                "t[%s, %s] z[%.1f, %.1f] lat[%.4f, %.4f] lon[%.4f, %.4f]; clamping to "
                "nearest edge value (further out-of-bounds queries in this run will not be logged).",
                t, z, location.lat, location.lon,
                self.bounds["t_min"], self.bounds["t_max"], self.bounds["z_min"], self.bounds["z_max"],
                self.bounds["lat_min"], self.bounds["lat_max"], self.bounds["lon_min"], self.bounds["lon_max"],
            )
            self.bounds["_out_of_bounds_logged"] = True

        point = [[t_c, z_c, lat_c, lon_c]]

        u = float(self._uv_interpolators[0](point)[0])
        v = float(self._uv_interpolators[1](point)[0])

        if bias is not None:
            u += bias[0]
            v += bias[1]

        if math.isnan(u):
            u = 0.0
        if math.isnan(v):
            v = 0.0

        return u, v

    def check_forecast_coverage(self, time_range: tuple[datetime, datetime]) -> bool:
        """
        Checks if the forecast dataset covers the specified time range.
        Returns True if it does, False otherwise.
        """
        start_time, end_time = time_range
        start_time_s = np.datetime64(start_time, "s").astype(np.float64)
        end_time_s = np.datetime64(end_time, "s").astype(np.float64)

        return (self.bounds["t_min"] <= start_time_s <= self.bounds["t_max"]) and (self.bounds["t_min"] <= end_time_s <= self.bounds["t_max"])

    
