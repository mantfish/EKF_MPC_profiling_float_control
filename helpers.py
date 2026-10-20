from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
from typing import Literal

import numpy as np

KM_PER_DEG_LAT = 111.32


def latlon_to_xy(lat: float, lon: float, start_lat: float, start_lon: float) -> tuple[float, float]:
    """Metres east and north of (start_lat, start_lon)."""
    x = (lon - start_lon) * 111_000.0 * math.cos(math.radians(start_lat))
    y = (lat - start_lat) * 111_000.0
    return x, y


@dataclass
class Location:
    latitude: float
    longitude: float

@dataclass
class Region:
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float


def region_around(location: Location, max_drift_km: float) -> Region:
    dlat = max_drift_km / KM_PER_DEG_LAT
    dlon = max_drift_km / (KM_PER_DEG_LAT * math.cos(math.radians(location.latitude)))
    return Region(location.latitude - dlat, location.latitude + dlat, location.longitude - dlon, location.longitude + dlon)

Phase = Literal["ascending", "descending", "drifting", "grounded", "communicating"]

@dataclass
class EstimatedState:
    time: datetime
    location: Location
    depth: float
    phase: Phase
    bx: float = 0.0  # estimated current bias east (m/s)
    by: float = 0.0  # estimated current bias north (m/s)
    P: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0, 1e-4, 1e-4]))  # covariance of [x, y, bx, by]

    def update_location(self, dx, dy):
        self.location.latitude += dy / 111_000.0
        self.location.longitude += dx / (111_000.0 * math.cos(math.radians(self.location.latitude)))


@dataclass
class ControlAction:
    drifting_depth: float       # metres
    duration_hours: float       # hours


@dataclass
class Config:
    """Mirrors the fields in a float's config.json (see floats/3902607/config.json)."""
    float_id: int
    target_lat: float
    target_lon: float
    vertical_dt: float
    parking_dt: float
    bathymetry_file_name: str
    delay_before_diving_hours: float  # time on the surface after the first GPS fix before descending
    ascent_speed_m_per_s: float
    descent_speed_m_per_s: float
    possible_actions: list[ControlAction]
    data_dir: Path
    model_type: str = "CMEMS"
    dataset_id: str = "cmems_mod_bal_phy_anfc_PT1H-i_202411"
    max_drift: float = 30.0  # max drift in km expected float can undergo
    buffer_time: float = 6.0  # hours of forecast to download either side of the simulation
    max_time_difference: float = 30.0  # minutes the real surfacing can be later than estimated before drifting on surface

@dataclass
class OU:
    """Ornstein-Uhlenbeck model of the forecast current bias, one process per axis.

    db = -b / tau dt + sqrt(q) dW   ->   steady-state variance q * tau / 2
    """
    tau_e: float = 50 * 3600.0  # seconds, east bias correlation time
    tau_n: float = 11 * 3600.0  # seconds, north bias correlation time
    q_e: float = 3e-8           # east bias noise intensity (m^2/s^3)
    q_n: float = 1.3e-7         # north bias noise intensity (m^2/s^3)

    def get_ou_step(self, dt: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Decay factor and per-step noise variance for an OU step of length dt, as (east, north)."""
        phi_e = np.exp(-dt / self.tau_e)
        phi_n = np.exp(-dt / self.tau_n)
        step_var_e = self.q_e * self.tau_e / 2 * (1 - phi_e ** 2)
        step_var_n = self.q_n * self.tau_n / 2 * (1 - phi_n ** 2)
        return (phi_e, step_var_e), (phi_n, step_var_n)
