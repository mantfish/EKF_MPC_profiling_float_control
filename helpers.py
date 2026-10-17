from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import  Literal

import numpy as np

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

Phase = Literal["ascending", "descending", "drifting", "grounded", "communicating"]

@dataclass
class EstimatedState:
    time: datetime
    location: Location
    depth: float
    phase: Phase
    z: float = 0.0
    bx: float = 0.0  # estimated current bias east (m/s)
    by: float = 0.0  # estimated current bias north (m/s)
    P: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0, 1e-4, 1e-4]))
    Q: np.ndarray = field(default_factory=lambda: np.zeros(4))  # process noise diagonal [x, y, bx, by] active when this row was computed
    def update_location(self,  dx, dy):
        self.location.latitude += dy / 111_000.0
        self.location.longitude += dx / (111_000.0 * math.cos(math.radians(self.location.latitude)))
    def update_bias(self,)

@dataclass
class ControlAction:
    drifting_depth: float        # metres
    duration_hours: float       # hours
    science_cost: float         # 0 (no science) to 1 (full science)
    grounding: bool # True if the action is a grounding action, False otherwise

@dataclass
class SurfacingLog:
    time: datetime
    location: Location
    action: ControlAction

@dataclass
class Config:
    """Mirrors the fields in a float's config.json (see floats/123456/config.json)."""
    float_id: int
    target_lat: float
    target_lon: float
    vertical_dt: float
    parking_dt: float
    bathymetry_file_name: str
    estimated_tranmission_time_hours: float
    ascent_speed_m_per_s: float
    descent_speed_m_per_s: float
    possible_actions: list[dict]
    data_dir: Path
    model_type: str = "CMEMS"
    dataset_id: str = "cmems_mod_bal_phy_anfc_PT1H-i_202411"
    max_drift: float = 30.0  # max drift in km expected float can undergo

@dataclass
class OU:
    """Ornstein-Uhlenbeck model of the forecast current bias, one process per axis.

    db = -b / tau dt + sqrt(q) dW   ->   steady-state variance q * tau / 2
    """
    tau_e: float = 50 * 3600.0  # seconds, east bias correlation time
    tau_n: float = 11 * 3600.0  # seconds, north bias correlation time
    q_e: float = 3e-8           # east bias noise intensity (m^2/s^3)
    q_n: float = 1.3e-7         # north bias noise intensity (m^2/s^3)

    def get_ou_step(self, tau: float, q: float, dt: float) -> tuple[float, float]:
        """Decay factor phi and per-step noise variance for an OU step of length dt."""
        phi_n = np.exp(-dt / self.tau_n)
        phi_e = np.exp(-dt / self.tau_e)
        steady_var_n = self.q_n * self.tau_n / 2
        steady_var_e = self.q_e * self.tau_e / 2

        step_var_n = steady_var_n * (1 - phi_n ** 2)
        step_var_e = steady_var_e * (1 - phi_e ** 2)

        return (phi_e, step_var_e), (phi_n, step_var_n)




