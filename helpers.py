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
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    bx: float = 0.0  # estimated current bias east (m/s)
    by: float = 0.0  # estimated current bias north (m/s)
    P: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0, 1e-4, 1e-4]))
    Q: np.ndarray = field(default_factory=lambda: np.zeros(4))  # process noise diagonal [x, y, bx, by] active when this row was computed
    
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
    daction: ControlAction

@dataclass
class Config:
    """Mirrors the fields in a float's config.json (see floats/123456/config.json)."""
    float_id: int
    target_lat: float
    target_lon: float
    radius_std: float
    flow_time_horizon_hours: float
    flow_weight: float
    distance_weight: float
    science_weight: float
    variance_weight: float
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
    max_drift: float = 100.0  # max drift in km expected float can undergo
    Q: np.ndarray = field(default_factory=lambda: np.zeros(4))  # process noise diagonal [x, y, bx, by]: zero on position, from config.json's process_noise_bias on bias
    bias_decay_rate: float | None = None  # bias mean-reversion time-constant tau (seconds), from config.json's bias_decay_rate


