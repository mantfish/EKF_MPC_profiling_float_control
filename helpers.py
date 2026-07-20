from dataclasses import dataclass
from datetime import datetime
from typing import  Literal

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

Phase = Literal["ascending", "descending", "parking", "on_seabed", "communicating"]

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
    
@dataclass
class ControlAction:
    parking_depth: float        # metres
    duration_hours: float       # hours
    science_cost: float         # 0 (no science) to 1 (full science)

def get_action(action_name: str, float_id: int) -> ControlAction:
    config = read_config(float_id)
    for action in config["possible_actions"]:
        if action["name"] == action_name:
            return ControlAction(
                parking_depth=action["parking_depth"],
                duration_hours=action["duration_hours"],
                science_cost=action["science_cost"]
            )
    raise ValueError(f"Action '{action_name}' not found in possible actions.")

@dataclass
class SurfacingLog:
    time: datetime
    location: Location
    daction: ControlAction

@dataclass
class Config:
    # TODO implement this here clearly