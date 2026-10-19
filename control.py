import logging
import math

import numpy as np

from helpers import ControlAction, EstimatedState, Location, latlon_to_xy

logger = logging.getLogger(__name__)


class MPC:

    def __init__(self, target_location: list[float]) -> None:  # [lat, lon]
        self.target = Location(latitude=target_location[0], longitude=target_location[1])

    def _distance_to_target(self, state: EstimatedState) -> float:
        dx, dy = latlon_to_xy(state.location.latitude, state.location.longitude, self.target.latitude, self.target.longitude)
        return math.hypot(dx, dy)

    def evaluate_cost(self, expected_surfacing: EstimatedState, last_surfaced: EstimatedState, action: ControlAction) -> float:
        """Rate of change of distance to target (m/s) over the action, lower is better."""
        distance_change = self._distance_to_target(expected_surfacing) - self._distance_to_target(last_surfaced)
        return 10 * distance_change / (action.duration_hours * 3600)


def compute_jacobian(state, forecast, ou, eps=500.0) -> np.ndarray:
    """4x4 continuous-time Jacobian F for state [x, y, bx, by].

    Position rows: current shear (central differences over eps metres) plus the bias.
    Bias rows: OU mean reversion, -1/tau.
    """

    def uv_at(dx, dy):
        location = Location(
            latitude=state.location.latitude + dy / 111_000.0,
            longitude=state.location.longitude + dx / (111_000.0 * math.cos(math.radians(state.location.latitude))),
        )
        return forecast.query_uv(state.time, state.depth, location)

    u_east, v_east = uv_at(eps, 0.0)
    u_west, v_west = uv_at(-eps, 0.0)
    u_north, v_north = uv_at(0.0, eps)
    u_south, v_south = uv_at(0.0, -eps)

    return np.array([
        [(u_east - u_west) / (2 * eps), (u_north - u_south) / (2 * eps), 1.0, 0.0],
        [(v_east - v_west) / (2 * eps), (v_north - v_south) / (2 * eps), 0.0, 1.0],
        [0.0, 0.0, -1.0 / ou.tau_e, 0.0],
        [0.0, 0.0, 0.0, -1.0 / ou.tau_n],
    ])
