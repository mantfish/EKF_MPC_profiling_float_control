import logging
import math
from datetime import datetime, timedelta

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from data_handler import query_uv, xy_to_latlon
from helpers import ControlAction, EstimatedState, Location

logger = logging.getLogger(__name__)


class MPC:

    def __init__(
        self,
        target_location: list[float],  # [lat, lon]

    ) -> None:
        self.target = Location(latitude=target_location[0], longitude=target_location[1])


    def _target_xy(self, start_lat: float, start_lon: float) -> tuple[float, float]:
        target_x = (self.target.longitude - start_lon) * 111_000.0 * math.cos(math.radians(start_lat))
        target_y = (self.target.latitude - start_lat) * 111_000.0
        return target_x, target_y

    def _distance_term(
        self,
        current_state: EstimatedState,
        previous_state: EstimatedState,
        action: ControlAction,
    ) -> float:
        
        target_x, target_y = self._target_xy(self.target.latitude, self.target.longitude)

        d_current = math.sqrt((current_state.x - target_x) ** 2 + (current_state.y - target_y) ** 2)

        d_prev = math.sqrt((previous_state.x - target_x) ** 2 + (previous_state.y - target_y) ** 2)

        return (d_current - d_prev) / (action.duration_hours*3600)


    def evaluate_cost(
        self,
        expected_surfacing: EstimatedState,
        last_surfaced: EstimatedState,
        action: ControlAction,
    ) -> tuple[float, float, float, float, float]:
        """Returns (total, flow_term, distance_term, science_term, variance_term)."""

        distance = 10*self._distance_term(current_state=expected_surfacing, previous_state=last_surfaced, action=action)

        return distance
    
    def export_parameters(self) -> dict[str, float]:
        return {
            "target_lat": self.target.latitude,
            "target_lon": self.target.longitude,
        }



def compute_jacobian(state, forecast, ou,
                      eps=500.0) -> np.ndarray:
    """4x4 continuous-time Jacobian F for state [x, y, bx, by].

    Position rows: current shear (central differences) plus the bias.
    Bias rows: OU mean reversion, -1/tau.

    Don't need to add bias term as gradient so doens't matter
    """


    dlat, dlon = ....

    v_mid = forecast.query_v(state.lon, state.lat)
    u_mid = forecast.query_u(state.lon, state.lat)

    u_uplus = forecast.query_u(state.lon + dlon, state.lat)
    u_uminus = forecast.query_u(state.lon - dlon, state.lat)

    v_uplus = forecast.query_v(state.lon + dlon, state.lat)
    v_uminus = forecast.query_v(state.lon - dlon, state.lat)


    v_vplus = forecast.query_v(state.lon, state.lat + dlat)
    v_vminus = forecast.query_v(state.lon, state.lat - dlat)

    v_uplus = forecast.query_v(state.lon, state.lat + dlat)
    v_uminus = forecast.query_v(state.lon, state.lat - dlat)


    return np.array([
        [(u_uplus - u_uminus) / (2 * eps), (u_vplus - u_vminus) / (2 * eps), 1.0, 0.0],
        [(v_uplus - v_uminus) / (2 * eps), (v_vplus - v_vminus) / (2 * eps), 0.0, 1.0],
        [0.0, 0.0, -1.0 / ou.tau_e, 0.0],
        [0.0, 0.0, 0.0, -1.0 / ou.tau_n],
    ])