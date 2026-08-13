import logging
import math
from datetime import datetime, timedelta

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from data_handler import query_uv, xy_to_latlon
from helpers import ControlAction, EstimatedState, Location

logger = logging.getLogger(__name__)


class KFMPC:

    def __init__(
        self,
        target_location: list[float],  # [lat, lon]
        flow_weight: float = -0.8,
        distance_weight: float = 1.0,
        science_weight: float = 1.0,
        variance_weight: float = 1.0,
        radius_std_m: float = 5000.0,
        time_horizon_hours: int = 6,
    ) -> None:
        self.target = Location(latitude=target_location[0], longitude=target_location[1])
        self.flow_weight = flow_weight
        self.distance_weight = distance_weight
        self.science_weight = science_weight
        self.variance_weight = variance_weight
        self.radius_std_m = radius_std_m
        self.time_horizon_hours = time_horizon_hours

    def _target_xy(self, start_lat: float, start_lon: float) -> tuple[float, float]:
        target_x = (self.target.longitude - start_lon) * 111_000.0 * math.cos(math.radians(start_lat))
        target_y = (self.target.latitude - start_lat) * 111_000.0
        return target_x, target_y

    def _flow_term(
        self,
        state: EstimatedState,
        interp_u: RegularGridInterpolator,
        interp_v: RegularGridInterpolator,
        bounds: dict[str, float],
        start_lat: float,
        start_lon: float,
    ) -> float:
        target_x, target_y = self._target_xy(start_lat, start_lon)
        to_target = np.array([target_x - state.x, target_y - state.y])
        norm = np.linalg.norm(to_target)
        if norm < 1.0:
            return 0.0
        n = to_target / norm

        ux, uy = 0.0, 0.0
        for h in range(self.time_horizon_hours):
            t = state.time + timedelta(hours=h)
            lat, lon = xy_to_latlon(state.x, state.y, start_lat, start_lon)
            u, v = query_uv(interp_u, interp_v, bounds, t, 0.0, lat, lon)
            ux += u + state.bx
            uy += v + state.by
        return float(np.dot(np.array([ux, uy]), n))/(self.time_horizon_hours)

    def _distance_term(
        self,
        current_state: EstimatedState,
        previous_state: EstimatedState,
        action: ControlAction,
        start_lat: float,
        start_lon: float,
    ) -> float:
        target_x, target_y = self._target_xy(start_lat, start_lon)
        d_current = math.sqrt((current_state.x - target_x) ** 2 + (current_state.y - target_y) ** 2)
        d_prev = math.sqrt((previous_state.x - target_x) ** 2 + (previous_state.y - target_y) ** 2)
        return (d_current - d_prev) / (action.duration_hours*3600)

    def _science_term(self, state: EstimatedState, action: ControlAction,
                      start_lat: float, start_lon: float) -> float:
        target_x, target_y = self._target_xy(start_lat, start_lon)
        dist = math.sqrt((state.x - target_x) ** 2 + (state.y - target_y) ** 2)
        proximity = math.exp(-dist ** 2 / (2 * self.radius_std_m ** 2))
        return proximity * action.science_cost

    def _variance_term(self, state: EstimatedState) -> float:
        return float(np.trace(state.P[2:, 2:]))

    def evaluate_cost(
        self,
        final_state: EstimatedState,
        prev_real_state: EstimatedState,
        action: ControlAction,
        interp_u: RegularGridInterpolator,
        interp_v: RegularGridInterpolator,
        bounds: dict[str, float],
        start_lat: float,
        start_lon: float,
    ) -> tuple[float, float, float, float, float]:
        """Returns (total, flow_term, distance_term, science_term, variance_term)."""
        flow = self._flow_term(final_state, interp_u, interp_v, bounds, start_lat, start_lon)
        distance = self._distance_term(current_state=final_state, previous_state=prev_real_state, action=action, start_lat=start_lat, start_lon=start_lon)
        science = self._science_term(final_state, action, start_lat, start_lon)
        variance = self._variance_term(final_state)
        total = (
            - self.flow_weight * flow
            + self.distance_weight * distance
            + self.science_weight * science # TODO check this
            + self.variance_weight * variance
        )
        logger.info("For action: %e, total cost is: %i. The flow term: %s, distance term: %d, science term: %f, variance: %g", action, total, flow, distance, science, variance)
        return total, flow, distance, science, variance

    def export_parameters(self) -> dict[str, float]:
        return {
            "flow_weight": self.flow_weight,
            "distance_weight": self.distance_weight,
            "science_weight": self.science_weight,
            "variance_weight": self.variance_weight,
            "radius_std_m": self.radius_std_m,
            "time_horizon_hours": self.time_horizon_hours,
            "target_lat": self.target.latitude,
            "target_lon": self.target.longitude,
        }


def compute_jacobian(
    tau: float,
    x: float,
    y: float,
    z: float,
    t: datetime,
    interp_u: RegularGridInterpolator,
    interp_v: RegularGridInterpolator,
    bounds: dict[str, float],
    start_lat: float,
    start_lon: float,
    eps: float = 500.0,
) -> np.ndarray:
    """4x4 linearised dynamics Jacobian F for state [x, y, bx, by]."""
    def qv(xi: float, yi: float) -> tuple[float, float]:
        lat, lon = xy_to_latlon(xi, yi, start_lat, start_lon)
        return query_uv(interp_u, interp_v, bounds, t, z, lat, lon)

    def in_bounds(xi: float, yi: float) -> bool:
        lat, lon = xy_to_latlon(xi, yi, start_lat, start_lon)
        return (bounds["lat_min"] <= lat <= bounds["lat_max"]
                and bounds["lon_min"] <= lon <= bounds["lon_max"])

    # A centered difference across a query_uv() clamp boundary compares a real
    # interior value against a clamped edge value, producing a spurious step
    # divided by 2*eps rather than a physical gradient. Prefer a centered
    # difference when both samples are real; fall back to one-sided when only
    # one side is; fall back to zero shear when the window is narrower than
    # `eps` in that direction (e.g. a tight max_drift box on a long action).
    u0, v0 = qv(x, y)

    def directional(axis: str) -> tuple[float, float]:
        plus = (x + eps, y) if axis == "x" else (x, y + eps)
        minus = (x - eps, y) if axis == "x" else (x, y - eps)
        plus_ok, minus_ok = in_bounds(*plus), in_bounds(*minus)
        if plus_ok and minus_ok:
            up, vp = qv(*plus)
            um, vm = qv(*minus)
            return (up - um) / (2 * eps), (vp - vm) / (2 * eps)
        if plus_ok:
            up, vp = qv(*plus)
            return (up - u0) / eps, (vp - v0) / eps
        if minus_ok:
            um, vm = qv(*minus)
            return (u0 - um) / eps, (v0 - vm) / eps
        logger.warning("compute_jacobian: forecast window narrower than eps=%.0fm at "
                        "(x=%.0f, y=%.0f); assuming zero shear on %s axis.", eps, x, y, axis)
        return 0.0, 0.0

    du_dx, dv_dx = directional("x")
    du_dy, dv_dy = directional("y")

    return np.array([
        [du_dx, du_dy, 1.0, 0.0],
        [dv_dx, dv_dy, 0.0, 1.0],
        [0.0, 0.0, -1/tau, 0.0],
        [0.0, 0.0, 0.0, -1/tau],
    ])
