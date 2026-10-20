
import copy
from datetime import datetime, timedelta
import logging
import time

import numpy as np

from bathymetry import Bathymetry
from control import MPC, compute_jacobian
from forecast import Forecast
from helpers import ControlAction, EstimatedState, Location, OU, latlon_to_xy
from records import Records

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def simulate_action(starting_state, forecast, bathymetry, action, config, ou):

    simulated_state = copy.deepcopy(starting_state)
    simulated_end_time = starting_state.time + timedelta(hours=action.duration_hours)
    dive_time = starting_state.time + timedelta(hours=config.delay_before_diving_hours)

    # Check that forecast and bathymetry cover the entire simulation
    if not forecast.check_forecast_coverage((starting_state.time, simulated_end_time)):
        raise ValueError("Forecast data does not cover the entire simulation period.")
    if not bathymetry.check_bathymetry_coverage(starting_state.location, config.max_drift):
        raise ValueError("Bathymetry data does not cover the entire simulation region.")

    states = []
    started = time.monotonic()

    while simulated_state.time < simulated_end_time:

        bottom_depth = bathymetry.get_depth(simulated_state.location)

        # Phase transitions
        if simulated_state.phase == "communicating":
            if simulated_state.time >= dive_time:
                simulated_state.phase = "descending"

        if simulated_state.phase == "descending":
            if simulated_state.depth >= bottom_depth:
                simulated_state.phase = "grounded"
            elif simulated_state.depth >= action.drifting_depth:
                simulated_state.phase = "drifting"

        if simulated_state.phase == "drifting" and simulated_state.depth >= bottom_depth:
            simulated_state.phase = "grounded"
            logger.warning("Float %s is drifting at depth %.2f m, below the bathymetry depth %.2f m at %s, so has now grounded.", config.float_id, simulated_state.depth, bottom_depth, simulated_state.location)

        if simulated_state.phase in ("drifting", "grounded"):
            if simulated_state.time > simulated_end_time - timedelta(seconds=simulated_state.depth / config.ascent_speed_m_per_s):
                simulated_state.phase = "ascending"

        if simulated_state.phase == "ascending" and simulated_state.depth <= 0.0:
            simulated_state.phase = "communicating"

        # Vertical motion + this iteration's timestep
        if simulated_state.phase == "descending":
            simulated_state.depth += config.descent_speed_m_per_s * config.vertical_dt
            dt = config.vertical_dt
        elif simulated_state.phase == "ascending":
            simulated_state.depth = max(0.0, simulated_state.depth - config.ascent_speed_m_per_s * config.vertical_dt)
            dt = config.vertical_dt
        else:
            dt = config.parking_dt

        (decay_e, bias_var_e), (decay_n, bias_var_n) = ou.get_ou_step(dt)
        Q_step = np.diag([0.0, 0.0, bias_var_e, bias_var_n])

        if simulated_state.phase != "grounded":
            F = compute_jacobian(simulated_state, forecast, ou)  # pre-step
            u, v = forecast.query_uv(simulated_state.time, simulated_state.depth, simulated_state.location, bias=(simulated_state.bx, simulated_state.by))
            simulated_state.update_location(u * dt, v * dt)
            # Position block: first-order Phi = I + F dt (|F dt| << 1).
            # Bias block: exact OU decay instead of 1 - dt/tau.
            Phi = np.eye(4) + F * dt
            Phi[2, 2] = decay_e
            Phi[3, 3] = decay_n
        else:
            # Float is stationary so position doesn't evolve,
            # but the bias keeps evolving in the real world regardless
            Phi = np.diag([1.0, 1.0, decay_e, decay_n])

        # Expected bias relaxes towards zero; covariance propagates through Phi
        simulated_state.bx *= decay_e
        simulated_state.by *= decay_n
        simulated_state.P = Phi @ simulated_state.P @ Phi.T + Q_step

        simulated_state.time += timedelta(seconds=dt)

        logger.debug("Simulated state at %s: %s, depth %.1f, phase %s", simulated_state.time, simulated_state.location, simulated_state.depth, simulated_state.phase)
        states.append(copy.deepcopy(simulated_state))

    logger.info("Simulated %s from %s in %.1f s, ending %s at %s", action, starting_state.time, time.monotonic() - started, states[-1].phase, states[-1].location)
    return states


def load_environment(records):
    config = records.config
    cycle_number = records.get_last_cycle_number()
    prev_surfaced_state = records.get_surfacing(cycle_number)
    prev_action = records.get_action(cycle_number)
    logger.info("Float %s: last surfacing cycle %s at %s %s, action %s, %s", records.float_id, cycle_number, prev_surfaced_state.time, prev_surfaced_state.location, prev_action, records.get_ou())

    max_action_hours = max(action.duration_hours for action in config.possible_actions)
    buffer = timedelta(hours=config.buffer_time)
    time_range = (prev_surfaced_state.time - buffer, prev_surfaced_state.time + timedelta(hours=2 * max_action_hours) + buffer)

    started = time.monotonic()
    forecast = Forecast(config, prev_surfaced_state.location, time_range)
    bathymetry = Bathymetry(config, prev_surfaced_state.location)
    logger.info("Loaded forecast and bathymetry in %.1f s", time.monotonic() - started)
    return config, cycle_number, prev_surfaced_state, prev_action, forecast, bathymetry


def get_action(float_id, debug=True):

    records = Records(float_id)
    config, prev_cycle_number, prev_surfaced_state, prev_action, forecast, bathymetry = load_environment(records)
    ou = records.get_ou()
    mpc = MPC([config.target_lat, config.target_lon])

    estimated_trajectory = simulate_action(prev_surfaced_state, forecast, bathymetry, prev_action, config, ou)
    estimated_future_surfacing = estimated_trajectory[-1]
    logger.info("Estimated next surfacing at %s: %s", estimated_future_surfacing.time, estimated_future_surfacing.location)

    best_action = None
    best_action_cost = np.inf
    for action in config.possible_actions:
        mpc_trajectory = simulate_action(copy.deepcopy(estimated_future_surfacing), forecast, bathymetry, action, config, ou)
        cost = mpc.evaluate_cost(expected_surfacing=mpc_trajectory[-1], last_surfaced=estimated_future_surfacing, action=action)
        logger.info("Action %s has cost %.3f", action, cost)

        if cost < best_action_cost:
            best_action = action
            best_action_cost = cost

    if not debug:
        records.save_trajectory(estimated_trajectory, prev_cycle_number + 1)
        records.save_prediction(estimated_future_surfacing, prev_cycle_number + 1)
        records.save_action(best_action, prev_cycle_number + 1)

    logger.info("Selected action for float %s cycle %s: %s, cost %.3f", float_id, prev_cycle_number + 1, best_action, best_action_cost)
    return best_action


def predict_surfacing(float_id, debug=True) -> EstimatedState:
    """Re-simulates the cycle in progress with the latest forecast, without choosing an action."""
    records = Records(float_id)
    config, cycle_number, prev_surfaced_state, prev_action, forecast, bathymetry = load_environment(records)
    trajectory = simulate_action(prev_surfaced_state, forecast, bathymetry, prev_action, config, records.get_ou())

    if not debug:
        records.save_trajectory(trajectory, cycle_number + 1)
        records.save_prediction(trajectory[-1], cycle_number + 1)
    return trajectory[-1]


def kf_update_step(est, surface_location, surface_time) -> tuple[EstimatedState, float]:
    """EKF measurement update from a real GPS surfacing fix.

    The fix is treated as exact (R=0). In that limit position is replaced
    outright by the fix, and the bias estimate gets a correction (and its
    variance a reduction) through the position-bias cross-covariance built
    up over the preceding drift -- that is where the bias "learning" happens.
    Returns the updated state and the normalised innovation squared (NIS),
    which should average ~2 if P is well calibrated.
    """

    P_XX = est.P[:2, :2]
    P_Xb = est.P[:2, 2:]
    P_bX = est.P[2:, :2]
    P_bb = est.P[2:, 2:]

    innovation = np.array(latlon_to_xy(surface_location.latitude, surface_location.longitude, est.location.latitude, est.location.longitude))
    # Tiny regulariser, not a modelled measurement noise -- keeps the inverse defined
    P_XX_inv = np.linalg.inv(P_XX + np.eye(2) * 1e-10)
    nis = float(innovation @ P_XX_inv @ innovation)
    bias_correction = P_bX @ P_XX_inv @ innovation

    P_new = np.zeros((4, 4))
    P_new[:2, :2] = np.eye(2) * 1e-10
    P_new[2:, 2:] = P_bb - P_bX @ P_XX_inv @ P_Xb

    updated = EstimatedState(
        time=surface_time,
        location=surface_location,
        depth=0.0,
        phase="communicating",
        bx=est.bx + bias_correction[0],
        by=est.by + bias_correction[1],
        P=P_new,
    )
    return updated, nis


def surface_trigger(float_id, surface_location: Location, surface_time, cycle_number, debug=True):

    records = Records(float_id)
    config, _, prev_surfaced_state, prev_action, forecast, bathymetry = load_environment(records)
    ou = records.get_ou()

    estimated_trajectory = simulate_action(prev_surfaced_state, forecast, bathymetry, prev_action, config, ou)

    # Drift on surface if the float surfaced later than estimated
    if estimated_trajectory[-1].time < surface_time - timedelta(minutes=config.max_time_difference):
        drift_on_surface_action = ControlAction(
            drifting_depth=0,
            duration_hours=(surface_time - estimated_trajectory[-1].time).total_seconds() / 3600,
        )
        logger.info("Simulated float surfaced %.1f minutes before the real float, adding drifting time on the surface", (surface_time - estimated_trajectory[-1].time).total_seconds() / 60)
        estimated_trajectory.extend(simulate_action(estimated_trajectory[-1], forecast, bathymetry, drift_on_surface_action, config, ou))

    if estimated_trajectory[-1].time > surface_time:
        # Overshooting by up to one timestep is normal, only warn beyond max_time_difference
        minutes_late = (estimated_trajectory[-1].time - surface_time).total_seconds() / 60
        if minutes_late > config.max_time_difference:
            logger.warning("Simulated float surfaced %.1f minutes after the real float, likely an error in the action", minutes_late)
        estimated_trajectory[-1].time = surface_time

    estimated_surfacing = estimated_trajectory[-1]
    updated_state, nis = kf_update_step(estimated_surfacing, surface_location, surface_time)
    estimated_trajectory.append(updated_state)
    logger.info("Surfaced at %s: %s, NIS %.2f, bias (%.4f, %.4f)", surface_time, surface_location, nis, updated_state.bx, updated_state.by)

    if not debug:
        records.save_surfacing(updated_state, cycle_number)
        records.save_trajectory(estimated_trajectory, cycle_number)
        records.update_errors(cycle_number, estimated_surfacing, updated_state, nis, ou)

    return updated_state


def initialise_float(float_id, config: dict, starting_location, starting_time, starting_action, cycle_number=0, rewrite=False):
    """Creates a new float from its config.json contents, first surfacing and first action.

    The bathymetry file named in the config still has to be copied into the float folder.
    """
    records = Records.create(float_id, config, rewrite)
    starting_state = EstimatedState(time=starting_time, location=starting_location, depth=0.0, phase="communicating")
    records.save_surfacing(starting_state, cycle_number)
    records.save_action(starting_action, cycle_number, source="initial")
    if not records.ou_path.exists():
        records.write_ou(OU(), cycle_number)
    logger.info("Initialised float %s at %s, cycle %s", float_id, starting_location, cycle_number)
    return records


def setup_logging(debug):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format=LOG_FORMAT)
    for library in ("botocore", "s3transfer", "urllib3", "copernicusmarine", "argopy"):
        logging.getLogger(library).setLevel(logging.WARNING)


def main(float_id, surface_location, surface_time, cycle_number, debug=True) -> None:
    setup_logging(debug)

    surface_trigger(float_id, surface_location, surface_time, cycle_number, debug)
    get_action(float_id, debug)


if __name__ == "__main__":
    #main(3902607, Location(latitude=56.3454, longitude=19.3095), datetime(2026, 7, 30, 5, 30), cycle_number=1)
    print(get)