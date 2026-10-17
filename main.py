
import copy
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from control import KFMPC, compute_jacobian
from data_handler import *
from helpers import ControlAction, EstimatedState, Location

logger = logging.getLogger(__name__)



def simulate_action(starting_state, forecast, bathymetry, action, config, OU):

    # Check that forecast data covers the entire simulation period
    if not forecast.check_forecast_coverage(starting_state.time, action.duration_hours):
        raise ValueError("Forecast data does not cover the entire simulation period.")
    if not bathymetry.check_bathymetry_coverage(starting_state.location, config.max_drift):
        raise ValueError("Bathymetry data does not cover the entire simulation region.")


    simulated_state = copy.deepcopy(starting_state)
    simulated_end_time = starting_state.time + timedelta(hours=action.duration_hours)

    decay_e, bias_var_e, decay_n, bias_var_n = OU.get_ou_step()
    Q_step = np.diag([0.0, 0.0, bias_var_e, bias_var_n]) 

    states = []

    while simulated_state.time < simulated_end_time:

        dx, dy = 0.0, 0.0
        u, v = forecast.query_uv(simulated_state.time, simulated_state.depth, simulated_state.location, bias=(simulated_state.bx, simulated_state.by))
        bottom_depth = bathymetry.get_depth(simulated_state.location)

        # Phase transitions
        if simulated_state.phase == "communicating":
            if simulated_state.time >= starting_state.time + timedelta(hours=config.estimated_tranmission_time_hours):
                simulated_state.phase = "descending"

        if simulated_state.phase == "descending":
            if simulated_state.depth >= action.drifting_depth and action.grounding is False: 
                simulated_state.phase = "drifting"
                if simulated_state.depth > bottom_depth:
                    logger.warning("Float %s is drifting at depth %.2f m, which is below the bathymetry depth %.2f m at location (%f, %f).", float_id, simulated_state.depth, bottom_depth, lat, lon)
            if simulated_state.depth >= bottom_depth and action.grounding is True:
                simulated_state.phase = "grounded"

        if simulated_state.phase == "drifting":
            if simulated_state.time > simulated_end_time - timedelta(seconds=simulated_state.depth / config.ascent_speed_m_per_s):
                simulated_state.phase = "ascending"
            if simulated_state.depth >= bottom_depth:
                simulated_state.phase = "grounded" 
                logger.warning("Float %s is drifting at depth %.2f m, which is below the bathymetry depth %.2f m at location (%f, %f). so has now grounded", float_id, simulated_state.depth, bottom_depth, lat, lon)

        if simulated_state.phase == "ascending" and simulated_state.depth <= 0.0:
            simulated_state.phase = "communicating"

        # Vertical motion + this iteration's timestep. Ascending/descending move in
        # vertical_dt increments; every other phase (communicating/parking) advances
        # by parking_dt, matching the granularity of the horizontal drift below.
        if simulated_state.phase == "descending":
            simulated_state.depth += config.descent_speed_m_per_s * config.vertical_dt
            dt = config.vertical_dt
        elif simulated_state.phase == "ascending":
            simulated_state.depth = max(0.0, simulated_state.depth - config.ascent_speed_m_per_s * config.vertical_dt)
            dt = config.vertical_dt
        else:
            dt = config.parking_dt

        if not simulated_state.on_seabed:
            F = compute_jacobian(simulated_state, forecast, ou)  # pre-step
            simulated_state.x += simulated_state.current_u * dt
            simulated_state.y += simulated_state.current_v * dt
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
        simulated_state.update_location(dx, dy)        

        logger.info("Simulated state at time %s: Location (%f, %f), Depth %f, Phase %s", simulated_state.time, simulated_state.location.latitude, simulated_state.location.longitude, simulated_state.depth, simulated_state.phase)
        states.append(simulated_state)

    return states


def get_action(float_id):

    config = read_config(float_id)
    prev_surfaced_state, prev_action = read_surfacing_and_action(float)
    max_action_hours = max(for action in config.actions times = [], times.append(duration_hours))
    time_buffer_hours = config.buffer_time
    OU = read_ou(float_id)
    mpc = MPC(config.target_location)

    forecast = Forecast(config, prev_surfaced_state.location, (prev_surfaced_state.time - time_buffer_hours, prev_surfaced_state + 2*max_action_hours + buffer ))
    bathymtery = Bathymetry(prev_surfaced_state.location, config.max_drift)



    estimated_trajectory = simulate_action(prev_surfaced_state, forecast, bathymetry, prev_action OU)
    estimated_future_surfacing = estimated_trajectory[-1]

    best_action = None
    best_action_cost = np.inf
    for action in config.possible_actions:
        logger.info("starting future simulation of ..")
        MPC_trajectory = simulate_action(estimated_future_surfacing.copy(), forecast, bathymetry, OU)
        cost = mpc.evaluate_cost(expected_surfacing = MPC_trajectory[-1], last_surfacing = estimated_future_surfacing, action = action)
        
        if cost < best_action_cost:
            best_action = action
            best_action_cost = cost

    return best_action


def kf_update_step(est, real) -> tuple[EstimatedState, float]:
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

    innovation = np.array([real.x - est.x, real.y - est.y])
    # Tiny regulariser, not a modelled measurement noise -- keeps the inverse defined
    P_XX_inv = np.linalg.inv(P_XX + np.eye(2) * 1e-10)
    nis = float(innovation @ P_XX_inv @ innovation)
    bias_correction = P_bX @ P_XX_inv @ innovation

    P_new = np.zeros((4, 4))
    P_new[:2, :2] = np.eye(2) * 1e-10
    P_new[2:, 2:] = P_bb - P_bX @ P_XX_inv @ P_Xb

    updated = EstimatedState(
        time=real.time,
        depth=0.0,
        phase=_COMMUNICATING,
        x=real.x,
        y=real.y,
        bx=est.bx + bias_correction[0],
        by=est.by + bias_correction[1],
        P=P_new,
    )
    return updated, nis



def surface_trigger(float_id, surface_location):
    #Todo make sure difference isn't too big

    config = read_config(float_id)
    prev_surfaced_state, prev_action = read_surfacing_and_action(float)
    max_action_hours = max(for action in config.actions times = [], times.append(duration_hours))
    time_buffer_hours = config.buffer_time
    OU = read_ou(float_id)

    forecast = Forecast(config, prev_surfaced_state.location, (prev_surfaced_state.time - time_buffer_hours, prev_surfaced_state + 2*max_action_hours + buffer ))
    bathymetry = Bathymetry(prev_surfaced_state.location, config.max_drift)

    estimated_trajectory = simulate_action(prev_surfaced_state, forecast, bathymetry, prev_action OU)
    estimated_surfacing = estimated_trajectory[-1]

    updated_state = kf_update_step(estimated_surfacing, surface_location)

    # save past
    save


    









def get_updated_state(real_surfaced_state):

    new_data = read_json(email_file)

    float_id = new_data["float_id"]
    location = Location(latitude=new_data["location"]["latitude"], longitude=new_data["location"]["longitude"])
    time_of_transmission = pd.Timestamp(new_data["time_of_transmission"]).tz_localize(None)

    logger.info("Read Input at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Float ID: %s, Location: %s, Time of Transmission: %s", float_id, location, time_of_transmission)

    config = read_config(float_id)
    start_lat, start_lon = config.target_lat, config.target_lon

    Control = KFMPC(
        target_location=[config.target_lat, config.target_lon],
    )

    # Convert location to x,y coordainates relative to target location
    surface_x, surface_y = latlon_to_xy(location.latitude, location.longitude, start_lat, start_lon)

    logger.info("updating state")
    estimated_state_history, _, _, _, forecast_ds = update_state(float_id)

    # Innovation step

    estimated_surface_state = row_to_state(estimated_state_history.iloc[-1])

    P_XX = estimated_surface_state.P[:2, :2]
    P_Xb = estimated_surface_state.P[:2, 2:]
    P_bX = estimated_surface_state.P[2:, :2]
    P_bb = estimated_surface_state.P[2:, 2:]

    innovation = np.array([surface_x - estimated_surface_state.x, surface_y - estimated_surface_state.y])
    P_XX_inv = np.linalg.inv(P_XX + np.eye(2) * 1e-10)
    nis = float(innovation @ P_XX_inv @ innovation)
    bias_correction = P_bX @ P_XX_inv @ innovation


    P_bb_new = P_bb - P_bX @ P_XX_inv @ P_Xb
    P_new = np.zeros((4, 4))
    P_new[:2, :2] = np.eye(2) * 1e-10
    P_new[2:, 2:] = P_bb_new


    innovated_state = EstimatedState(
        time=time_of_transmission,
        location=estimated_surface_state.location,
        depth=0.0,
        phase="communicating",
        x=surface_x,
        y=surface_y,
        bx=estimated_surface_state.bx + bias_correction[0],
        by=estimated_surface_state.by + bias_correction[1],
        P=P_new
    )

    # Score every candidate action by simulating it forward and evaluating the MPC cost.
    action_cost = {}
    for action_spec in config.possible_actions:
        action_name = action_spec["name"]
        print(action_name)
        logger.info("evaluting action: %s", action_name)
        candidate_action = get_action(action_name, config)
        history, u_interp, v_interp, bounds, forecast_ds = update_state(float_id, candidate_action, persist=False, forecast_ds=forecast_ds)
        final_state = row_to_state(history.iloc[-1])
        if final_state.phase != "communicating":
            action_cost[action_name] = np.nan
            logger.warning("Evalutional of action %s did not finish correctly, float did not end communicating therefore skipped", action_name)
        else:
            total, *_ = Control.evaluate_cost(
                final_state, innovated_state, candidate_action, u_interp, v_interp, bounds, start_lat, start_lon
            )
            action_cost[action_name] = total
            logger.info("Action: %s is scored %.3f", action_name, total)

    feasible_costs = {name: cost for name, cost in action_cost.items() if not np.isnan(cost)}
    if not feasible_costs:
        raise RuntimeError(f"No feasible action found for float {float_id}: every candidate action failed to reach 'communicating' within its simulation horizon.")
    selected_action_name = min(feasible_costs, key=lambda name: feasible_costs[name])

    write_surfacing_action_log(float_id, selected_action_name, time_of_transmission, location, estimated_surface_state, innovated_state, nis, innovation, action_cost)

    # Update the state after the innovation step, reusing the forecast already
    # downloaded above instead of triggering a second full CMEMS download.
    update_state(float_id, forecast_ds=forecast_ds)

    return selected_action_name







def update_state(
    float_id: int,
    action: ControlAction | None = None,
    persist: bool = True,
    forecast_ds: xr.Dataset | None = None,
) -> tuple[pd.DataFrame, RegularGridInterpolator, RegularGridInterpolator, dict[str, float], xr.Dataset]:

    logger.info("Update state called for float ID: %s at %s", float_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    config = read_config(float_id)
    estimated_state_history = read_state(float_id)
    last_surfacing_and_action = max(read_surfacing_and_action(float_id), key=lambda entry: entry["surfaced_timestamp"])
    last_surfaced_timestamp = pd.Timestamp(last_surfacing_and_action["surfaced_timestamp"]).tz_localize(None)
    last_surfaced_location = Location(
        latitude=last_surfacing_and_action["surfaced_location"]["latitude"],
        longitude=last_surfacing_and_action["surfaced_location"]["longitude"],
    )

    logger.info("The last action that we will simulate forward is: %s", last_surfacing_and_action["action_sent"])

    start_lat, start_lon = config.target_lat, config.target_lon

    if action is None:
        action = get_action(last_surfacing_and_action["action_sent"], config)
        logger.info("Using last action stored in config")

    simulation_end_time = last_surfaced_timestamp + timedelta(hours=action.duration_hours + config.estimated_tranmission_time_hours)

    logger.info("The next surfacing time is : %s", simulation_end_time)

    # Bathymetry files are basin-wide (e.g. the whole Baltic), but a float only
    # ever operates within max_drift of its last surfaced location, so crop to
    # that same box before materializing it — mirrors the CMEMS request below.
    float_region = define_region_aroung_float(last_surfaced_location, config.max_drift or MAX_DRIFT)

    bathy_ds = load_bathymetry(config.data_dir / "bathymetry.nc")
    bathy_interp = build_bathymetry_interpolator(bathy_ds, bbox=float_region)


    if config.model_type != "CMEMS":
        raise ValueError(f"Unknown model specified in config: {config.model_type}")

    bathy_lat_min = float(bathy_ds.lat.min())
    bathy_lat_max = float(bathy_ds.lat.max())
    bathy_lon_min = float(bathy_ds.lon.min())
    bathy_lon_max = float(bathy_ds.lon.max())

    if forecast_ds is None:
        # Download a window wide enough to cover the longest candidate action, so the
        # same forecast can be reused across every update_state call in a single
        # surface_trigger invocation instead of re-downloading per candidate action.
        max_action_duration_hours = max(a["duration_hours"] for a in config.possible_actions)
        forecast_window_end = last_surfaced_timestamp + timedelta(hours=max_action_duration_hours + config.estimated_tranmission_time_hours)

        forecast_ds = download_cmems_data_around_float(last_surfaced_location, last_surfaced_timestamp, forecast_window_end, config)

    u_interp, v_interp, bounds = build_uv_interpolators(forecast_ds)
    earliest_forecast_datetime = pd.Timestamp(forecast_ds.time[0].values)

    if simulation_end_time < earliest_forecast_datetime:
        raise TypeError("The predicted next surfacing time is before forecast time begins.")
        # TODO implement fix

    if earliest_forecast_datetime < last_surfaced_timestamp:
        # TODO check something here about what happens if it's drifting on surface
        start_simulation_time = last_surfaced_timestamp
        logger.info("Forecast data starts before last surfacing. Using last surfacing time as start of simulation: %s", start_simulation_time)
    else:
        start_simulation_time = earliest_forecast_datetime
        logger.info("Forecast data starts after last surfacing. Using forecast start time as start of simulation: %s", start_simulation_time)



    if estimated_state_history.empty:
        # No persisted state yet (first run for this float): bootstrap from the
        # estimated_state recorded alongside the last surfacing.
        logger.info("No estimated state history found. Bootstrapping from last surfacing's estimated_state.")
        estimated_future_state = dict_to_state(last_surfacing_and_action["estimated_state"])
    else:
        closest_row = pd.merge_asof( pd.DataFrame({'time': [start_simulation_time]}),
                            estimated_state_history,
                            on='time',
                            direction='forward' )
        logger.info(closest_row)

        # TODO check what happens with depth and stuff
        estimated_future_state = row_to_state(closest_row.iloc[0])

    logger.info("Simulation start time: %s, Simulation end time: %s", start_simulation_time, simulation_end_time)

    estimated_future_state.phase = "communicating"
    list_of_estimated_future_states = []
    while estimated_future_state.time < simulation_end_time:

        lat, lon = xy_to_latlon(estimated_future_state.x, estimated_future_state.y, start_lat, start_lon)

        if not (bathy_lat_min <= lat <= bathy_lat_max and bathy_lon_min <= lon <= bathy_lon_max):
            raise RuntimeError(
                f"Float {float_id} left the bathymetry region at ({lat:.4f}, {lon:.4f}); "
                f"region covers latitude [{bathy_lat_min:.4f}, {bathy_lat_max:.4f}] "
                f"and longitude [{bathy_lon_min:.4f}, {bathy_lon_max:.4f}]."
            )

        bottom_depth = bathy_interp(lat, lon)

        # Phase transitions
        if estimated_future_state.phase == "communicating":
            if estimated_future_state.time >= start_simulation_time + timedelta(hours=config.estimated_tranmission_time_hours):
                estimated_future_state.phase = "descending"

        if estimated_future_state.phase == "descending":
            if estimated_future_state.depth >= action.drifting_depth and action.grounding is False: 
                estimated_future_state.phase = "drifting"
                if estimated_future_state.depth > bottom_depth:
                    logger.warning("Float %s is drifting at depth %.2f m, which is below the bathymetry depth %.2f m at location (%f, %f).", float_id, estimated_future_state.depth, bottom_depth, lat, lon)
            if estimated_future_state.depth >= bottom_depth and action.grounding is True:
                estimated_future_state.phase = "grounded"

        if estimated_future_state.phase == "drifting":
            if estimated_future_state.time > simulation_end_time - timedelta(seconds=estimated_future_state.depth / config.ascent_speed_m_per_s):
                estimated_future_state.phase = "ascending"

        if estimated_future_state.phase == "ascending" and estimated_future_state.depth <= 0.0:
            estimated_future_state.phase = "communicating"

        # Vertical motion + this iteration's timestep. Ascending/descending move in
        # vertical_dt increments; every other phase (communicating/parking) advances
        # by parking_dt, matching the granularity of the horizontal drift below.
        if estimated_future_state.phase == "descending":
            estimated_future_state.depth += config.descent_speed_m_per_s * config.vertical_dt
            dt = config.vertical_dt
        elif estimated_future_state.phase == "ascending":
            estimated_future_state.depth = max(0.0, estimated_future_state.depth - config.ascent_speed_m_per_s * config.vertical_dt)
            dt = config.vertical_dt
        else:
            dt = config.parking_dt

        # TODO implement that if the float runs into ground during drifting it stays?
        if not estimated_future_state.phase == "grounded":
            u, v = query_uv(u_interp, v_interp, bounds, estimated_future_state.time, estimated_future_state.depth, estimated_future_state.location.latitude, estimated_future_state.location.longitude)
            F = compute_jacobian(config.bias_decay_rate, estimated_future_state.x, estimated_future_state.y, estimated_future_state.depth, estimated_future_state.time,
                        u_interp, v_interp, bounds, start_lat, start_lon)  # pre-step
            # Process noise on bias only, not position (see config.Q / process_noise_bias)
            estimated_future_state.x += (u + estimated_future_state.bx) * dt
            estimated_future_state.y += (v + estimated_future_state.by) * dt
            Phi = np.eye(4) + F * dt
            estimated_future_state.P = Phi @ estimated_future_state.P @ Phi.T + np.diag(config.Q) * dt
        else:
            F = np.zeros((4, 4))
            F[2, 2] = -1 / config.bias_decay_rate
            F[3, 3] = -1 / config.bias_decay_rate
            Phi = np.eye(4) + F * dt
            estimated_future_state.P = Phi @ estimated_future_state.P @ Phi.T + np.diag(config.Q) * dt


        estimated_future_state.time += timedelta(seconds=dt)

        estimated_future_state.Q = config.Q
        estimated_future_state.location = Location(*xy_to_latlon(estimated_future_state.x, estimated_future_state.y, start_lat, start_lon))
        logger.info("Estimated future state at time %s: Location (%f, %f), Depth %f, Phase %s", estimated_future_state.time, estimated_future_state.location.latitude, estimated_future_state.location.longitude, estimated_future_state.depth, estimated_future_state.phase)
        list_of_estimated_future_states.append(copy.deepcopy(estimated_future_state))


    future_estimated_states_df = pd.DataFrame([state_to_row(state) for state in list_of_estimated_future_states])

    if estimated_state_history.empty:
        # A brand-new float has no persisted history yet (read_state returns a
        # columnless pd.DataFrame()), so there's nothing to filter — just use
        # the freshly-simulated states.
        estimated_state_history = future_estimated_states_df
    else:
        estimated_state_history = pd.concat([estimated_state_history[estimated_state_history['time'] < start_simulation_time], future_estimated_states_df], ignore_index=True)
    estimated_state_history = estimated_state_history.sort_values('time').reset_index(drop=True)

    if persist:
        write_state(float_id, estimated_state_history)  # Save the updated state history back to the file
    return estimated_state_history, u_interp, v_interp, bounds, forecast_ds
 

def main() -> None:
    print("Hello from server!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    surface_trigger("fake_surfacing_data.json")
