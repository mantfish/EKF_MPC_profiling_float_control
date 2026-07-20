
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd

from data_handler import *
from control import KFMPC, compute_jacobian

logger = logging.getLogger(__name__)


def _state_row(state: EstimatedState) -> dict:
    "Flattens an EstimatedState into a parquet-friendly row (see read_state/write_state)."
    return {
        "time": state.time,
        "latitude": state.location.latitude,
        "longitude": state.location.longitude,
        "depth": state.depth,
        "phase": state.phase,
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "bx": state.bx,
        "by": state.by,
        "P": state.P.tolist(),
    }

def _row_to_state(row) -> EstimatedState:
    "Inverse of _state_row: builds an EstimatedState from a single state-history row."
    return EstimatedState(
        time=row["time"],
        location=Location(latitude=row["latitude"], longitude=row["longitude"]),
        depth=row["depth"],
        phase=row["phase"],
        x=row["x"],
        y=row["y"],
        z=row["z"],
        bx=row["bx"],
        by=row["by"],
        P=np.array(list(row["P"]), dtype=float),
    )

def surface_trigger(email_file: str):

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
        flow_weight=config.flow_weight,
        distance_weight=config.distance_weight,
        science_weight=config.science_weight,
        variance_weight=config.variance_weight,
        radius_std_m=config.radius_std,
        time_horizon_hours=int(config.flow_time_horizon_hours),
    )

    # Convert location to x,y coordainates relative to target location
    surface_x, surface_y = latlon_to_xy(location.latitude, location.longitude, start_lat, start_lon)

    estimated_state_history, _, _, _ = update_state(float_id)

    # Innovation step

    estimated_surface_state = _row_to_state(estimated_state_history.iloc[-1])

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
        candidate_action = get_action(action_name, config)
        history, u_interp, v_interp, bounds = update_state(float_id, candidate_action, persist=False)
        final_state = _row_to_state(history.iloc[-1])
        if final_state.phase != "communicating":
            action_cost[action_name] = np.nan
        else:
            total, *_ = Control.evaluate_cost(
                final_state, innovated_state, candidate_action, u_interp, v_interp, bounds, start_lat, start_lon
            )
            action_cost[action_name] = total

    feasible_costs = {name: cost for name, cost in action_cost.items() if not np.isnan(cost)}
    if not feasible_costs:
        raise RuntimeError(f"No feasible action found for float {float_id}: every candidate action failed to reach 'communicating' within its simulation horizon.")
    selected_action_name = min(feasible_costs, key=feasible_costs.get)

    write_surfacing_action_log(float_id, selected_action_name, time_of_transmission, location, estimated_surface_state, innovated_state, nis, action_cost)

    update_state(float_id)  # Update the state after the innovation step

    return selected_action_name




def update_state(float_id: int, action: ControlAction | None = None, persist: bool = True):

    logger.info("Update state called for float ID: %s at %s", float_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    config = read_config(float_id)
    estimated_state_history = read_state(float_id)
    last_surfacing_and_action = read_last_surfacing_and_action(float_id)
    last_surfaced_timestamp = pd.Timestamp(last_surfacing_and_action["surfaced_timestamp"]).tz_localize(None)
    last_surfaced_location = Location(
        latitude=last_surfacing_and_action["surfaced_location"]["latitude"],
        longitude=last_surfacing_and_action["surfaced_location"]["longitude"],
    )

    start_lat, start_lon = config.target_lat, config.target_lon

    if action is None:
        action = get_action(last_surfacing_and_action["action_sent"], config)
        logger.info("Using last action stored in config")

    simulation_end_time = last_surfaced_timestamp + timedelta(hours=action.duration_hours + config.estimated_tranmission_time_hours)

    bathy_ds = load_bathymetry(config.data_dir / config.bathymetry_file_name)
    bathy_interp = build_bathymetry_interpolator(bathy_ds)


    if config.model == "CMEMS":
        forecast_ds = download_cmems_data_around_float(last_surfaced_location, last_surfaced_timestamp, simulation_end_time)
    else:
        raise ValueError(f"Unknown model specified in config: {config.model}")

    u_interp, v_interp, bounds = build_uv_interpolators(forecast_ds)
    first_time = pd.Timestamp(forecast_ds.time[0].values)

    if first_time < last_surfaced_timestamp:
        # TODO check something here about what happens if it's drifting on surface
        start_simulation_time = last_surfaced_timestamp
        logger.info("Forecast data starts before last surfacing. Using last surfacing time as start of simulation: %s", start_simulation_time)
    else:
        start_simulation_time = first_time
        logger.info("Forecast data starts after last surfacing. Using forecast start time as start of simulation: %s", start_simulation_time)

    closest_row = pd.merge_asof( pd.DataFrame({'time': [start_simulation_time]}),
                        estimated_state_history,
                        on='time',
                        direction='forward' )
    
    # TODO check what happens with depth and stuff
    estimated_future_state = _row_to_state(closest_row.iloc[0])

    logger.info("Simulation start time: %s, Simulation end time: %s", start_simulation_time, simulation_end_time)

    estimated_future_state.phase = "communicating"
    list_of_estimated_future_states = []
    while estimated_future_state.time < simulation_end_time:

        lat, lon = xy_to_latlon(estimated_future_state.x, estimated_future_state.y, start_lat, start_lon)

        bottom_depth = bathy_interp(lat, lon)

        # Phase transitions
        if estimated_future_state.phase == "communicating":
            if estimated_future_state.time > start_simulation_time + timedelta(hours=config.estimated_tranmission_time_hours):
                estimated_future_state.phase = "descending"

        if estimated_future_state.phase == "descending":
            if estimated_future_state.depth >= action.parking_depth or estimated_future_state.depth >= bottom_depth:
                estimated_future_state.phase = "parking"
            else:
                estimated_future_state.depth += config.descent_speed_m_per_s * config.vertical_dt
                estimated_future_state.time += timedelta(seconds=config.vertical_dt)

        if estimated_future_state.phase == "parking":
            if estimated_future_state.time <= simulation_end_time - timedelta(seconds=estimated_future_state.depth / config.ascent_speed_m_per_s):
                estimated_future_state.phase = "ascending"

        if estimated_future_state.phase == "ascending":
            estimated_future_state.depth = max(0.0, estimated_future_state.depth - config.ascent_speed_m_per_s * config.vertical_dt)
            estimated_future_state.time += timedelta(seconds=config.vertical_dt)
            if estimated_future_state.depth <= 0.0:
                estimated_future_state.phase = "communicating"

        # Horizontal drift
        parked_on_bottom = estimated_future_state.phase == "parking" and estimated_future_state.depth >= bottom_depth
        if not parked_on_bottom:
            u, v = query_uv(u_interp, v_interp, bounds, estimated_future_state.time, estimated_future_state.depth, estimated_future_state.location.latitude, estimated_future_state.location.longitude)
            F = compute_jacobian(estimated_future_state.x, estimated_future_state.y, estimated_future_state.depth, estimated_future_state.time,
                        u_interp, v_interp, bounds, start_lat, start_lon)  # pre-step
            # Process noise on position only (first 2 components of Q)
            estimated_future_state.x += (u + estimated_future_state.bx) * config.parking_dt
            estimated_future_state.y += (v + estimated_future_state.by) * config.parking_dt
            Phi = np.eye(4) + F * config.parking_dt
            estimated_future_state.P = Phi @ estimated_future_state.P @ Phi.T + np.diag(config.Q) * config.parking_dt
            estimated_future_state.time += timedelta(seconds=config.parking_dt)


        estimated_future_state.location = Location(*xy_to_latlon(estimated_future_state.x, estimated_future_state.y, start_lat, start_lon))
        logger.info("Estimated future state at time %s: Location (%f, %f), Depth %f, Phase %s", estimated_future_state.time, estimated_future_state.location.latitude, estimated_future_state.location.longitude, estimated_future_state.depth, estimated_future_state.phase)
        list_of_estimated_future_states.append(estimated_future_state)


    future_estimated_states_df = pd.DataFrame([_state_row(state) for state in list_of_estimated_future_states])

    estimated_state_history = pd.concat([estimated_state_history[estimated_state_history['time'] < start_simulation_time], future_estimated_states_df], ignore_index=True)
    estimated_state_history = estimated_state_history.sort_values('time').reset_index(drop=True)

    if persist:
        write_state(float_id, estimated_state_history)  # Save the updated state history back to the file
    return estimated_state_history, u_interp, v_interp, bounds


    
    

    
    

    
    

    


    


    


    

    



def main():
    print("Hello from server!")


if __name__ == "__main__":
    main()
