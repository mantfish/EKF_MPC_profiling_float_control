
import datetime
from os import path
import logging
from turtle import pd
from data_handler import *
from control import KFMPC

logger = logging.getLogger(__name__)

def surface_trigger(email_file: path):

    new_data = read_json(email_file)

    float_id = new_data["float_id"]
    location = new_data["location"]
    time_of_transmission = new_data["time_of_transmission"]

    logger.info("Read Input at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Float ID: %s, Location: %s, Time of Transmission: %s", float_id, location, time_of_transmission)

    config = read_config(float_id)
    Control = KFMPC(

    )

    # Convert location to x,y coordainates relative to target location
    surface_x, surface_y = latlon_to_xy(location["latitude"], location["longitude"], config["target_lat"], config["target_lon"])


    estimated_state_history = update_state(float_id)

    # Innovation step

    estimated_surface_state = estimated_state_history[-1]

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

    action_cost = {}
    for action in config.actions:
        final_state, (u_interp, v_interp) = update_state(float_id, action, write_state=False)[-1]
        # Check if completed
        if final_state.phase != "COMMUNICATING":
            action_cost[action.name] = np.Nan
        else:
            action_cost[action.name] = Control.evaluate_cost(
                final_state, innovated_state, action, u_interp, v_interp
            )
    



    write_surfacing_action_log(float_id, control time_of_transmission, location, innovated_state, nis) # TODO fix
    
    estimated_state_history.append(innovated_state)

    update_state(float_id)  # Update the state after the innovation step

    return control




def update_state(float_id: int, action: None, write_state = True):
    
    logger.info("Update state called for float ID: %s at %s", float_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    config = read_config(float_id)
    estimated_state_history = read_state(float_id)
    last_surfacing_and_action= read_last_surfacing_and_action(float_id)

    if action is None
        action = get_action(last_surfacing_and_action["action_sent"], float_id)
        logger.info("Using last action stored in config")
    else:
        action = action

    bathy_ds = load_bathymetry(config.data_dir / config["bathymetry_file_name"])
    bathy_interp = build_bathymetry_interpolator(bathy_ds)


    if config["model"] == "CMEMS":
        forecast_ds = download_cmems_data_around_float(last_surfacing_and_action["location"])
    else:
        raise ValueError(f"Unknown model specified in config: {config['model']}")
    
    u_interp, v_interp, bounds = build_uv_interpolators(forecast_ds)
    first_time = forecast_ds.time[0].values

    if first_time < last_surfacing_and_action["surfaced_timestamp"]:
        # TODO check something here about what happens if it's drifting on surface
        start_simulation_time = last_surfacing_and_action["surfaced_timestamp"]
        logger.info("Forecast data starts before last surfacing. Using last surfacing time as start of simulation: %s", start_simulation_time)
    else:
        start_simulation_time = forecast_ds.time[0].values
        logger.info("Forecast data starts after last surfacing. Using forecast start time as start of simulation: %s", start_simulation_time)
    
    closest_row = pd.merge_asof( pd.DataFrame({'time': [start_simulation_time]}),
                        estimated_state_history,
                        on='time',
                        direction='forward' )
    
    # TODO check what happens with depth and stuff
    estimated_future_state = EstimatedState(
        time=closest_row['time'].values[0],
        location=Location(latitude=closest_row['latitude'].values[0], longitude=closest_row['longitude'].values[0]),
        depth=closest_row['depth'].values[0],
        phase=closest_row['phase'].values[0],
        x=closest_row['x'].values[0],
        y=closest_row['y'].values[0],
        z=closest_row['z'].values[0],
        bx=closest_row['bx'].values[0],
        by=closest_row['by'].values[0],
        P=np.array(closest_row['P'].values[0])
    )

    simulation_end_time = last_surfacing_and_action["surfaced_timestamp"] + timedelta(hours=action.duration_hours + config["estimated_tranmission_time_hours"])

    logger.info("Simulation start time: %s, Simulation end time: %s", start_simulation_time, simulation_end_time)

    estimated_future_state.phase = COMMUNICATING
    list_of_estimated_future_states = []
    while estimated_future_state.time < simulation_end_time:
        
        lat, lon = xy_to_latlon(estimated_future_state.x, estimated_future_state.y, config["target_lat"], config["target_lon"])
        
        bottom_depth = bathy_interp(lat, lon)

        # Phase transitions
        if estimated_future_state.phase == COMMUNICATING:
            if estimated_future_state.time > simulation_start_time + datetime.timedelta(hours=config["estimated_tranmission_time_hours"]):
                estimated_future_state.phase = DESCENDING

        if estimated_future_state.phase == _DESCENDING:
            if estimated_future_state.depth >= action.parking_depth or estimated_future_state.depth >= bottom_depth:
                estimated_future_state.phase = _PARKING
            else:
                estimated_future_state.depth += action.descent_speed_ms * config["vertical_dt"]
                estimated_future_state.time += timedelta(seconds=config["vertical_dt"])

        if estimated_future_state.phase == _PARKING:
            if estimate_future_state.time <= simulation_end_time - estimated_future_state.depth / config["ascent_speed_ms"]:
                estimated_future_state.phase = _ASCENDING

        if estimated_future_state.phase == _ASCENDING:
            estimated_future_state.depth = max(0.0, estimated_future_state.depth - config["ascent_speed_ms"]* config["vertical_dt"])
            estimated_future_state.time += timedelta(seconds=config["vertical_dt"])

        # Horizontal drift
        parked_on_bottom = estimated_future_state.phase == _PARKING and estimated_future_state.depth >= bottom_depth
        if not parked_on_bottom:
            u, v = query_uv(u_interp, v_interp, bounds, estimated_future_state.time, estimated_future_state.depth, estimated_future_state.location.latitude, estimated_future_state.location.longitude)
            F = _compute_jacobian(estimated_future_state.x, estimated_future_state.y, estimated_future_state.depth, estimated_future_state.time,
                        u_interp, v_interp, start_lat, start_lon)  # pre-step
            # Process noise on position only (first 2 components of Q)
            estimated_future_state.x += (u + estimated_future_state.bx) * config["parking_dt"]
            estimated_future_state.y += (v + estimated_future_state.by) * config["parking_dt"]
            Phi = np.eye(4) + F * config["parking_dt"]
            estimated_future_state.P = Phi @ estimated_future_state.P @ Phi.T + Q * config["parking_dt"]
            estimated_future_state.time += timedelta(seconds=config["parking_dt"])


        estimated_future_state.location = Location(*xy_to_latlon(estimated_future_state.x, estimated_future_state.y, start_lat, start_lon))
        logger.info("Estimated future state at time %s: Location (%f, %f), Depth %f, Phase %s", estimated_future_state.time, estimated_future_state.location.latitude, estimated_future_state.location.longitude, estimated_future_state.depth, estimated_future_state.phase)
        list_of_estimated_future_states.append(estimated_future_state)

    
    future_estimated_states_df = pd.DataFrame([vars(obj) for state in list_of_estimated_future_states])  # or [obj.__dict__ for obj in objects]

    estimated_state_history = pd.concat([estimated_state_history[estimated_state_history['time'] < simulation_start_time], future_estimated_states_df], ignore_index=True)
    estimated_state_history = estimated_state_history.sort_values('time').reset_index(drop=True)

    if write_state is True:
        write_state(float_id, estimated_state_history)  # Save the updated state history back to the file
    return estimated_state_history, (u_interp, v_interp)


    
    

    
    

    
    

    


    


    


    

    



def main():
    print("Hello from server!")


if __name__ == "__main__":
    main()
