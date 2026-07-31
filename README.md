# Argo Piloting Server

EKF/MPC pipeline that pilots an Argo float toward a target location using CMEMS current forecasts and bathymetry data.

## Float layout

Each float has a directory under `floats/<float_id>/`:

```
floats/<float_id>/
├── config.json                  # float parameters (see below)
├── bathymetry.nc                # local bathymetry grid
├── estimated_state.csv          # EKF state history
├── surfacing_action_log.json    # log of past surfacings and chosen actions
└── logs/                        # per-run log files
```

## config.json

One JSON object per float, mirrored by the `Config` dataclass in `helpers.py`.

- `float_id` — integer float ID; must match the directory name under `floats/`.
- `target_lat`, `target_lon` — target destination, decimal degrees.
- `radius_std` — standard deviation (m) of the Gaussian used to score proximity to the target.
- `flow_time_horizon_hours` — how far ahead (hours) the MPC looks when scoring current/flow.
- `flow_weight`, `distance_weight`, `science_weight`, `variance_weight` — relative weights in the MPC cost function.
- `vertical_dt`, `parking_dt` — integration timestep (seconds) while ascending/descending vs. while parked/drifting.
- `bathymetry_file_name` — filename of the bathymetry NetCDF file, resolved relative to the float's own directory.
- `estimated_tranmission_time_hours` — assumed time (hours) the float spends at the surface transmitting [sic, matches the field name in code].
- `ascent_speed_m_per_s`, `descent_speed_m_per_s` — vertical speed (m/s).
- `process_noise_diagonal` — 4-element diagonal of the EKF process noise matrix `Q`, in state order `[x, y, bx, by]`.
- `possible_actions` — list of candidate actions the MPC chooses between at each surfacing. Each entry:
  - `name` — unique identifier, referenced from `surfacing_action_log.json`.
  - `duration_hours` — how long the action lasts.
  - `depth_m` — depth (m) to park/drift at.
  - `action_type` — `"drift"` or `"park"`, descriptive only (not read by the pipeline).
  - `science_cost` — 0 (no science value) to 1 (full science value); scored against proximity to target.

Numbers are floats unless noted; JSON does not support trailing commas or comments, so keep entries comma-separated and quoted as in `floats/123456/config.json`.
