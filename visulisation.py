import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import compute_bias_series, compute_innovation_acf, compute_nis_series
from data_handler import read_config, read_state, read_surfacing_and_action, xy_to_latlon

ELLIPSE_POINTS = 40


def _ellipse_latlon(x: float, y: float, P: list, n_sigma: float, target_lat: float, target_lon: float) -> tuple[np.ndarray, np.ndarray]:
    "n_sigma uncertainty ellipse around (x, y) meters, projected to lat/lon."
    P_XX = np.asarray(P)[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eigh(P_XX)
    eigenvalues = np.clip(eigenvalues, 0, None)
    radii = n_sigma * np.sqrt(eigenvalues)

    angles = np.linspace(0, 2 * np.pi, ELLIPSE_POINTS)
    circle = np.stack([np.cos(angles), np.sin(angles)]) * radii[:, None]
    offsets = eigenvectors @ circle

    lats = np.empty(ELLIPSE_POINTS)
    lons = np.empty(ELLIPSE_POINTS)
    for i in range(ELLIPSE_POINTS):
        lats[i], lons[i] = xy_to_latlon(x + offsets[0, i], y + offsets[1, i], target_lat, target_lon)
    return lats, lons


def _sampled_state_history(state_history: pd.DataFrame, max_frames: int) -> pd.DataFrame:
    if len(state_history) <= max_frames:
        return state_history
    indices = np.linspace(0, len(state_history) - 1, max_frames).round().astype(int)
    indices = np.unique(indices)
    return state_history.iloc[indices].reset_index(drop=True)


def build_visualization_html(float_id: int, n_sigma: float = 2.0, max_frames: int = 150) -> str:
    "Builds a self-contained interactive HTML page for inspecting a float's EKF/MPC history."
    config = read_config(float_id)

    surfacing_entries = read_surfacing_and_action(float_id)
    surfacing_entries = sorted(surfacing_entries, key=lambda entry: entry["surfaced_timestamp"])
    surface_times = [pd.Timestamp(entry["surfaced_timestamp"]) for entry in surfacing_entries]
    surface_lats = [entry["surfaced_location"]["latitude"] for entry in surfacing_entries]
    surface_lons = [entry["surfaced_location"]["longitude"] for entry in surfacing_entries]

    state_history = read_state(float_id).sort_values("time").reset_index(drop=True)
    frames_df = _sampled_state_history(state_history, max_frames)

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "scattergeo", "rowspan": 2}, {"type": "xy"}, {"type": "xy"}],
            [None, {"type": "xy", "colspan": 2}, None],
        ],
        column_widths=[0.5, 0.25, 0.25],
        subplot_titles=("Float Track", "NIS over time", "ACF of Innovation", "Bias (x, y) over time"),
    )

    # --- Map: real surfacing points, colored by time ---
    fig.add_trace(
        go.Scattergeo(
            lat=surface_lats,
            lon=surface_lons,
            mode="markers",
            marker=dict(
                size=10,
                color=[t.value for t in surface_times],
                colorscale="Viridis",
                colorbar=dict(
                    title="Surfaced time",
                    x=-0.08,
                    tickvals=[t.value for t in surface_times],
                    ticktext=[t.strftime("%Y-%m-%d %H:%M") for t in surface_times],
                ),
            ),
            name="Real surfacing points",
            text=[t.isoformat() for t in surface_times],
        ),
        row=1, col=1,
    )

    # --- Map: estimated position + uncertainty ellipse (slider-driven) ---
    first_row = frames_df.iloc[0]
    ellipse_lats, ellipse_lons = _ellipse_latlon(first_row.x, first_row.y, first_row.P, n_sigma, config.target_lat, config.target_lon)

    fig.add_trace(
        go.Scattergeo(lat=[first_row.latitude], lon=[first_row.longitude], mode="markers",
                      marker=dict(size=12, color="red", symbol="x"), name="Estimated position"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergeo(lat=ellipse_lats, lon=ellipse_lons, mode="lines",
                      line=dict(color="red"), name=f"{n_sigma}-sigma uncertainty"),
        row=1, col=1,
    )

    position_trace_index = len(fig.data) - 2
    ellipse_trace_index = len(fig.data) - 1

    all_lats = surface_lats + state_history["latitude"].tolist()
    all_lons = surface_lons + state_history["longitude"].tolist()
    fig.update_geos(
        resolution=50,
        showland=True, landcolor="rgb(235, 230, 220)",
        showocean=True, oceancolor="rgb(220, 235, 245)",
        showcountries=True,
        lataxis_range=[min(all_lats) - 1, max(all_lats) + 1],
        lonaxis_range=[min(all_lons) - 1, max(all_lons) + 1],
    )

    # --- NIS over time (static) ---
    nis_time, nis = compute_nis_series(float_id)
    fig.add_trace(go.Scatter(x=nis_time, y=nis, mode="lines+markers", name="NIS"), row=1, col=2)

    # --- ACF of innovation (static, vs. lag) ---
    acf_x, acf_y = compute_innovation_acf(float_id)
    if len(acf_x) > 1:
        lags = np.arange(len(acf_x))
        fig.add_trace(go.Bar(x=lags - 0.2, y=acf_x, width=0.4, name="ACF innovation x"), row=1, col=3)
        fig.add_trace(go.Bar(x=lags + 0.2, y=acf_y, width=0.4, name="ACF innovation y"), row=1, col=3)
    else:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(opacity=0),
                                  showlegend=False, hoverinfo="skip"), row=1, col=3)
        acf_xaxis = fig.data[-1].xaxis or "x"
        acf_yaxis = fig.data[-1].yaxis or "y"
        fig.add_annotation(
            text="No innovation data available yet", showarrow=False,
            xref="x" + acf_xaxis[1:] + " domain", yref="y" + acf_yaxis[1:] + " domain", x=0.5, y=0.5,
        )

    # --- Bias (x, y) over time with uncertainty bands (static lines, slider-driven vertical line) ---
    bias_time, bias_x, bias_y, bias_x_lower, bias_x_upper, bias_y_lower, bias_y_upper = compute_bias_series(float_id, n_sigma=n_sigma)
    fig.add_trace(go.Scatter(x=bias_time, y=bias_x_upper, mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=bias_time, y=bias_x_lower, mode="lines", line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(31,119,180,0.2)", showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=bias_time, y=bias_x, mode="lines", name="bias_x", line=dict(color="rgb(31,119,180)")), row=2, col=2)

    fig.add_trace(go.Scatter(x=bias_time, y=bias_y_upper, mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=bias_time, y=bias_y_lower, mode="lines", line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(255,127,14,0.2)", showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=bias_time, y=bias_y, mode="lines", name="bias_y", line=dict(color="rgb(255,127,14)")), row=2, col=2)

    bias_xaxis = fig.data[-1].xaxis or "x"
    bias_yaxis = fig.data[-1].yaxis or "y"
    bias_xref = "x" + bias_xaxis[1:]
    bias_yref_domain = "y" + bias_yaxis[1:] + " domain"

    # --- Slider frames: move estimated position, ellipse, and the bias vertical line ---
    frames = []
    slider_steps = []
    for i, row in frames_df.iterrows():
        ellipse_lats, ellipse_lons = _ellipse_latlon(row.x, row.y, row.P, n_sigma, config.target_lat, config.target_lon)
        frame_name = row.time.isoformat()

        frames.append(go.Frame(
            name=frame_name,
            data=[
                go.Scattergeo(lat=[row.latitude], lon=[row.longitude]),
                go.Scattergeo(lat=ellipse_lats, lon=ellipse_lons),
            ],
            traces=[position_trace_index, ellipse_trace_index],
            layout=go.Layout(shapes=[dict(
                type="line", xref=bias_xref, yref=bias_yref_domain,
                x0=row.time, x1=row.time, y0=0, y1=1,
                line=dict(color="black", width=2, dash="dot"),
            )]),
        ))
        slider_steps.append(dict(
            method="animate",
            args=[[frame_name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=row.time.strftime("%Y-%m-%d %H:%M"),
        ))

    fig.frames = frames
    fig.update_layout(shapes=frames[0].layout.shapes if frames else [])

    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Time: "),
            steps=slider_steps,
        )],
        updatemenus=[dict(
            type="buttons", direction="left", x=0.05, y=-0.08,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(fromcurrent=True, frame=dict(duration=200, redraw=True), transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))]),
            ],
        )],
        title=f"Float {float_id} — EKF/MPC visualization",
        height=800,
    )

    return fig.to_html(full_html=True, include_plotlyjs="cdn")
