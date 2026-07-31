import numpy as np
import pandas as pd

from data_handler import read_state, read_surfacing_and_action


def compute_bias_series(float_id: int, n_sigma: float = 2.0) -> tuple[np.ndarray, ...]:
    """Bias (bx, by) time series with +/- n_sigma uncertainty bands from P's diagonal.

    Returns (time, bias_x, bias_y, bias_x_lower, bias_x_upper, bias_y_lower, bias_y_upper).
    """
    state_history = read_state(float_id)
    if state_history.empty:
        raise FileNotFoundError(f"No estimated state history found for float ID: {float_id}")

    state_history = state_history.sort_values("time").reset_index(drop=True)

    time = state_history["time"].to_numpy()
    bias_x = state_history["bx"].to_numpy(dtype=float)
    bias_y = state_history["by"].to_numpy(dtype=float)

    # P is ordered [x, y, bx, by], so bias variance sits on the diagonal at (2, 2) and (3, 3).
    std_x = np.array([np.sqrt(np.asarray(P)[2, 2]) for P in state_history["P"]])
    std_y = np.array([np.sqrt(np.asarray(P)[3, 3]) for P in state_history["P"]])

    bias_x_lower = bias_x - n_sigma * std_x
    bias_x_upper = bias_x + n_sigma * std_x
    bias_y_lower = bias_y - n_sigma * std_y
    bias_y_upper = bias_y + n_sigma * std_y

    return time, bias_x, bias_y, bias_x_lower, bias_x_upper, bias_y_lower, bias_y_upper


def compute_nis_series(float_id: int) -> tuple[np.ndarray, np.ndarray]:
    """NIS time series from the surfacing/action log. Returns (time, nis)."""
    entries = read_surfacing_and_action(float_id)
    entries = sorted(entries, key=lambda entry: entry["surfaced_timestamp"])

    time = np.array([pd.Timestamp(entry["surfaced_timestamp"]) for entry in entries])
    nis = np.array([entry["nis"] for entry in entries], dtype=float)

    return time, nis


def compute_innovation_series(float_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw innovation (x, y) time series from the surfacing/action log.

    Older log entries predate the "innovation" field, so entries missing it are skipped.
    Returns (time, innovation_x, innovation_y).
    """
    entries = read_surfacing_and_action(float_id)
    entries = [entry for entry in entries if entry.get("innovation") is not None]
    entries = sorted(entries, key=lambda entry: entry["surfaced_timestamp"])

    time = np.array([pd.Timestamp(entry["surfaced_timestamp"]) for entry in entries])
    innovation_x = np.array([entry["innovation"][0] for entry in entries], dtype=float)
    innovation_y = np.array([entry["innovation"][1] for entry in entries], dtype=float)

    return time, innovation_x, innovation_y


def _sample_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    "Mean-centered sample autocorrelation of `series` for lags 0..min(max_lag, n-1)."
    n = len(series)
    if n < 2:
        return np.array([1.0])

    max_lag = min(max_lag, n - 1)
    centered = series - series.mean()
    denom = np.dot(centered, centered)
    if denom == 0:
        return np.full(max_lag + 1, np.nan)

    return np.array([
        np.dot(centered[:n - lag], centered[lag:]) / denom if lag > 0 else 1.0
        for lag in range(max_lag + 1)
    ])


def compute_innovation_acf(float_id: int, max_lag: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation of the innovation, separated by x and y. Returns (acf_x, acf_y)."""
    _, innovation_x, innovation_y = compute_innovation_series(float_id)
    return _sample_acf(innovation_x, max_lag), _sample_acf(innovation_y, max_lag)
