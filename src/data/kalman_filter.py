"""
Kalman Filter Module for Financial Price Time Series
-----------------------------------------------------
Provides a causal 1D Local-Level Kalman Filter to eliminate market noise
and extract a robust dynamic mean without look-ahead bias.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def apply_kalman_filter(
    series: pd.Series,
    q: float = 0.001,
    r: float = 0.1,
    initial_variance: float | None = None,
) -> pd.DataFrame:
    """
    Apply a causal 1D Local-Level Kalman Filter to a price series.

    State Model:
        x_t = x_{t-1} + w_t,  w_t ~ N(0, q)   (Hidden True Price)
        y_t = x_t + v_t,      v_t ~ N(0, r)   (Observed Noisy Price)

    Parameters:
    -----------
    series : pd.Series
        Close price series.
    q : float
        Process variance (speed of state update / flexibility).
    r : float
        Measurement noise variance (smoothing strength).
    initial_variance : float | None
        Initial state uncertainty estimate.

    Returns:
    --------
    pd.DataFrame with:
        - kf_price: Filtered dynamic mean price
        - kf_innovation: One-step-ahead prediction error (y_t - x_{t|t-1})
        - kf_slope: Instantaneous slope of filtered price
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    obs = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    n = len(obs)
    filtered = np.full(n, np.nan, dtype=np.float64)
    innovation = np.full(n, np.nan, dtype=np.float64)

    state = np.nan
    variance = r if initial_variance is None else float(initial_variance)

    for i in range(n):
        y = obs[i]
        if not np.isfinite(y):
            continue

        if not np.isfinite(state):
            state = y
            innovation[i] = 0.0
            filtered[i] = state
            continue

        # Predict step
        pred_state = state
        pred_var = variance + q

        # Update step
        innov = y - pred_state
        gain = pred_var / (pred_var + r)
        state = pred_state + gain * innov
        variance = (1.0 - gain) * pred_var

        filtered[i] = state
        innovation[i] = innov

    # Causal slope
    slope = pd.Series(filtered, index=series.index).diff().fillna(0.0)

    return pd.DataFrame(
        {
            "kf_price": filtered,
            "kf_innovation": innovation,
            "kf_slope": slope.to_numpy(dtype=np.float64),
        },
        index=series.index,
    )
