"""
XAUUSD M15 Data Processor for Mean Reversion (Strict Liquidity Sweep + Session Gates)
-------------------------------------------------------------------------------------
Loads raw Exness M15 CSV data, applies 1D Kalman filter smoothing,
computes core mean reversion indicators, strict 20-bar Swing Liquidity Sweeps,
and enforces hard Asian + Late NY session gating.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.data.kalman_filter import apply_kalman_filter


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    return atr.bfill().fillna(1.0)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = compute_atr(high, low, close, period=period)

    plus_di = 100.0 * pd.Series(plus_dm, index=close.index).ewm(alpha=1.0 / period, min_periods=period).mean() / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm, index=close.index).ewm(alpha=1.0 / period, min_periods=period).mean() / (atr + 1e-10)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean().fillna(20.0)

    return adx, plus_di.fillna(20.0), minus_di.fillna(20.0)


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    low_min = low.rolling(k_period, min_periods=1).min()
    high_max = high.rolling(k_period, min_periods=1).max()
    stoch_k = 100.0 * (close - low_min) / (high_max - low_min + 1e-10)
    stoch_d = stoch_k.rolling(d_period, min_periods=1).mean()
    return stoch_k.fillna(50.0), stoch_d.fillna(50.0)


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=1).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=1).mean()
    cci = (tp - sma) / (0.015 * mad + 1e-10)
    return cci.fillna(0.0)


def load_and_prepare_xauusd_m15(
    csv_path: str = "data/XAUUSD_M15_Exness.csv",
    start_date: str = "2018-01-01",
    rolling_norm_window: int = 200,
) -> pd.DataFrame:
    """
    Load XAUUSD M15 CSV, clean, generate Kalman and Mean Reversion features,
    detect Strict Swing Liquidity Sweeps, and apply Asian + Late NY Session Gating.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"[DataPrep] Loading {csv_path}...")
    df = pd.read_csv(csv_path, sep="\t", low_memory=False)
    df.columns = [c.strip().replace("<", "").replace(">", "").lower() for c in df.columns]

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Filter out early daily summary rows (keep strictly M15 records from start_date)
    df = df[df.index >= pd.Timestamp(start_date)].copy()
    print(f"[DataPrep] Valid M15 records from {start_date}: {len(df):,} bars (to {df.index[-1]})")

    # Rename & extract prices
    df.rename(columns={"tickvol": "volume"}, inplace=True)
    close = df["close"].astype(np.float64)
    open_p = df["open"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    volume = df["volume"].astype(np.float64)

    # Spread scaling
    sample_close = close.dropna().iloc[0]
    decimals = len(str(sample_close).split(".")[1]) if "." in str(sample_close) else 2
    point_val = 10.0 ** (-decimals)
    spread_raw = df["spread"].astype(np.float64) * point_val

    # 1. Noise Reduction: Apply 1D Causal Kalman Filter
    print("[DataPrep] Applying Causal Kalman Filter on Close price...")
    kf_df = apply_kalman_filter(close, q=0.005, r=0.08)
    kf_price = kf_df["kf_price"]
    kf_slope = kf_df["kf_slope"]

    # 2. Volatility & Trend Indicators
    atr = compute_atr(high, low, close, period=14)
    adx, di_plus, di_minus = compute_adx(high, low, close, period=14)
    di_diff = (di_plus - di_minus) / (di_plus + di_minus + 1e-10)

    # 3. Bollinger Bands (20, 2)
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std().fillna(1e-5)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_bandwidth = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # Distances to BB & Kalman in ATR units
    bb_upper_dist = (close - bb_upper) / (atr + 1e-10)
    bb_lower_dist = (close - bb_lower) / (atr + 1e-10)
    bb_mid_dist = (close - bb_mid) / (atr + 1e-10)
    kf_dist_atr = (close - kf_price) / (atr + 1e-10)

    # Z-scores
    zscore_kalman = (close - kf_price) / (bb_std + 1e-10)
    zscore_sma20 = (close - bb_mid) / (bb_std + 1e-10)

    # 4. Momentum & Oscillators
    rsi = compute_rsi(close, period=14)
    rsi_slope = rsi.diff(3).fillna(0.0)
    stoch_k, stoch_d = compute_stochastic(high, low, close, k_period=14, d_period=3)
    cci = compute_cci(high, low, close, period=14)

    # 5. Price Action, Wicks & Candle Anatomy
    candle_range = (high - low).replace(0.0, 1e-5)
    body = (close - open_p).abs()
    upper_wick = (high - np.maximum(open_p, close)) / candle_range
    lower_wick = (np.minimum(open_p, close) - low) / candle_range
    body_ratio = body / candle_range

    # 6. Swing High / Swing Low & Strict Liquidity Sweeps (20-bar lookback)
    prev_swing_high = high.shift(1).rolling(20, min_periods=5).max()
    prev_swing_low = low.shift(1).rolling(20, min_periods=5).min()

    # Bearish Sweep: Spiked above 20-bar swing high but rejected & closed back below
    sweep_high = ((high >= prev_swing_high) & (close < prev_swing_high)).astype(np.float64)
    # Bullish Sweep: Spiked below 20-bar swing low but rejected & closed back above
    sweep_low = ((low <= prev_swing_low) & (close > prev_swing_low)).astype(np.float64)

    # 7. Microstructure & Temporal
    vol_mean_200 = volume.rolling(rolling_norm_window, min_periods=10).mean()
    vol_std_200 = volume.rolling(rolling_norm_window, min_periods=10).std().replace(0.0, 1.0)
    volume_zscore = ((volume - vol_mean_200) / vol_std_200).clip(-5.0, 5.0).fillna(0.0)

    hour = df.index.hour
    minute = df.index.minute
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    minute_sin = np.sin(2.0 * np.pi * minute / 60.0)
    minute_cos = np.cos(2.0 * np.pi * minute / 60.0)

    session_asia = ((hour >= 0) & (hour < 8)).astype(np.float64)
    session_london = ((hour >= 8) & (hour < 16)).astype(np.float64)
    session_ny = ((hour >= 13) & (hour < 21)).astype(np.float64)
    session_late_ny = ((hour >= 18) & (hour < 23)).astype(np.float64)

    # Session gate: Asian session (0-8 UTC) or Late NY (18-23 UTC) where Gold mean-reverts reliably
    session_mean_rev_ok = ((session_asia == 1.0) | (session_late_ny == 1.0))

    # 8. Strict Hard Action Mask Gates
    # Long Gate: Strict Bullish Liquidity Sweep + Oversold (RSI < 40) + Extreme Z-score + ADX < 32 + Session Filter
    long_gate = (
        (sweep_low == 1.0)
        & (adx < 32.0)
        & (rsi < 40.0)
        & ((zscore_kalman < -1.4) | (zscore_sma20 < -1.5) | (bb_pct < 0.10))
        & session_mean_rev_ok
    ).astype(np.float64)

    # Short Gate: Strict Bearish Liquidity Sweep + Overbought (RSI > 60) + Extreme Z-score + ADX < 32 + Session Filter
    short_gate = (
        (sweep_high == 1.0)
        & (adx < 32.0)
        & (rsi > 60.0)
        & ((zscore_kalman > 1.4) | (zscore_sma20 > 1.5) | (bb_pct > 0.90))
        & session_mean_rev_ok
    ).astype(np.float64)

    print(f"[DataPrep] Total Long Gate Trigger Bars : {long_gate.sum():,.0f} ({long_gate.mean()*100:.2f}%)")
    print(f"[DataPrep] Total Short Gate Trigger Bars: {short_gate.sum():,.0f} ({short_gate.mean()*100:.2f}%)")

    # 9. Assembling Feature Dictionary
    features_dict = {
        # Raw prices & execution references (Excluded from Observation Vector)
        "open_raw": open_p,
        "high_raw": high,
        "low_raw": low,
        "close_raw": close,
        "atr_raw": atr,
        "spread_raw": spread_raw,
        "kf_price_raw": kf_price,
        "long_gate_raw": long_gate,
        "short_gate_raw": short_gate,
        # Reversion Signals & Distances
        "zscore_kalman": zscore_kalman.clip(-5.0, 5.0),
        "zscore_sma20": zscore_sma20.clip(-5.0, 5.0),
        "kf_dist_atr": kf_dist_atr.clip(-5.0, 5.0),
        "kf_slope": kf_slope,
        "bb_pct": bb_pct.clip(-0.5, 1.5),
        "bb_bandwidth": bb_bandwidth,
        "bb_upper_dist": bb_upper_dist.clip(-5.0, 5.0),
        "bb_lower_dist": bb_lower_dist.clip(-5.0, 5.0),
        "bb_mid_dist": bb_mid_dist.clip(-5.0, 5.0),
        # Momentum & Oscillators
        "rsi_14": (rsi - 50.0) / 25.0,
        "rsi_slope": rsi_slope / 10.0,
        "stoch_k": (stoch_k - 50.0) / 25.0,
        "stoch_d": (stoch_d - 50.0) / 25.0,
        "cci": (cci / 100.0).clip(-3.0, 3.0),
        # Volatility & Trend Filters
        "atr_norm": atr / (close + 1e-10) * 1000.0,
        "adx": (adx - 25.0) / 15.0,
        "di_diff": di_diff,
        # Sweeps & Price Action
        "sweep_high": sweep_high,
        "sweep_low": sweep_low,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "body_ratio": body_ratio,
        "volume_zscore": volume_zscore,
        # Temporal
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "minute_sin": minute_sin,
        "minute_cos": minute_cos,
        "session_asia": session_asia,
        "session_london": session_london,
        "session_ny": session_ny,
    }

    out_df = pd.DataFrame(features_dict, index=df.index)

    # 10. Causal Rolling Normalization for continuous feature series
    cols_to_roll = [
        "kf_slope", "bb_bandwidth", "atr_norm"
    ]
    for col in cols_to_roll:
        r_mean = out_df[col].rolling(rolling_norm_window, min_periods=20).mean()
        r_std = out_df[col].rolling(rolling_norm_window, min_periods=20).std().replace(0.0, 1.0)
        out_df[col] = ((out_df[col] - r_mean) / r_std).clip(-4.0, 4.0).fillna(0.0)

    # Drop early warm-up rows (first 200 bars)
    out_df = out_df.iloc[rolling_norm_window:].copy()
    out_df.dropna(inplace=True)

    print(f"[DataPrep] Completed. Total feature dimensions: {len(out_df.columns)} ({len(out_df):,} bars)")
    return out_df


if __name__ == "__main__":
    df = load_and_prepare_xauusd_m15()
    print("Head:\n", df.head(3))
