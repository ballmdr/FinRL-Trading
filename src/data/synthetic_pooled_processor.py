"""
Multi-Asset Synthetic Pooled Data Processor (FinRL-Trading)
------------------------------------------------------------
Loads and processes 9-symbol synthetic and real continuity datasets:
- synthetic_only.parquet (668,160 rows, 1,152 episodes)
- real_only.parquet (668,160 rows, 1,152 episodes)
- mixed_50_50.parquet (668,160 rows, 1,152 episodes)

Extracts 45 precomputed features (26 technical/SMC + 10 global macro + 9 one-hot)
and formats raw execution columns for multi-episode DRL trading environments.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd


# 26 Core Feature Columns
CORE_FEATURES = [
    "returns", "log_returns", "returns_5ma", "returns_20ma",
    "body_ratio", "upper_wick", "lower_wick", "close_position",
    "atr_ratio", "volatility_20", "bb_position",
    "log_volume", "volume_ma_ratio", "volume_trend",
    "fvg_signal", "bullish_ob", "bearish_ob",
    "market_structure", "liquidity_proximity",
    "rsi_14", "macd_histogram", "adx_14", "trend_direction",
    "hour_sin", "hour_cos", "day_of_week"
]

# 10 Global Macro & Multi-Asset Context Columns
GLOBAL_CONTEXT = [
    "usd_strength_return", "usd_strength_vol_144",
    "fx_dispersion", "fx_mean_abs_corr_144",
    "xau_return", "xau_vol_144",
    "btc_return", "btc_vol_144",
    "fx_active_fraction", "xau_available"
]

# 9 One-Hot Symbol Columns
ONE_HOT_FIELDS = [
    "onehot_EURUSD", "onehot_GBPUSD", "onehot_USDJPY",
    "onehot_USDCHF", "onehot_AUDUSD", "onehot_USDCAD",
    "onehot_NZDUSD", "onehot_XAUUSD", "onehot_BTCUSD"
]

ALL_FEATURE_COLS = CORE_FEATURES + GLOBAL_CONTEXT + ONE_HOT_FIELDS

RAW_EXECUTION_COLS = [
    "target_open", "target_high", "target_low", "target_close",
    "target_volume", "target_spread"
]


def load_synthetic_pooled_data(
    data_dir: str = "../data/v3_src/runs/20260813-v3-2-pooled-continuity-r1",
    load_real: bool = True,
    symbol_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load synthetic_only.parquet (for training) and real_only.parquet (for testing).
    Optionally filter by symbol (e.g. symbol_filter='XAUUSD').
    """
    syn_path = os.path.join(data_dir, "synthetic_only.parquet")
    real_path = os.path.join(data_dir, "real_only.parquet")

    if not os.path.exists(syn_path):
        raise FileNotFoundError(f"Synthetic dataset not found at {syn_path}")

    print(f"[SyntheticData] Loading training dataset from {syn_path}...")
    df_syn = pd.read_parquet(syn_path)

    df_real = None
    if load_real and os.path.exists(real_path):
        print(f"[SyntheticData] Loading real evaluation dataset from {real_path}...")
        df_real = pd.read_parquet(real_path)

    if symbol_filter is not None:
        print(f"[SyntheticData] Filtering for symbol: {symbol_filter}")
        df_syn = df_syn[df_syn["symbol"] == symbol_filter].copy().reset_index(drop=True)
        if df_real is not None:
            df_real = df_real[df_real["symbol"] == symbol_filter].copy().reset_index(drop=True)

    # Validate feature presence
    missing_cols = [c for c in ALL_FEATURE_COLS if c not in df_syn.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in dataset: {missing_cols}")

    print(f"[SyntheticData] Loaded Synthetic: {len(df_syn):,} rows ({df_syn['episode_id'].nunique()} episodes)")
    if df_real is not None:
        print(f"[SyntheticData] Loaded Real Data : {len(df_real):,} rows ({df_real['episode_id'].nunique()} episodes)")
    print(f"[SyntheticData] Feature Vector Dimension: {len(ALL_FEATURE_COLS)} features")

    return df_syn, df_real


def extract_episodes_dict(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """
    Pre-indexes the dataframe by episode_id into high-speed numpy arrays for fast Gym stepping.
    Returns: {
        episode_id: {
            'symbol': str,
            'features': np.ndarray (shape: [580, 45]),
            'open': np.ndarray,
            'high': np.ndarray,
            'low': np.ndarray,
            'close': np.ndarray,
            'spread': np.ndarray,
            'volume': np.ndarray,
            'timestamps': list/array
        }
    }
    """
    episodes = {}
    grouped = df.groupby("episode_id", sort=False)

    for ep_id, group in grouped:
        sym = group["symbol"].iloc[0]
        # Robust clipping to [-10.0, 10.0] to prevent gradient explosion from synthetic stress sentinels (e.g. 1e9)
        feats = np.clip(group[ALL_FEATURE_COLS].values.astype(np.float32), -10.0, 10.0)
        feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)

        # Robust forward/backward fill on price columns to handle weekend/holiday gaps
        c_open = group["target_open"].ffill().bfill().values.astype(np.float64)
        c_high = group["target_high"].ffill().bfill().values.astype(np.float64)
        c_low = group["target_low"].ffill().bfill().values.astype(np.float64)
        c_close = group["target_close"].ffill().bfill().values.astype(np.float64)
        c_spread = group["target_spread"].ffill().bfill().fillna(0.0002).values.astype(np.float64)
        c_volume = group["target_volume"].ffill().bfill().fillna(1.0).values.astype(np.float64)

        episodes[ep_id] = {
            "symbol": sym,
            "features": feats,
            "open": c_open,
            "high": c_high,
            "low": c_low,
            "close": c_close,
            "spread": c_spread,
            "volume": c_volume,
            "timestamps": group["timestamp_utc"].values,
        }

    return episodes


if __name__ == "__main__":
    df_s, df_r = load_synthetic_pooled_data()
    print("Synthetic shape:", df_s.shape)
    episodes = extract_episodes_dict(df_s)
    first_ep = list(episodes.keys())[0]
    print(f"Sample episode {first_ep}: shape = {episodes[first_ep]['features'].shape}")
