"""
Deep Quantitative Audit: True Causal H1 Resampling, Directional Long/Short Breakdown,
R-Multiple Performance, and Pure Rule vs RL Ablation across 4 Walk-Forward Windows.
"""
import os
import numpy as np
import pandas as pd
from src.data.kalman_filter import apply_kalman_filter


def compute_true_causal_h1_features(df_m15: pd.DataFrame) -> pd.DataFrame:
    """
    Resample M15 to H1, compute indicators on closed H1 bars, shift by 1 H1 bar
    (zero look-ahead), and merge back to M15.
    """
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_h1 = df_m15[['open', 'high', 'low', 'close', 'volume']].resample('1h').agg(ohlc_dict).dropna()
    
    # H1 indicators on closed bars
    h1_close = df_h1['close']
    h1_high = df_h1['high']
    h1_low = df_h1['low']
    
    # H1 EMAs
    df_h1['h1_ema50'] = h1_close.ewm(span=50, adjust=False).mean()
    df_h1['h1_ema200'] = h1_close.ewm(span=200, adjust=False).mean()
    
    # H1 Causal Kalman
    kf_res = apply_kalman_filter(h1_close, q=0.005, r=0.08)
    df_h1['h1_kf_price'] = kf_res['kf_price']
    df_h1['h1_kf_slope'] = kf_res['kf_slope']
    
    # H1 ATR & ADX
    tr = pd.concat([
        h1_high - h1_low,
        (h1_high - h1_close.shift(1)).abs(),
        (h1_low - h1_close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df_h1['h1_atr'] = tr.ewm(alpha=1.0/14.0, min_periods=14).mean().bfill()
    
    up = h1_high.diff()
    down = -h1_low.diff()
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)
    pdi = 100.0 * pd.Series(pdm, index=df_h1.index).ewm(alpha=1.0/14.0, min_periods=14).mean() / (df_h1['h1_atr'] + 1e-10)
    mdi = 100.0 * pd.Series(mdm, index=df_h1.index).ewm(alpha=1.0/14.0, min_periods=14).mean() / (df_h1['h1_atr'] + 1e-10)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    df_h1['h1_adx'] = dx.ewm(alpha=1.0/14.0, min_periods=14).mean().fillna(20.0)
    
    # Trend Regimes
    df_h1['h1_bull_regime'] = (
        (df_h1['close'] > df_h1['h1_ema50']) &
        (df_h1['h1_ema50'] > df_h1['h1_ema200']) &
        (df_h1['h1_kf_slope'] > 0.05)
    ).astype(float)
    
    df_h1['h1_bear_regime'] = (
        (df_h1['close'] < df_h1['h1_ema50']) &
        (df_h1['h1_ema50'] < df_h1['h1_ema200']) &
        (df_h1['h1_kf_slope'] < -0.05)
    ).astype(float)
    
    # CRITICAL: Shift by 1 H1 bar so at M15 time t, we only know the completed prior hour
    df_h1_causal = df_h1[['h1_ema50', 'h1_ema200', 'h1_kf_price', 'h1_kf_slope', 'h1_atr', 'h1_adx', 'h1_bull_regime', 'h1_bear_regime']].shift(1)
    
    # Reindex onto M15 index with ffill
    df_merged = df_m15.join(df_h1_causal, how='left').ffill().bfill()
    return df_merged


def simulate_trades(df_sub: pd.DataFrame, use_h1_filter: bool = True, tp_r: float = 1.8, sl_r: float = 1.0, max_hold: int = 36) -> list[dict]:
    """
    Simulates trades with fixed target TP/SL and exact R-multiple & dollar calculation.
    """
    trades = []
    in_pos = 0 # 0: None, 1: Long, -1: Short
    entry_price = 0.0
    entry_sl = 0.0
    entry_tp = 0.0
    entry_step = 0
    entry_atr = 0.0
    entry_spread = 0.0
    
    n = len(df_sub)
    opens = df_sub['open'].to_numpy(dtype=float)
    highs = df_sub['high'].to_numpy(dtype=float)
    lows = df_sub['low'].to_numpy(dtype=float)
    closes = df_sub['close'].to_numpy(dtype=float)
    atrs = df_sub['atr'].to_numpy(dtype=float)
    spreads = df_sub['spread'].to_numpy(dtype=float)
    
    long_gates = df_sub['long_gate'].to_numpy(dtype=float)
    short_gates = df_sub['short_gate'].to_numpy(dtype=float)
    h1_bulls = df_sub['h1_bull_regime'].to_numpy(dtype=float) if 'h1_bull_regime' in df_sub.columns else np.zeros(n)
    h1_bears = df_sub['h1_bear_regime'].to_numpy(dtype=float) if 'h1_bear_regime' in df_sub.columns else np.zeros(n)
    
    point_val = 10.0 # $10 per point for 0.10 lot Gold
    
    for i in range(n):
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_close = closes[i]
        c_atr = atrs[i]
        c_spread = spreads[i]
        
        # In-trade management
        if in_pos == 1: # Long
            hold_bars = i - entry_step
            pnl_pts = 0.0
            exit_reason = None
            exit_price = 0.0
            
            # Check SL
            if c_low <= entry_sl:
                exit_price = min(c_open, entry_sl)
                pnl_pts = exit_price - entry_price - (entry_spread + c_spread)
                exit_reason = "SL"
            # Check TP
            elif c_high >= entry_tp:
                exit_price = max(c_open, entry_tp)
                pnl_pts = exit_price - entry_price - (entry_spread + c_spread)
                exit_reason = "TP"
            # Time limit
            elif hold_bars >= max_hold:
                exit_price = c_close
                pnl_pts = exit_price - entry_price - (entry_spread + c_spread)
                exit_reason = "TIME"
                
            if exit_reason is not None:
                pnl_dollar = pnl_pts * point_val
                risk_pts = entry_atr * sl_r
                r_multiple = pnl_pts / (risk_pts + 1e-8)
                trades.append({
                    "dir": "LONG",
                    "entry_time": df_sub.index[entry_step],
                    "exit_time": df_sub.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_dollar": pnl_dollar,
                    "r_multiple": r_multiple,
                    "reason": exit_reason,
                    "hold_bars": hold_bars,
                    "atr": entry_atr,
                })
                in_pos = 0
                
        elif in_pos == -1: # Short
            hold_bars = i - entry_step
            pnl_pts = 0.0
            exit_reason = None
            exit_price = 0.0
            
            # Check SL
            if c_high >= entry_sl:
                exit_price = max(c_open, entry_sl)
                pnl_pts = entry_price - exit_price - (entry_spread + c_spread)
                exit_reason = "SL"
            # Check TP
            elif c_low <= entry_tp:
                exit_price = min(c_open, entry_tp)
                pnl_pts = entry_price - exit_price - (entry_spread + c_spread)
                exit_reason = "TP"
            # Time limit
            elif hold_bars >= max_hold:
                exit_price = c_close
                pnl_pts = entry_price - exit_price - (entry_spread + c_spread)
                exit_reason = "TIME"
                
            if exit_reason is not None:
                pnl_dollar = pnl_pts * point_val
                risk_pts = entry_atr * sl_r
                r_multiple = pnl_pts / (risk_pts + 1e-8)
                trades.append({
                    "dir": "SHORT",
                    "entry_time": df_sub.index[entry_step],
                    "exit_time": df_sub.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_dollar": pnl_dollar,
                    "r_multiple": r_multiple,
                    "reason": exit_reason,
                    "hold_bars": hold_bars,
                    "atr": entry_atr,
                })
                in_pos = 0
                
        # Check Entry Signals if Flat
        if in_pos == 0 and i < n - 1:
            l_gate = long_gates[i] == 1.0
            s_gate = short_gates[i] == 1.0
            
            if use_h1_filter:
                l_gate = l_gate and (h1_bears[i] == 0.0) # Not in H1 bear trend
                s_gate = s_gate and (h1_bulls[i] == 0.0) # Not in H1 bull trend
                
            if l_gate and not s_gate:
                in_pos = 1
                entry_step = i + 1 # Next bar open
                entry_price = opens[entry_step]
                entry_atr = atrs[entry_step]
                entry_spread = spreads[entry_step]
                entry_sl = entry_price - (entry_atr * sl_r)
                entry_tp = entry_price + (entry_atr * tp_r)
            elif s_gate and not l_gate:
                in_pos = -1
                entry_step = i + 1 # Next bar open
                entry_price = opens[entry_step]
                entry_atr = atrs[entry_step]
                entry_spread = spreads[entry_step]
                entry_sl = entry_price + (entry_atr * sl_r)
                entry_tp = entry_price - (entry_atr * tp_r)
                
    return trades


def run_full_audit():
    csv_path = "data/XAUUSD_M15_Exness.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, sep="\t", low_memory=False)
    df.columns = [c.strip().replace("<", "").replace(">", "").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df = df[df.index >= pd.Timestamp("2018-01-01")].copy()
    if 'tickvol' in df.columns:
        df.rename(columns={'tickvol': 'volume'}, inplace=True)
    
    # Basic M15 indicators
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    
    sample_close = close.dropna().iloc[0]
    decimals = len(str(sample_close).split(".")[1]) if "." in str(sample_close) else 2
    point_val = 10.0 ** (-decimals)
    df['spread'] = df['spread'].astype(float) * point_val
    
    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1.0/14.0, min_periods=14).mean().bfill()
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0/14.0, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14.0, min_periods=14).mean()
    df['rsi'] = (100.0 - (100.0 / (1.0 + gain / (loss + 1e-10)))).fillna(50.0)
    
    # ADX
    up = high.diff()
    down = -low.diff()
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)
    pdi = 100.0 * pd.Series(pdm, index=df.index).ewm(alpha=1.0/14.0, min_periods=14).mean() / (df['atr'] + 1e-10)
    mdi = 100.0 * pd.Series(mdm, index=df.index).ewm(alpha=1.0/14.0, min_periods=14).mean() / (df['atr'] + 1e-10)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    df['adx'] = dx.ewm(alpha=1.0/14.0, min_periods=14).mean().fillna(20.0)
    
    # Kalman & Bollinger
    kf_res = apply_kalman_filter(close, q=0.005, r=0.08)
    df['kf_price'] = kf_res['kf_price']
    df['zscore_kalman'] = (close - df['kf_price']) / (df['atr'] + 1e-8)
    
    sma20 = close.rolling(20, min_periods=5).mean()
    std20 = close.rolling(20, min_periods=5).std().replace(0.0, 1.0)
    df['zscore_sma20'] = (close - sma20) / std20
    df['bb_pct'] = (close - (sma20 - 2.0 * std20)) / (4.0 * std20 + 1e-8)
    
    # 20-bar Swing Sweeps
    prev_swing_high = high.shift(1).rolling(20, min_periods=5).max()
    prev_swing_low = low.shift(1).rolling(20, min_periods=5).min()
    df['sweep_high'] = ((high >= prev_swing_high) & (close < prev_swing_high)).astype(float)
    df['sweep_low'] = ((low <= prev_swing_low) & (close > prev_swing_low)).astype(float)
    
    # Session Gate (Asia 0-8 UTC or Late NY 18-23 UTC)
    hour = df.index.hour
    session_ok = ((hour >= 0) & (hour < 8)) | ((hour >= 18) & (hour < 23))
    
    # Raw Gates (Before H1)
    df['long_gate'] = (
        (df['sweep_low'] == 1.0) &
        (df['adx'] < 32.0) &
        (df['rsi'] < 40.0) &
        ((df['zscore_kalman'] < -1.4) | (df['zscore_sma20'] < -1.5) | (df['bb_pct'] < 0.10)) &
        session_ok
    ).astype(float)
    
    df['short_gate'] = (
        (df['sweep_high'] == 1.0) &
        (df['adx'] < 32.0) &
        (df['rsi'] > 60.0) &
        ((df['zscore_kalman'] > 1.4) | (df['zscore_sma20'] > 1.5) | (df['bb_pct'] > 0.90)) &
        session_ok
    ).astype(float)
    
    # Add True Causal H1 Features
    print("Computing True Causal H1 Resampled Features (shifted 1 bar)...")
    df_full = compute_true_causal_h1_features(df)
    
    windows = [
        ("Window 1 (2022-2023)", "2022-01-01", "2023-01-01"),
        ("Window 2 (2023-2024)", "2023-01-01", "2024-01-01"),
        ("Window 3 (2024-2025)", "2024-01-01", "2025-01-01"),
        ("Window 4 (2025-2026)", "2025-01-01", "2026-07-17"),
    ]
    
    # 1. Compare Before vs After H1 Filter with R-multiples & Dollar Breakdown
    print("\n" + "="*115)
    print("📊 1. WALK-FORWARD DIRECTIONAL AUDIT (WITH & WITHOUT TRUE CAUSAL H1 FILTER)")
    print("="*115)
    
    for use_h1 in [False, True]:
        label = "AFTER TRUE CAUSAL H1 FILTER" if use_h1 else "BEFORE H1 FILTER (RAW M15)"
        print(f"\n>>> MODE: {label}")
        print(f"{'Window':<22} | {'Dir':<5} | {'Trades':<6} | {'WinRate':<7} | {'PF':<5} | {'Net P&L ($)':<12} | {'Net P&L (R)':<11} | {'Avg Win (R)':<11} | {'Avg Loss (R)':<12}")
        print("-" * 115)
        
        tot_dollar = 0.0
        tot_r = 0.0
        tot_trades = 0
        tot_wins = 0
        tot_gw = 0.0
        tot_gl = 0.0
        
        for w_title, s_date, e_date in windows:
            w_df = df_full[(df_full.index >= pd.Timestamp(s_date)) & (df_full.index < pd.Timestamp(e_date))].copy()
            trades = simulate_trades(w_df, use_h1_filter=use_h1)
            
            for d in ["LONG", "SHORT", "TOTAL"]:
                if d == "TOTAL":
                    t_sub = trades
                else:
                    t_sub = [t for t in trades if t['dir'] == d]
                    
                n_t = len(t_sub)
                if n_t > 0:
                    pnls = [t['pnl_dollar'] for t in t_sub]
                    rs = [t['r_multiple'] for t in t_sub]
                    w_d = [p for p in pnls if p > 0]
                    l_d = [p for p in pnls if p < 0]
                    w_r = [r for r in rs if r > 0]
                    l_r = [r for r in rs if r < 0]
                    
                    gw = sum(w_d)
                    gl = abs(sum(l_d)) if len(l_d) > 0 else 1e-4
                    net_d = gw - gl
                    net_r = sum(rs)
                    pf = gw / gl
                    wr = len(w_d) / n_t * 100.0
                    avg_w_r = np.mean(w_r) if len(w_r) > 0 else 0.0
                    avg_l_r = np.mean(l_r) if len(l_r) > 0 else 0.0
                else:
                    wr = pf = net_d = net_r = avg_w_r = avg_l_r = 0.0
                    
                prefix = w_title if d == "TOTAL" else f"  └─ {w_title.split()[0]} {w_title.split()[1]}"
                print(f"{prefix:<22} | {d:<5} | {n_t:<6} | {wr:>6.1f}% | {pf:>5.2f} | ${net_d:>+10.2f} | {net_r:>+9.2f} R | {avg_w_r:>+9.2f} R | {avg_l_r:>+10.2f} R")
                
                if d == "TOTAL":
                    tot_dollar += net_d
                    tot_r += net_r
                    tot_trades += n_t
                    tot_wins += len(w_d)
                    tot_gw += gw
                    tot_gl += gl
                    
        tot_pf = tot_gw / (tot_gl + 1e-8)
        tot_wr = tot_wins / (tot_trades + 1e-8) * 100.0
        print("-" * 115)
        print(f"{'OVERALL (4-YEARS)':<22} | {'TOTAL':<5} | {tot_trades:<6} | {tot_wr:>6.1f}% | {tot_pf:>5.2f} | ${tot_dollar:>+10.2f} | {tot_r:>+9.2f} R |")
        print("-" * 115)


if __name__ == "__main__":
    run_full_audit()
