"""
H1 Trend-Following Benchmark Audit Engine (FinRL-Trading)
---------------------------------------------------------
Evaluates pure systematic H1 trend-following rules across 4 Walk-Forward Windows:
1. Donchian 20 Breakout (Fixed R:R: SL 1.0 ATR, TP 2.0R, 2.5R, 3.0R)
2. Donchian 20 Breakout + EMA(50/200) Trend Filter
3. Classic Donchian Channel Exit (Entry 20-bar break, Exit 10-bar reverse break, Catastrophic SL 1.5 ATR)

Calculates exact R-multiples, directional (Long/Short) splits, and cost friction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def load_h1_gold_data(csv_path: str = "data/XAUUSD_M15_Exness.csv") -> pd.DataFrame:
    """Loads M15 and resamples to pure H1 bars with exact causal aggregation."""
    df = pd.read_csv(csv_path, sep="\t", low_memory=False)
    df.columns = [c.strip().replace("<", "").replace(">", "").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df = df[df.index >= pd.Timestamp("2018-01-01")].copy()
    if 'tickvol' in df.columns:
        df.rename(columns={'tickvol': 'volume'}, inplace=True)
        
    sample_close = df['close'].dropna().iloc[0]
    decimals = len(str(sample_close).split(".")[1]) if "." in str(sample_close) else 2
    point_val = 10.0 ** (-decimals)
    df['spread'] = df['spread'].astype(float) * point_val

    # Resample to H1
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'spread': 'mean'
    }
    df_h1 = df.resample('1h').agg(ohlc_dict).dropna().copy()
    
    # H1 Indicators
    high = df_h1['high'].astype(float)
    low = df_h1['low'].astype(float)
    close = df_h1['close'].astype(float)
    
    # ATR(14) on H1
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df_h1['atr'] = tr.ewm(alpha=1.0/14.0, min_periods=14).mean().bfill()
    
    # EMAs
    df_h1['ema50'] = close.ewm(span=50, adjust=False).mean()
    df_h1['ema200'] = close.ewm(span=200, adjust=False).mean()
    
    # 20-bar Donchian High/Low on completed prior bars
    df_h1['donchian_high_20'] = high.shift(1).rolling(20, min_periods=5).max()
    df_h1['donchian_low_20'] = low.shift(1).rolling(20, min_periods=5).min()
    
    # 10-bar Donchian High/Low for Chandelier/Channel exits
    df_h1['donchian_high_10'] = high.shift(1).rolling(10, min_periods=3).max()
    df_h1['donchian_low_10'] = low.shift(1).rolling(10, min_periods=3).min()
    
    return df_h1


def simulate_h1_rule(
    df_sub: pd.DataFrame,
    strategy_type: str = "fixed_rr",
    tp_mult: float = 2.5,
    sl_mult: float = 1.0,
    use_ema_filter: bool = False,
    max_hold_bars: int = 120,
) -> list[dict]:
    """
    Simulates H1 Trend-Following rules with exact dollar and R-multiple tracking.
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
    
    don_h20 = df_sub['donchian_high_20'].to_numpy(dtype=float)
    don_l20 = df_sub['donchian_low_20'].to_numpy(dtype=float)
    don_h10 = df_sub['donchian_high_10'].to_numpy(dtype=float)
    don_l10 = df_sub['donchian_low_10'].to_numpy(dtype=float)
    ema50 = df_sub['ema50'].to_numpy(dtype=float)
    ema200 = df_sub['ema200'].to_numpy(dtype=float)
    
    point_val = 10.0 # $10 per point for 0.10 lot Gold
    
    for i in range(n):
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_close = closes[i]
        c_atr = atrs[i]
        c_spread = spreads[i]
        
        # 1. Manage Active Positions
        if in_pos == 1: # Long
            hold_bars = i - entry_step
            exit_reason = None
            exit_price = 0.0
            pnl_pts = 0.0
            
            # Check Hard SL
            if c_low <= entry_sl:
                exit_price = min(c_open, entry_sl)
                pnl_pts = exit_price - entry_price - (entry_spread + c_spread)
                exit_reason = "SL"
            # Check Fixed TP (if fixed_rr)
            elif strategy_type == "fixed_rr" and c_high >= entry_tp:
                exit_price = max(c_open, entry_tp)
                pnl_pts = exit_price - entry_price - (entry_spread + c_spread)
                exit_reason = "TP"
            # Check Channel Exit (if channel_exit: close drops below 10-bar low)
            elif strategy_type == "channel_exit" and c_low <= don_l10[i]:
                exit_price = min(c_open, don_l10[i])
                pnl_pts = exit_price - entry_price - (entry_spread + c_spread)
                exit_reason = "CH_EXIT"
            # Time limit
            elif hold_bars >= max_hold_bars:
                exit_price = c_close
                pnl_pts = exit_price - entry_price - (entry_spread + c_spread)
                exit_reason = "TIME"
                
            if exit_reason is not None:
                pnl_dollar = pnl_pts * point_val
                risk_pts = entry_atr * sl_mult
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
            exit_reason = None
            exit_price = 0.0
            pnl_pts = 0.0
            
            # Check Hard SL
            if c_high >= entry_sl:
                exit_price = max(c_open, entry_sl)
                pnl_pts = entry_price - exit_price - (entry_spread + c_spread)
                exit_reason = "SL"
            # Check Fixed TP (if fixed_rr)
            elif strategy_type == "fixed_rr" and c_low <= entry_tp:
                exit_price = min(c_open, entry_tp)
                pnl_pts = entry_price - exit_price - (entry_spread + c_spread)
                exit_reason = "TP"
            # Check Channel Exit (if channel_exit: close rises above 10-bar high)
            elif strategy_type == "channel_exit" and c_high >= don_h10[i]:
                exit_price = max(c_open, don_h10[i])
                pnl_pts = entry_price - exit_price - (entry_spread + c_spread)
                exit_reason = "CH_EXIT"
            # Time limit
            elif hold_bars >= max_hold_bars:
                exit_price = c_close
                pnl_pts = entry_price - exit_price - (entry_spread + c_spread)
                exit_reason = "TIME"
                
            if exit_reason is not None:
                pnl_dollar = pnl_pts * point_val
                risk_pts = entry_atr * sl_mult
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

        # 2. Check Entry Signals if Flat
        if in_pos == 0 and i < n - 1:
            # Long breakout: close > 20-bar Donchian high
            long_signal = closes[i] > don_h20[i]
            # Short breakout: close < 20-bar Donchian low
            short_signal = closes[i] < don_l20[i]
            
            if use_ema_filter:
                long_signal = long_signal and (closes[i] > ema50[i]) and (ema50[i] > ema200[i])
                short_signal = short_signal and (closes[i] < ema50[i]) and (ema50[i] < ema200[i])
                
            if long_signal and not short_signal:
                in_pos = 1
                entry_step = i + 1 # Enter on next H1 bar open
                entry_price = opens[entry_step]
                entry_atr = atrs[entry_step]
                entry_spread = spreads[entry_step]
                entry_sl = entry_price - (entry_atr * sl_mult)
                entry_tp = entry_price + (entry_atr * tp_mult)
            elif short_signal and not long_signal:
                in_pos = -1
                entry_step = i + 1 # Enter on next H1 bar open
                entry_price = opens[entry_step]
                entry_atr = atrs[entry_step]
                entry_spread = spreads[entry_step]
                entry_sl = entry_price + (entry_atr * sl_mult)
                entry_tp = entry_price - (entry_atr * tp_mult)
                
    return trades


def print_walkforward_table(df_h1: pd.DataFrame, strat_name: str, **kwargs):
    windows = [
        ("Window 1 (2022-2023)", "2022-01-01", "2023-01-01"),
        ("Window 2 (2023-2024)", "2023-01-01", "2024-01-01"),
        ("Window 3 (2024-2025)", "2024-01-01", "2025-01-01"),
        ("Window 4 (2025-2026)", "2025-01-01", "2026-07-17"),
    ]
    
    print("\n" + "="*125)
    print(f"📈 STRATEGY: {strat_name}")
    print("="*125)
    print(f"{'Window':<22} | {'Dir':<5} | {'Trades':<6} | {'WinRate':<7} | {'PF':<5} | {'Net P&L ($)':<12} | {'Net P&L (R)':<11} | {'EV (R/trade)':<12} | {'Avg Win (R)':<11} | {'Avg Loss (R)':<12}")
    print("-" * 125)
    
    tot_trades = 0
    tot_wins = 0
    tot_dollar = 0.0
    tot_r = 0.0
    tot_gw = 0.0
    tot_gl = 0.0
    win_flags = []
    
    for w_title, s_date, e_date in windows:
        w_df = df_h1[(df_h1.index >= pd.Timestamp(s_date)) & (df_h1.index < pd.Timestamp(e_date))].copy()
        trades = simulate_h1_rule(w_df, **kwargs)
        
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
                ev_r = net_r / n_t
                pf = gw / gl
                wr = len(w_d) / n_t * 100.0
                avg_w_r = np.mean(w_r) if len(w_r) > 0 else 0.0
                avg_l_r = np.mean(l_r) if len(l_r) > 0 else 0.0
            else:
                wr = pf = net_d = net_r = ev_r = avg_w_r = avg_l_r = 0.0
                
            prefix = w_title if d == "TOTAL" else f"  └─ {w_title.split()[0]} {w_title.split()[1]}"
            status_icon = "🟢" if (d == "TOTAL" and net_r > 0) else ("🔴" if d == "TOTAL" else "")
            print(f"{prefix:<22} | {d:<5} | {n_t:<6} | {wr:>6.1f}% | {pf:>5.2f} | ${net_d:>+10.2f} | {net_r:>+9.2f} R {status_icon} | {ev_r:>+10.2f} R | {avg_w_r:>+9.2f} R | {avg_l_r:>+10.2f} R")
            
            if d == "TOTAL":
                tot_trades += n_t
                tot_wins += len(w_d)
                tot_dollar += net_d
                tot_r += net_r
                tot_gw += gw
                tot_gl += gl
                win_flags.append(net_r > 0)
                
    tot_pf = tot_gw / (tot_gl + 1e-8)
    tot_wr = tot_wins / (tot_trades + 1e-8) * 100.0
    tot_ev = tot_r / (tot_trades + 1e-8)
    passed_windows = sum(win_flags)
    
    print("-" * 125)
    print(f"{'OVERALL (4-YEARS)':<22} | {'TOTAL':<5} | {tot_trades:<6} | {tot_wr:>6.1f}% | {tot_pf:>5.2f} | ${tot_dollar:>+10.2f} | {tot_r:>+9.2f} R | {tot_ev:>+10.2f} R |")
    print(f"📌 Criteria Check: Positive Windows = {passed_windows}/4 | Overall EV = {tot_ev:+.3f} R/trade | Pass? {'✅ YES' if (passed_windows >= 3 and tot_ev >= 0.05) else '❌ NO'}")
    print("-" * 125)


def run_all_benchmarks():
    print("Loading and preparing H1 Gold data...")
    df_h1 = load_h1_gold_data("data/XAUUSD_M15_Exness.csv")
    print(f"Total H1 bars: {len(df_h1):,} (from {df_h1.index[0]} to {df_h1.index[-1]})")
    
    # Benchmark 1: Pure 20-bar Donchian Breakout (SL 1.0 ATR, TP 2.5 ATR)
    print_walkforward_table(
        df_h1,
        strat_name="1. Pure Donchian 20 Breakout (SL 1.0 ATR, TP 2.5 ATR)",
        strategy_type="fixed_rr",
        tp_mult=2.5,
        sl_mult=1.0,
        use_ema_filter=False,
        max_hold_bars=72
    )
    
    # Benchmark 2: Pure 20-bar Donchian Breakout (SL 1.0 ATR, TP 3.0 ATR)
    print_walkforward_table(
        df_h1,
        strat_name="2. Pure Donchian 20 Breakout (SL 1.0 ATR, TP 3.0 ATR)",
        strategy_type="fixed_rr",
        tp_mult=3.0,
        sl_mult=1.0,
        use_ema_filter=False,
        max_hold_bars=96
    )

    # Benchmark 3: Donchian 20 Breakout + EMA(50/200) Trend Filter (SL 1.0 ATR, TP 2.5 ATR)
    print_walkforward_table(
        df_h1,
        strat_name="3. Donchian 20 Breakout + EMA(50/200) Filter (SL 1.0 ATR, TP 2.5 ATR)",
        strategy_type="fixed_rr",
        tp_mult=2.5,
        sl_mult=1.0,
        use_ema_filter=True,
        max_hold_bars=72
    )

    # Benchmark 4: Classic Turtle Channel Exit (Entry 20-bar Break, Exit 10-bar Reverse Break, Hard SL 1.5 ATR)
    print_walkforward_table(
        df_h1,
        strat_name="4. Turtle Channel Exit (Entry 20-bar, Exit 10-bar Reverse, Hard SL 1.5 ATR)",
        strategy_type="channel_exit",
        sl_mult=1.5,
        use_ema_filter=False,
        max_hold_bars=120
    )


if __name__ == "__main__":
    run_all_benchmarks()
