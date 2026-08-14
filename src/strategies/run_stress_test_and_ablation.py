"""
Comprehensive Stress Testing, Walk-Forward, Same-Entry Replay & RL Ablation Suite
--------------------------------------------------------------------------------
Executes 4 rigorous validation tasks for XAUUSD M15 Mean Reversion:
1. Task 1: Same-Entry Replay (Replays all 4 exit policies on identical entry timestamps to kill occupancy bias)
2. Task 2: Walk-Forward Validation (4 temporal windows + Monthly PnL breakdown)
3. Task 3: Cost & Slippage Stress Test (Baseline vs +$5 vs +$8 vs +$12/trade + Bootstrap PF 95% CI)
4. Task 4: Pure Rule Ablation (Rule-only entry vs RL-filtered entry)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.data.xauusd_m15_mean_rev_processor import load_and_prepare_xauusd_m15
from src.strategies.xauusd_mean_rev_env import XAUUSDMeanRevEnv
from src.strategies.dueling_ddqn_agent import DuelingDDQNAgent


def run_same_entry_replay(df_test: pd.DataFrame, base_agent: DuelingDDQNAgent, lot_size: float = 0.10) -> pd.DataFrame:
    """
    Task 1: Same-Entry Replay
    Records exact entry bar indices from Arm 3, then simulates each exit policy
    on the identical entry bars without allowing path-dependent occupancy divergence.
    """
    print("\n" + "=" * 85)
    print("🔬 TASK 1: SAME-ENTRY REPLAY (Kills Occupancy Bias)")
    print("=" * 85)

    # 1. Generate base entry signals using Arm 3 (Fixed TP/SL)
    env_base = XAUUSDMeanRevEnv(df_test, exit_policy="fixed_tp_sl", lot_size=lot_size, is_eval=True)
    obs, info = env_base.reset()
    done = False

    entries_record = []  # list of (step_index, position, entry_price, atr, spread)

    while not done:
        mask = env_base.action_masks()
        action = base_agent.select_action(obs, action_mask=mask, evaluate=True)

        if env_base.position == 0 and action in (1, 2) and mask[action]:
            curr_step = env_base.current_step
            curr_close = env_base.close_np[curr_step]
            curr_atr = env_base.atr_np[curr_step]
            curr_spread = env_base.spread_np[curr_step]
            pos = 1 if action == 1 else -1
            entry_p = curr_close + (0.5 * curr_spread if pos == 1 else -0.5 * curr_spread)
            entries_record.append({
                "step": curr_step,
                "datetime": df_test.index[curr_step],
                "position": pos,
                "entry_price": entry_p,
                "atr": curr_atr,
                "spread": curr_spread,
            })

        obs, reward, term, trunc, info = env_base.step(action)
        done = term or trunc

    print(f"Recorded {len(entries_record)} baseline entry signals across {len(df_test):,} bars.")

    # 2. Replay all 4 exit policies on this EXACT entry list
    policies = [
        {"name": "Fixed Target R:R (1.8/1.0)", "type": "fixed", "tp_atr": 1.8, "sl_atr": 1.0},
        {"name": "BE-then-Trail (1.0/1.4/0.5)", "type": "be_trail", "tp_atr": 1.8, "sl_atr": 1.0, "be_trig": 1.0, "trail_trig": 1.4, "trail_dist": 0.5},
        {"name": "Dynamic Mean-Target", "type": "mean_target", "tp_atr": 1.8, "sl_atr": 1.0},
        {"name": "Agent Discretionary (+1.0 ATR)", "type": "agent_discretion", "tp_atr": 1.8, "sl_atr": 1.0},
    ]

    close_np = df_test["close_raw"].values
    high_np = df_test["high_raw"].values
    low_np = df_test["low_raw"].values
    kf_np = df_test["kf_price_raw"].values
    spread_np = df_test["spread_raw"].values
    atr_np = df_test["atr_raw"].values
    n_bars = len(df_test)

    replay_results = []

    for pol in policies:
        trades = []
        for e in entries_record:
            start_bar = e["step"]
            pos = e["position"]
            entry_p = e["entry_price"]
            entry_atr = e["atr"]
            sl_p = entry_p - (pol["sl_atr"] * entry_atr) if pos == 1 else entry_p + (pol["sl_atr"] * entry_atr)
            tp_p = entry_p + (pol["tp_atr"] * entry_atr) if pos == 1 else entry_p - (pol["tp_atr"] * entry_atr)

            be_active = False
            trailing_active = False
            peak_p = high_np[start_bar] if pos == 1 else low_np[start_bar]

            exit_price = None
            exit_reason = None
            hold_bars = 0

            for bar in range(start_bar + 1, min(start_bar + 33, n_bars)):
                hold_bars += 1
                c_high = high_np[bar]
                c_low = low_np[bar]
                c_close = close_np[bar]
                c_spread = spread_np[bar]
                c_kf = kf_np[bar]
                c_atr = atr_np[bar]

                if pos == 1:
                    peak_p = max(peak_p, c_high)
                    profit_atr = (peak_p - entry_p) / c_atr

                    if pol["type"] == "be_trail":
                        if not be_active and profit_atr >= pol["be_trig"]:
                            be_active = True
                            sl_p = max(sl_p, entry_p + 0.1 * c_atr)
                        if not trailing_active and profit_atr >= pol["trail_trig"]:
                            trailing_active = True
                        if trailing_active:
                            sl_p = max(sl_p, peak_p - pol["trail_dist"] * c_atr)

                    elif pol["type"] == "mean_target":
                        if c_close >= c_kf and (c_close - entry_p) >= 0.5 * c_atr:
                            exit_price = c_close - 0.5 * c_spread
                            exit_reason = "MEAN_REVERTED"
                            break

                    elif pol["type"] == "agent_discretion":
                        if (c_close - entry_p) >= 1.0 * c_atr:
                            exit_price = c_close - 0.5 * c_spread
                            exit_reason = "AGENT_EXIT"
                            break

                    if c_high >= tp_p:
                        exit_price = tp_p - 0.5 * c_spread
                        exit_reason = "TP"
                        break
                    elif c_low <= sl_p:
                        exit_price = sl_p - 0.5 * c_spread
                        exit_reason = "BE_STOP" if be_active else "SL"
                        break
                    elif hold_bars >= 32:
                        exit_price = c_close - 0.5 * c_spread
                        exit_reason = "MAX_HOLD"
                        break

                else:  # pos == -1 (Short)
                    peak_p = min(peak_p, c_low)
                    profit_atr = (entry_p - peak_p) / c_atr

                    if pol["type"] == "be_trail":
                        if not be_active and profit_atr >= pol["be_trig"]:
                            be_active = True
                            sl_p = min(sl_p, entry_p - 0.1 * c_atr)
                        if not trailing_active and profit_atr >= pol["trail_trig"]:
                            trailing_active = True
                        if trailing_active:
                            sl_p = min(sl_p, peak_p + pol["trail_dist"] * c_atr)

                    elif pol["type"] == "mean_target":
                        if c_close <= c_kf and (entry_p - c_close) >= 0.5 * c_atr:
                            exit_price = c_close + 0.5 * c_spread
                            exit_reason = "MEAN_REVERTED"
                            break

                    elif pol["type"] == "agent_discretion":
                        if (entry_p - c_close) >= 1.0 * c_atr:
                            exit_price = c_close + 0.5 * c_spread
                            exit_reason = "AGENT_EXIT"
                            break

                    if c_low <= tp_p:
                        exit_price = tp_p + 0.5 * c_spread
                        exit_reason = "TP"
                        break
                    elif c_high >= sl_p:
                        exit_price = sl_p + 0.5 * c_spread
                        exit_reason = "BE_STOP" if be_active else "SL"
                        break
                    elif hold_bars >= 32:
                        exit_price = c_close + 0.5 * c_spread
                        exit_reason = "MAX_HOLD"
                        break

            if exit_price is None:
                exit_price = close_np[min(start_bar + 32, n_bars - 1)]
                exit_reason = "MAX_HOLD"

            pnl_price = (exit_price - entry_p) if pos == 1 else (entry_p - exit_price)
            pnl_dollar = pnl_price * (lot_size * 100.0)
            trades.append({"pnl_dollar": pnl_dollar, "reason": exit_reason, "hold_bars": hold_bars})

        # Calculate exact metrics on identical entries
        wins = [t["pnl_dollar"] for t in trades if t["pnl_dollar"] > 0]
        losses = [t["pnl_dollar"] for t in trades if t["pnl_dollar"] < 0]
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-4
        net_pnl = gross_win - gross_loss
        pf = gross_win / gross_loss
        wr = (len(wins) / len(trades)) * 100.0
        avg_w = gross_win / len(wins) if wins else 0.0
        avg_l = gross_loss / len(losses) if losses else 1e-4
        payoff_r = avg_w / avg_l
        tp_cnt = len([t for t in trades if t["reason"] == "TP"])
        tp_rate = (tp_cnt / len(trades)) * 100.0

        replay_results.append({
            "Exit Policy": pol["name"],
            "Same Entries": len(trades),
            "Net P&L ($)": f"${net_pnl:+,.2f}",
            "Win Rate (%)": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Payoff R": f"{payoff_r:.2f}",
            "Avg Win ($)": f"${avg_w:.2f}",
            "Avg Loss ($)": f"${avg_l:.2f}",
            "TP Rate (%)": f"{tp_rate:.1f}% ({tp_cnt})",
        })

    df_replay = pd.DataFrame(replay_results)
    print(df_replay.to_string(index=False))
    return df_replay


def run_walk_forward_and_monthly(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Task 2: Walk-Forward Validation across 4 Windows + Monthly P&L Breakdown
    """
    print("\n" + "=" * 85)
    print("📈 TASK 2: WALK-FORWARD VALIDATION & MONTHLY CLUSTERING ANALYSIS")
    print("=" * 85)

    windows = [
        {"name": "Window 1 (2022-2023)", "train_end": "2022-06-30", "test_start": "2022-07-01", "test_end": "2023-06-30"},
        {"name": "Window 2 (2023-2024)", "train_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2024-06-30"},
        {"name": "Window 3 (2024-2025)", "train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2025-06-30"},
        {"name": "Window 4 (2025-2026)", "train_end": "2025-06-30", "test_start": "2025-07-01", "test_end": "2026-07-17"},
    ]

    wf_results = []
    monthly_pnls = []

    for w in windows:
        df_train = df_all[df_all.index <= w["train_end"]].copy()
        df_test = df_all[(df_all.index >= w["test_start"]) & (df_all.index <= w["test_end"])].copy()

        train_env = XAUUSDMeanRevEnv(df_train, exit_policy="fixed_tp_sl", lot_size=0.10, is_eval=False)
        test_env = XAUUSDMeanRevEnv(df_test, exit_policy="fixed_tp_sl", lot_size=0.10, is_eval=True)

        agent = DuelingDDQNAgent(
            state_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.n,
            hidden_dim=256,
            lr=3e-4,
            buffer_size=80_000,
            batch_size=128,
        )

        # Quick train on window
        step = 0
        while step < 18_000:
            obs, _ = train_env.reset()
            done = False
            while not done and step < 18_000:
                mask = train_env.action_masks()
                action = agent.select_action(obs, action_mask=mask, evaluate=False)
                next_obs, rew, term, trunc, _ = train_env.step(action)
                done = term or trunc
                agent.memory.push(obs, action, rew, next_obs, done)
                agent.update()
                obs = next_obs
                step += 1

        # Test
        obs, _ = test_env.reset()
        done = False
        while not done:
            mask = test_env.action_masks()
            action = agent.select_action(obs, action_mask=mask, evaluate=True)
            obs, _, term, trunc, _ = test_env.step(action)
            done = term or trunc

        trades = test_env.trades_history
        wins = [t["pnl_dollar"] for t in trades if t["pnl_dollar"] > 0]
        losses = [t["pnl_dollar"] for t in trades if t["pnl_dollar"] < 0]
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-4
        net_pnl = gross_win - gross_loss
        pf = gross_win / gross_loss
        wr = (len(wins) / len(trades) * 100.0) if trades else 0.0

        wf_results.append({
            "Window": w["name"],
            "Bars": len(df_test),
            "Trades": len(trades),
            "Win Rate (%)": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Net P&L ($)": f"${net_pnl:+,.2f}",
            "Status": "🟢 POSITIVE" if net_pnl > 0 else "🔴 NEGATIVE",
        })

        # Collect monthly PnL for window 4
        if "Window 4" in w["name"]:
            for t in trades:
                t_dt = df_test.index[t["step"]]
                month_key = t_dt.strftime("%Y-%m")
                monthly_pnls.append({"month": month_key, "pnl": t["pnl_dollar"]})

    df_wf = pd.DataFrame(wf_results)
    print(df_wf.to_string(index=False))

    df_m = pd.DataFrame(monthly_pnls).groupby("month")["pnl"].sum().reset_index() if monthly_pnls else pd.DataFrame()
    print("\n📅 Window 4 Monthly P&L Breakdown:")
    print(df_m.to_string(index=False) if not df_m.empty else "No trades recorded")

    return df_wf, df_m


def run_cost_stress_and_bootstrap(df_test: pd.DataFrame, base_agent: DuelingDDQNAgent) -> pd.DataFrame:
    """
    Task 3: Cost Stress Testing (+$5, +$8, +$12 per trade) & 1,000-Iteration Bootstrap PF CI
    """
    print("\n" + "=" * 85)
    print("⚡ TASK 3: COST STRESS TEST & BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 85)

    test_env = XAUUSDMeanRevEnv(df_test, exit_policy="fixed_tp_sl", lot_size=0.10, is_eval=True)
    obs, _ = test_env.reset()
    done = False
    while not done:
        mask = test_env.action_masks()
        action = base_agent.select_action(obs, action_mask=mask, evaluate=True)
        obs, _, term, trunc, _ = test_env.step(action)
        done = term or trunc

    base_trades = [t["pnl_dollar"] for t in test_env.trades_history]
    n_trades = len(base_trades)
    print(f"Base Trades: {n_trades} trades")

    cost_penalties = [0.0, 5.0, 8.0, 12.0]
    stress_rows = []

    for cost in cost_penalties:
        adjusted_pnls = np.array(base_trades) - cost
        wins = adjusted_pnls[adjusted_pnls > 0]
        losses = abs(adjusted_pnls[adjusted_pnls < 0])
        gross_w = wins.sum() if len(wins) > 0 else 0.0
        gross_l = losses.sum() if len(losses) > 0 else 1e-4
        net_p = gross_w - gross_l
        pf = gross_w / gross_l
        wr = (len(wins) / n_trades) * 100.0

        # Bootstrap 1,000 resamples for 95% CI of Profit Factor
        boot_pfs = []
        for _ in range(1000):
            sample = np.random.choice(adjusted_pnls, size=n_trades, replace=True)
            s_w = sample[sample > 0].sum()
            s_l = abs(sample[sample < 0].sum()) if len(sample[sample < 0]) > 0 else 1e-4
            boot_pfs.append(s_w / s_l)

        ci_5 = np.percentile(boot_pfs, 5)
        ci_95 = np.percentile(boot_pfs, 95)

        stress_rows.append({
            "Cost Stress": f"Baseline + ${cost:.0f}/trade",
            "Net P&L ($)": f"${net_p:+,.2f}",
            "Profit Factor": f"{pf:.2f}",
            "Win Rate (%)": f"{wr:.1f}%",
            "Bootstrap 90% CI (5th-95th)": f"[{ci_5:.2f} , {ci_95:.2f}]",
            "Lower Bound > 1.0?": "✅ YES" if ci_5 > 1.0 else "❌ NO (Insignificant)",
        })

    df_stress = pd.DataFrame(stress_rows)
    print(df_stress.to_string(index=False))
    return df_stress


def run_rule_ablation(df_test: pd.DataFrame, base_agent: DuelingDDQNAgent) -> pd.DataFrame:
    """
    Task 4: Pure Technical Rule vs RL-Agent Filtered Entry Ablation
    """
    print("\n" + "=" * 85)
    print("⚖️ TASK 4: PURE TECHNICAL RULE VS RL FILTER ABLATION")
    print("=" * 85)

    # 1. Pure Rule: Enter every single long/short gate signal unconditionally with 1.8/1.0 TP/SL
    rule_env = XAUUSDMeanRevEnv(df_test, exit_policy="fixed_tp_sl", lot_size=0.10, is_eval=True)
    obs, _ = rule_env.reset()
    done = False
    while not done:
        mask = rule_env.action_masks()
        # Pure rule action: if long gate -> action 1; if short gate -> action 2; else 0
        if mask[1]:
            action = 1
        elif mask[2]:
            action = 2
        else:
            action = 0

        obs, _, term, trunc, _ = rule_env.step(action)
        done = term or trunc

    rule_trades = rule_env.trades_history
    r_wins = [t["pnl_dollar"] for t in rule_trades if t["pnl_dollar"] > 0]
    r_losses = [t["pnl_dollar"] for t in rule_trades if t["pnl_dollar"] < 0]
    r_gw = sum(r_wins)
    r_gl = abs(sum(r_losses))
    r_net = r_gw - r_gl
    r_pf = r_gw / r_gl
    r_wr = (len(r_wins) / len(rule_trades)) * 100.0

    # 2. RL Agent Filtered
    rl_env = XAUUSDMeanRevEnv(df_test, exit_policy="fixed_tp_sl", lot_size=0.10, is_eval=True)
    obs, _ = rl_env.reset()
    done = False
    while not done:
        mask = rl_env.action_masks()
        action = base_agent.select_action(obs, action_mask=mask, evaluate=True)
        obs, _, term, trunc, _ = rl_env.step(action)
        done = term or trunc

    rl_trades = rl_env.trades_history
    rl_wins = [t["pnl_dollar"] for t in rl_trades if t["pnl_dollar"] > 0]
    rl_losses = [t["pnl_dollar"] for t in rl_trades if t["pnl_dollar"] < 0]
    rl_gw = sum(rl_wins)
    rl_gl = abs(sum(rl_losses))
    rl_net = rl_gw - rl_gl
    rl_pf = rl_gw / rl_gl
    rl_wr = (len(rl_wins) / len(rl_trades)) * 100.0

    ablation_rows = [
        {
            "System Architecture": "Pure Rule (Sweep + Session Gate)",
            "Total Trades": len(rule_trades),
            "Win Rate (%)": f"{r_wr:.1f}%",
            "Profit Factor": f"{r_pf:.2f}",
            "Gross Profit ($)": f"${r_gw:+,.2f}",
            "Gross Loss ($)": f"${r_gl:,.2f}",
            "Net P&L ($)": f"${r_net:+,.2f}",
        },
        {
            "System Architecture": "RL Filtered (DDQN + Sweep Gate)",
            "Total Trades": len(rl_trades),
            "Win Rate (%)": f"{rl_wr:.1f}%",
            "Profit Factor": f"{rl_pf:.2f}",
            "Gross Profit ($)": f"${rl_gw:+,.2f}",
            "Gross Loss ($)": f"${rl_gl:,.2f}",
            "Net P&L ($)": f"${rl_net:+,.2f}",
        },
    ]

    df_ablation = pd.DataFrame(ablation_rows)
    print(df_ablation.to_string(index=False))
    return df_ablation


def run_full_suite():
    df_all = load_and_prepare_xauusd_m15(csv_path="data/XAUUSD_M15_Exness.csv", start_date="2018-01-01")

    train_end = "2024-06-30"
    df_train = df_all[df_all.index <= train_end].copy()
    df_test = df_all[df_all.index > "2025-06-30"].copy()

    # Train base Arm 3 agent
    train_env = XAUUSDMeanRevEnv(df_train, exit_policy="fixed_tp_sl", lot_size=0.10, is_eval=False)
    agent = DuelingDDQNAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.n,
        hidden_dim=256,
        lr=3e-4,
        buffer_size=100_000,
        batch_size=128,
    )

    print("\n🏋️ Training Base Fixed R:R Agent for Stress Suite (25,000 steps)...")
    step = 0
    while step < 25_000:
        obs, _ = train_env.reset()
        done = False
        while not done and step < 25_000:
            mask = train_env.action_masks()
            action = agent.select_action(obs, action_mask=mask, evaluate=False)
            next_obs, rew, term, trunc, _ = train_env.step(action)
            done = term or trunc
            agent.memory.push(obs, action, rew, next_obs, done)
            agent.update()
            obs = next_obs
            step += 1

    # Execute all 4 Stress & Ablation Tasks
    run_same_entry_replay(df_test, agent)
    run_cost_stress_and_bootstrap(df_test, agent)
    run_rule_ablation(df_test, agent)
    run_walk_forward_and_monthly(df_all)


if __name__ == "__main__":
    run_full_suite()
