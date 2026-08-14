"""
Comprehensive 4-Arm A/B Exit Policy Experiment & Walk-Forward Evaluation
-------------------------------------------------------------------------
A/B tests 4 distinct exit paradigms on XAUUSD M15 Mean Reversion:
1. Arm 1: BE-then-Trail (BE at +1.0x ATR, Trail at +1.4x ATR @ 0.5x dist, TP 1.8x ATR, SL 1.0x ATR)
2. Arm 2: Dynamic Mean-Target (Exit on Kalman Mean / Midline reversion, TP 1.8x ATR, SL 1.0x ATR)
3. Arm 3: Fixed Target R:R (Pure TP 1.8x ATR / SL 1.0x ATR without scratch)
4. Arm 4: Agent Exit After +1.0x ATR (Discretionary closing only after profit >= 1.0x ATR)
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.xauusd_m15_mean_rev_processor import load_and_prepare_xauusd_m15
from src.strategies.xauusd_mean_rev_env import XAUUSDMeanRevEnv
from src.strategies.dueling_ddqn_agent import DuelingDDQNAgent


def calculate_quant_metrics(equity_series: pd.Series, trades: list[dict], initial_capital: float = 10_000.0) -> dict[str, any]:
    """Calculate professional quantitative trading performance metrics including Payoff Ratio."""
    returns = equity_series.pct_change().dropna()

    ending_capital = float(equity_series.iloc[-1])
    net_pnl_dollar = ending_capital - initial_capital
    total_return = (net_pnl_dollar / initial_capital) * 100.0

    ann_factor = np.sqrt(24192)
    ret_mean = returns.mean()
    ret_std = returns.std()
    sharpe = float((ret_mean / (ret_std + 1e-9)) * ann_factor) if ret_std > 0 else 0.0

    # Drawdowns
    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown = float(drawdowns.min() * 100.0)

    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t["pnl_dollar"] for t in trades]
        wins = [p for p in pnls if p > 0.0]
        losses = [p for p in pnls if p < 0.0]
        win_rate = (len(wins) / n_trades) * 100.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-4
        profit_factor = gross_profit / gross_loss

        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 1e-4
        payoff_r = avg_win / avg_loss if avg_loss > 0 else 0.0

        tp_trades = [t for t in trades if t["reason"] == "TP"]
        tp_rate = (len(tp_trades) / n_trades) * 100.0
        avg_hold_bars = sum(t["hold_bars"] for t in trades) / n_trades
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        payoff_r = 0.0
        tp_rate = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        avg_hold_bars = 0.0

    # Criteria Evaluation
    pass_criteria = (profit_factor >= 1.20) and (payoff_r >= 0.80) and (max_drawdown >= -15.0)
    kill_criteria = (avg_win < 60.0) or (tp_rate < 15.0)

    return {
        "Initial Capital ($)": initial_capital,
        "Ending Capital ($)": ending_capital,
        "Net P&L ($)": net_pnl_dollar,
        "Total Return (%)": total_return,
        "Annualized Sharpe": sharpe,
        "Max Drawdown (%)": max_drawdown,
        "Total Trades": n_trades,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Gross Profit ($)": gross_profit,
        "Gross Loss ($)": gross_loss,
        "Avg Win ($)": avg_win,
        "Avg Loss ($)": avg_loss,
        "Payoff Ratio (R)": payoff_r,
        "TP Rate (%)": tp_rate,
        "Avg Hold (Bars)": avg_hold_bars,
        "Pass Criteria": pass_criteria,
        "Kill Triggered": kill_criteria,
    }


def evaluate_env(env: XAUUSDMeanRevEnv, agent: DuelingDDQNAgent) -> tuple[dict[str, any], pd.Series, list[dict]]:
    """Run an evaluation pass on an environment using action masks."""
    obs, info = env.reset()
    done = False

    initial_capital = env.initial_capital
    equity = [initial_capital]
    dates = [env.df.index[env.current_step]]

    while not done:
        mask = env.action_masks()
        action = agent.select_action(obs, action_mask=mask, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_equity = initial_capital + info["realized_pnl_total"] + info["unrealized_dollar"]
        equity.append(current_equity)
        current_idx = min(env.current_step, len(env.df) - 1)
        dates.append(env.df.index[current_idx])

    equity_series = pd.Series(equity, index=dates)
    metrics = calculate_quant_metrics(equity_series, env.trades_history, initial_capital=initial_capital)
    return metrics, equity_series, env.trades_history


def run_ab_experiment(
    csv_path: str = "data/XAUUSD_M15_Exness.csv",
    reports_dir: str = "reports",
):
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 RUNNING 4-ARM EXIT POLICY A/B EXPERIMENT (XAUUSD M15 MEAN REVERSION)")
    print("=" * 80)

    df = load_and_prepare_xauusd_m15(csv_path=csv_path, start_date="2018-01-01")

    # Time Splits
    train_end = "2024-06-30"
    val_end = "2025-06-30"

    df_train = df[df.index <= train_end].copy()
    df_val = df[(df.index > train_end) & (df.index <= val_end)].copy()
    df_test = df[df.index > val_end].copy()

    print(f"\n📊 Data Splits:")
    print(f"   - Train (2018-2024): {len(df_train):,} bars")
    print(f"   - Val   (2024-2025): {len(df_val):,} bars")
    print(f"   - Test  (2025-2026): {len(df_test):,} bars (Out-of-Sample)")

    # 4 Exit Policy Configurations to test
    arms_config = [
        {
            "name": "Arm 1: BE-then-Trail",
            "policy": "be_then_trail",
            "atr_sl": 1.0,
            "atr_tp": 1.8,
            "be_trigger": 1.0,
            "trail_trigger": 1.4,
            "trail_dist": 0.5,
            "color": "#00ff88",
        },
        {
            "name": "Arm 2: Dynamic Mean-Target",
            "policy": "mean_target",
            "atr_sl": 1.0,
            "atr_tp": 1.8,
            "be_trigger": 1.0,
            "trail_trigger": 1.4,
            "trail_dist": 0.5,
            "color": "#00d4ff",
        },
        {
            "name": "Arm 3: Fixed Target R:R (1.8/1.0)",
            "policy": "fixed_tp_sl",
            "atr_sl": 1.0,
            "atr_tp": 1.8,
            "be_trigger": 99.0,
            "trail_trigger": 99.0,
            "trail_dist": 99.0,
            "color": "#ffaa00",
        },
        {
            "name": "Arm 4: Agent Exit After +1.0x ATR",
            "policy": "agent_after_1atr",
            "atr_sl": 1.0,
            "atr_tp": 1.8,
            "be_trigger": 99.0,
            "trail_trigger": 99.0,
            "trail_dist": 99.0,
            "color": "#ff00ff",
        },
    ]

    results_table = []
    equity_curves = {}

    plt.figure(figsize=(14, 8))

    for arm in arms_config:
        arm_name = arm["name"]
        print(f"\n--- Testing {arm_name} ---")

        # 1. Train Agent for this specific policy
        train_env = XAUUSDMeanRevEnv(
            df_train,
            window_size=12,
            max_steps_per_episode=384,
            exit_policy=arm["policy"],
            atr_sl_mult=arm["atr_sl"],
            atr_tp_mult=arm["atr_tp"],
            be_trigger_atr=arm["be_trigger"],
            trail_trigger_atr=arm["trail_trigger"],
            trail_dist_atr=arm["trail_dist"],
            lot_size=0.10,
            initial_capital=10_000.0,
            is_eval=False,
        )
        val_env = XAUUSDMeanRevEnv(
            df_val,
            window_size=12,
            exit_policy=arm["policy"],
            atr_sl_mult=arm["atr_sl"],
            atr_tp_mult=arm["atr_tp"],
            be_trigger_atr=arm["be_trigger"],
            trail_trigger_atr=arm["trail_trigger"],
            trail_dist_atr=arm["trail_dist"],
            lot_size=0.10,
            initial_capital=10_000.0,
            is_eval=True,
        )
        test_env = XAUUSDMeanRevEnv(
            df_test,
            window_size=12,
            exit_policy=arm["policy"],
            atr_sl_mult=arm["atr_sl"],
            atr_tp_mult=arm["atr_tp"],
            be_trigger_atr=arm["be_trigger"],
            trail_trigger_atr=arm["trail_trigger"],
            trail_dist_atr=arm["trail_dist"],
            lot_size=0.10,
            initial_capital=10_000.0,
            is_eval=True,
        )

        agent = DuelingDDQNAgent(
            state_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.n,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9997,
            buffer_size=100_000,
            batch_size=128,
        )

        # Train 25,000 steps with validation checkpointing
        step = 0
        best_score = -np.inf
        best_agent_weights = None

        while step < 25_000:
            obs, _ = train_env.reset()
            done = False
            while not done and step < 25_000:
                mask = train_env.action_masks()
                action = agent.select_action(obs, action_mask=mask, evaluate=False)
                next_obs, reward, term, trunc, _ = train_env.step(action)
                done = term or trunc

                agent.memory.push(obs, action, reward, next_obs, done)
                agent.update()

                obs = next_obs
                step += 1

                if step % 5_000 == 0:
                    val_metrics, _, _ = evaluate_env(val_env, agent)
                    pf = val_metrics["Profit Factor"]
                    ret = val_metrics["Total Return (%)"]
                    cnt = val_metrics["Total Trades"]
                    score = (pf * 10.0) + ret if cnt >= 10 else (pf - 10.0)

                    if score > best_score:
                        best_score = score
                        best_agent_weights = {k: v.cpu() for k, v in agent.q_net.state_dict().items()}

        if best_agent_weights is not None:
            agent.q_net.load_state_dict({k: v.to(agent.q_net.parameters().__next__().device) for k, v in best_agent_weights.items()})

        # Evaluate Out-of-Sample
        test_metrics, test_equity, test_trades = evaluate_env(test_env, agent)
        equity_curves[arm_name] = test_equity

        # Exit reasons breakdown
        reasons_summary = {}
        for t in test_trades:
            r = t["reason"]
            reasons_summary[r] = reasons_summary.get(r, 0) + 1

        print(f"  Trades: {test_metrics['Total Trades']} | WR: {test_metrics['Win Rate (%)']:.1f}% | PF: {test_metrics['Profit Factor']:.2f}")
        print(f"  Net PnL: ${test_metrics['Net P&L ($)']:+,.2f} | Return: {test_metrics['Total Return (%)']:+.2f}% | Max DD: {test_metrics['Max Drawdown (%)']:.2f}%")
        print(f"  Avg Win: ${test_metrics['Avg Win ($)']:.2f} | Avg Loss: ${test_metrics['Avg Loss ($)']:.2f} | Payoff R: {test_metrics['Payoff Ratio (R)']:.2f}")
        print(f"  TP Rate: {test_metrics['TP Rate (%)']:.1f}% | Exits: {reasons_summary}")
        print(f"  Pass: {test_metrics['Pass Criteria']} | Kill Triggered: {test_metrics['Kill Triggered']}")

        results_table.append({
            "Arm": arm_name,
            "Net P&L ($)": f"${test_metrics['Net P&L ($)']:+,.2f}",
            "Return (%)": f"{test_metrics['Total Return (%)']:+.2f}%",
            "Win Rate (%)": f"{test_metrics['Win Rate (%)']:.1f}%",
            "Profit Factor": f"{test_metrics['Profit Factor']:.2f}",
            "Payoff R": f"{test_metrics['Payoff Ratio (R)']:.2f}",
            "Avg Win ($)": f"${test_metrics['Avg Win ($)']:.2f}",
            "Avg Loss ($)": f"${test_metrics['Avg Loss ($)']:.2f}",
            "Max DD (%)": f"{test_metrics['Max Drawdown (%)']:.2f}%",
            "Total Trades": test_metrics["Total Trades"],
            "TP Rate (%)": f"{test_metrics['TP Rate (%)']:.1f}%",
            "Pass": "✅ PASS" if test_metrics["Pass Criteria"] else "❌ FAIL",
            "Kill": "⚠️ KILL" if test_metrics["Kill Triggered"] else "🟢 OK",
        })

        plt.plot(test_equity.index, test_equity.values, label=f"{arm_name} (PF: {test_metrics['Profit Factor']:.2f})", color=arm["color"], linewidth=1.8)

    # Finalize Plot
    plt.axhline(10_000.0, color="#888888", linestyle="--", alpha=0.6, label="Base Capital ($10,000)")
    plt.title("XAUUSD M15 Out-of-Sample A/B Test: Exit Policy Payoff Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Portfolio Equity ($)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    chart_path = os.path.join(reports_dir, "xauusd_m15_ab_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print("\n" + "=" * 80)
    print("📋 SUMMARY A/B COMPARISON TABLE")
    print("=" * 80)
    df_results = pd.DataFrame(results_table)
    print(df_results.to_string(index=False))
    print(f"\n📊 Comparative chart saved to: {chart_path}")
    print("=" * 80)

    return df_results


if __name__ == "__main__":
    run_ab_experiment()
