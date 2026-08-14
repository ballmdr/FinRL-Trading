"""
Unified Training and Backtesting Pipeline for XAUUSD M15 Mean Reversion
-----------------------------------------------------------------------
Executes Strict Sweep Gating, Force-Hold Execution, Trailing Stop,
and selects the best model checkpoint based on Validation Profit Factor & Net PnL.
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


def calculate_quant_metrics(equity_series: pd.Series, trades: list[dict], initial_capital: float = 10_000.0) -> dict[str, float]:
    """Calculate professional quantitative trading performance metrics on real dollar equity."""
    returns = equity_series.pct_change().dropna()

    ending_capital = float(equity_series.iloc[-1])
    net_pnl_dollar = ending_capital - initial_capital
    total_return = (net_pnl_dollar / initial_capital) * 100.0

    # Annualization factor for M15 (252 trading days * 96 15-min bars/day = 24,192 bars/year)
    ann_factor = np.sqrt(24192)

    ret_mean = returns.mean()
    ret_std = returns.std()
    sharpe = float((ret_mean / (ret_std + 1e-9)) * ann_factor) if ret_std > 0 else 0.0

    downside_returns = returns[returns < 0.0]
    downside_std = downside_returns.std()
    sortino = float((ret_mean / (downside_std + 1e-9)) * ann_factor) if (downside_std is not None and downside_std > 0) else 0.0

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
        avg_trade_pnl = sum(pnls) / n_trades
        avg_hold_bars = sum(t["hold_bars"] for t in trades) / n_trades
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        avg_hold_bars = 0.0
        gross_profit = 0.0
        gross_loss = 0.0

    return {
        "Initial Capital ($)": initial_capital,
        "Ending Capital ($)": ending_capital,
        "Net P&L ($)": net_pnl_dollar,
        "Total Return (%)": total_return,
        "Annualized Sharpe": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_drawdown,
        "Total Trades": n_trades,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Gross Profit ($)": gross_profit,
        "Gross Loss ($)": gross_loss,
        "Avg Trade PnL ($)": avg_trade_pnl,
        "Avg Hold (Bars)": avg_hold_bars,
    }


def evaluate_agent(env: XAUUSDMeanRevEnv, agent: DuelingDDQNAgent) -> tuple[float, list[dict], pd.Series]:
    """Run a deterministic evaluation pass through an environment using action masks."""
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    initial_capital = env.initial_capital
    equity = [initial_capital]
    dates = [env.df.index[env.current_step]]

    while not done:
        mask = env.action_masks()
        action = agent.select_action(obs, action_mask=mask, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Real Dollar Equity = Initial Capital + Realized PnL ($) + Unrealized PnL ($)
        current_equity = initial_capital + info["realized_pnl_total"] + info["unrealized_dollar"]
        equity.append(current_equity)
        current_idx = min(env.current_step, len(env.df) - 1)
        dates.append(env.df.index[current_idx])

    equity_series = pd.Series(equity, index=dates)
    return total_reward, env.trades_history, equity_series


def train_and_backtest(
    csv_path: str = "data/XAUUSD_M15_Exness.csv",
    total_timesteps: int = 50_000,
    window_size: int = 12,
    batch_size: int = 128,
    eval_freq: int = 5_000,
    models_dir: str = "models",
    reports_dir: str = "reports",
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 75)
    print("🚀 Starting XAUUSD M15 Strict-Gated Mean Reversion DRL Pipeline (Force-Hold)")
    print("=" * 75)

    # 1. Load and process data
    df = load_and_prepare_xauusd_m15(csv_path=csv_path, start_date="2018-01-01")

    # 2. Time-series Split (Strictly non-random)
    train_end = "2024-06-30"
    val_end = "2025-06-30"

    df_train = df[df.index <= train_end].copy()
    df_val = df[(df.index > train_end) & (df.index <= val_end)].copy()
    df_test = df[df.index > val_end].copy()

    print(f"\n📊 Dataset Splits:")
    print(f"   - Train : {df_train.index[0].date()} -> {df_train.index[-1].date()} ({len(df_train):,} bars)")
    print(f"   - Val   : {df_val.index[0].date()} -> {df_val.index[-1].date()} ({len(df_val):,} bars)")
    print(f"   - Test  : {df_test.index[0].date()} -> {df_test.index[-1].date()} ({len(df_test):,} bars)")

    # 3. Create Environments (Fixed Target R:R 1.8x TP / 1.0x SL with H1 Trend Filter)
    train_env = XAUUSDMeanRevEnv(
        df_train,
        window_size=window_size,
        max_steps_per_episode=384,
        atr_sl_mult=1.0,
        atr_tp_mult=1.8,
        max_hold_bars=36,
        lot_size=0.10,
        initial_capital=10_000.0,
        exit_policy="fixed_tp_sl",
        is_eval=False,
    )
    val_env = XAUUSDMeanRevEnv(
        df_val,
        window_size=window_size,
        max_steps_per_episode=384,
        atr_sl_mult=1.0,
        atr_tp_mult=1.8,
        max_hold_bars=36,
        lot_size=0.10,
        initial_capital=10_000.0,
        exit_policy="fixed_tp_sl",
        is_eval=True,
    )
    test_env = XAUUSDMeanRevEnv(
        df_test,
        window_size=window_size,
        max_steps_per_episode=384,
        atr_sl_mult=1.0,
        atr_tp_mult=1.8,
        max_hold_bars=36,
        lot_size=0.10,
        initial_capital=10_000.0,
        exit_policy="fixed_tp_sl",
        is_eval=True,
    )

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    # 4. Initialize Agent
    agent = DuelingDDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9997,
        buffer_size=150_000,
        batch_size=batch_size,
    )

    print(f"\n🧠 Initialized Dueling DDQN Agent (State Dim: {state_dim}, Action Dim: {action_dim})")
    print(f"🏋️ Training for {total_timesteps:,} steps with Net PnL/Profit Factor validation...\n")

    # 5. Training Loop
    best_score = -np.inf
    best_model_path = os.path.join(models_dir, "xauusd_m15_mean_rev_force_hold_best.pt")

    step = 0
    episodes = 0
    start_time = time.time()

    while step < total_timesteps:
        obs, info = train_env.reset()
        done = False
        ep_reward = 0.0

        while not done and step < total_timesteps:
            mask = train_env.action_masks()
            action = agent.select_action(obs, action_mask=mask, evaluate=False)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            agent.memory.push(obs, action, reward, next_obs, done)
            agent.update()

            obs = next_obs
            ep_reward += reward
            step += 1

            # Periodic Validation Check
            if step % eval_freq == 0:
                val_reward, val_trades, val_equity = evaluate_agent(val_env, agent)
                val_metrics = calculate_quant_metrics(val_equity, val_trades, initial_capital=10_000.0)
                elapsed = time.time() - start_time

                # Checkpoint Score: Balance Profit Factor, Return, and Trade Count
                pf = val_metrics["Profit Factor"]
                ret = val_metrics["Total Return (%)"]
                trades_cnt = val_metrics["Total Trades"]
                score = (pf * 10.0) + ret if trades_cnt >= 15 else (pf - 10.0)

                print(
                    f"Step [{step:6d}/{total_timesteps:6d}] | "
                    f"Eps: {agent.epsilon:.3f} | "
                    f"Net PnL: ${val_metrics['Net P&L ($)']:+7.2f} | "
                    f"Return: {ret:+6.2f}% | "
                    f"PF: {pf:4.2f} | "
                    f"WR: {val_metrics['Win Rate (%)']:4.1f}% | "
                    f"Trades: {trades_cnt:3d} | "
                    f"Time: {elapsed:4.0f}s"
                )

                if score > best_score:
                    best_score = score
                    agent.save(best_model_path)
                    print(f"  ⭐ Saved new best checkpoint (PF: {pf:.2f}, Return: {ret:+.2f}%, Trades: {trades_cnt}) to {best_model_path}")

        episodes += 1

    print(f"\n✅ Training completed in {time.time() - start_time:.1f}s ({episodes} episodes).")

    # 6. Load Best Model for Out-of-Sample Backtesting
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        print(f"📦 Loaded best model from {best_model_path} for final Out-of-Sample evaluation.")

    # 7. Out-of-Sample Backtest on Test Set
    print("\n" + "=" * 75)
    print("📈 OUT-OF-SAMPLE BACKTEST RESULTS (2025.07 - 2026.07)")
    print("=" * 75)

    test_reward, test_trades, test_equity = evaluate_agent(test_env, agent)
    test_metrics = calculate_quant_metrics(test_equity, test_trades, initial_capital=10_000.0)

    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:+10.2f}" if ("Return" in k or "Sharpe" in k or "Drawdown" in k or "Profit" in k or "Loss" in k or "Capital" in k or "P&L" in k) else f"  {k:25s}: {v:10.2f}")
        else:
            print(f"  {k:25s}: {v:10d}")

    # Exit reason breakdown
    if len(test_trades) > 0:
        reasons = [t["reason"] for t in test_trades]
        print("\n🔍 Exit Reasons Breakdown:")
        for r in set(reasons):
            cnt = reasons.count(r)
            print(f"   - {r:15s}: {cnt:4d} trades ({cnt/len(test_trades)*100:5.1f}%)")

    # 8. Save Performance Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Equity Curve
    ax1.plot(test_equity.index, test_equity.values, label="Kalman Mean-Rev DRL Strategy ($10k Base)", color="#00ff88", linewidth=1.8)
    ax1.axhline(10_000.0, color="#888888", linestyle="--", alpha=0.6, label="Starting Capital ($10,000)")
    ax1.set_title("XAUUSD M15 Out-of-Sample Performance (Force-Hold + Trailing Stop DDQN)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Equity ($)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Drawdown
    running_max = test_equity.cummax()
    dd = (test_equity - running_max) / running_max * 100.0
    ax2.fill_between(dd.index, dd.values, 0, color="#ff4444", alpha=0.4, label="Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    chart_path = os.path.join(reports_dir, "xauusd_m15_mean_rev_force_hold_backtest.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print(f"\n📊 Performance chart saved to: {chart_path}")
    print("=" * 75)

    return test_metrics


if __name__ == "__main__":
    train_and_backtest(total_timesteps=40_000, eval_freq=5_000)
