"""
Synthetic Pooled DRL Pipeline & TSTR Evaluation Engine (FinRL-Trading)
----------------------------------------------------------------------
Executes TSTR (Train on Synthetic, Test on Real):
1. Trains Dueling Double DQN on 668,160 synthetic continuity rows (9 symbols, 5 stress regimes).
2. Uses Real Validation episodes to checkpoint the best model.
3. Evaluates out-of-sample on 668,160 real historical market rows (real_only.parquet).
4. Produces comprehensive multi-asset quantitative reporting:
   - Multi-Asset Pooled Portfolio (9 symbols)
   - XAUUSD Gold Specialist
   - Per-Symbol Performance Table
5. Generates performance plots and saves best model checkpoints.
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.synthetic_pooled_processor import load_synthetic_pooled_data, extract_episodes_dict
from src.strategies.pooled_episode_env import PooledEpisodeEnv
from src.strategies.dueling_ddqn_agent import DuelingDDQNAgent


def evaluate_tstr_dataset(
    episodes_dict: dict[str, dict[str, np.ndarray]],
    agent: DuelingDDQNAgent,
    window_size: int = 8,
    initial_capital: float = 10_000.0,
    title: str = "Evaluation",
) -> tuple[dict[str, any], pd.Series, list[dict]]:
    """Evaluates agent deterministically across all episodes in the dataset."""
    env = PooledEpisodeEnv(
        episodes_dict=episodes_dict,
        window_size=window_size,
        atr_sl_mult=1.0,
        atr_tp_mult=1.8,
        initial_capital=initial_capital,
        is_eval=True,
    )

    n_episodes = len(episodes_dict)
    equity = [initial_capital]

    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action = agent.select_action(obs, action_mask=mask, evaluate=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

        curr_eq = initial_capital + env.realized_pnl_total
        equity.append(curr_eq)

    trades = env.all_trades_history
    n_t = len(trades)

    if n_t > 0:
        pnls = [t["pnl_dollar"] for t in trades]
        wins = [p for p in pnls if p > 0.0]
        losses = [p for p in pnls if p < 0.0]
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-4
        net_pnl = gross_profit - gross_loss
        total_return = (net_pnl / initial_capital) * 100.0
        win_rate = (len(wins) / n_t) * 100.0
        profit_factor = gross_profit / gross_loss
        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 1e-4
        payoff_r = avg_win / avg_loss
        tp_cnt = len([t for t in trades if t["reason"] == "TP"])
        tp_rate = (tp_cnt / n_t) * 100.0
    else:
        gross_profit = 0.0
        gross_loss = 0.0
        net_pnl = 0.0
        total_return = 0.0
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        payoff_r = 0.0
        tp_cnt = 0
        tp_rate = 0.0

    # Drawdown
    equity_series = pd.Series(equity)
    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown = float(drawdowns.min() * 100.0)

    metrics = {
        "Evaluation": title,
        "Total Episodes": n_episodes,
        "Total Trades": n_t,
        "Net P&L ($)": net_pnl,
        "Total Return (%)": total_return,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Payoff Ratio (R)": payoff_r,
        "Avg Win ($)": avg_win,
        "Avg Loss ($)": avg_loss,
        "Max Drawdown (%)": max_drawdown,
        "TP Rate (%)": tp_rate,
        "Gross Profit ($)": gross_profit,
        "Gross Loss ($)": gross_loss,
    }

    return metrics, equity_series, trades


def train_and_evaluate_tstr(
    total_timesteps: int = 50_000,
    window_size: int = 8,
    batch_size: int = 128,
    eval_freq: int = 10_000,
    models_dir: str = "models",
    reports_dir: str = "reports",
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 85)
    print("🚀 LAUNCHING MULTI-ASSET TSTR (TRAIN ON SYNTHETIC, TEST ON REAL)")
    print("=" * 85)

    # 1. Load Datasets
    df_syn, df_real = load_synthetic_pooled_data()
    syn_episodes = extract_episodes_dict(df_syn)
    real_episodes = extract_episodes_dict(df_real)

    # Real Validation subset (e.g. first 60 real episodes for periodic checkpointing)
    real_keys = list(real_episodes.keys())
    val_real_episodes = {k: real_episodes[k] for k in real_keys[:60]}
    test_real_episodes = {k: real_episodes[k] for k in real_keys[60:]}

    # Extract XAUUSD-only Real Episodes for specialist evaluation
    xauusd_real_episodes = {k: v for k, v in real_episodes.items() if v["symbol"] == "XAUUSD"}

    print(f"\n📊 Extracted Episode Sets:")
    print(f"   - Synthetic Training Episodes: {len(syn_episodes):,} episodes (668,160 bars across 9 symbols)")
    print(f"   - Real Test Set (Out-of-Sample): {len(real_episodes):,} episodes (668,160 bars)")
    print(f"   - Real XAUUSD Specialist Set : {len(xauusd_real_episodes):,} episodes (74,240 bars)")

    # 2. Setup Training Environment
    train_env = PooledEpisodeEnv(
        episodes_dict=syn_episodes,
        window_size=window_size,
        atr_sl_mult=1.0,
        atr_tp_mult=1.8,
        initial_capital=10_000.0,
        is_eval=False,
    )

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    agent = DuelingDDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=2e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9997,
        buffer_size=150_000,
        batch_size=batch_size,
    )

    print(f"\n🧠 Initialized Dueling DDQN Agent (State Dim: {state_dim}, Action Dim: {action_dim})")
    print(f"🏋️ Training for {total_timesteps:,} steps across 1,152 Synthetic Episodes...\n")

    # 3. Training Loop
    best_score = -np.inf
    best_model_path = os.path.join(models_dir, "tstr_synthetic_pooled_best.pt")

    step = 0
    episodes_count = 0
    start_time = time.time()

    while step < total_timesteps:
        obs, _ = train_env.reset()
        done = False

        while not done and step < total_timesteps:
            mask = train_env.action_masks()
            action = agent.select_action(obs, action_mask=mask, evaluate=False)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            agent.memory.push(obs, action, reward, next_obs, done)
            agent.update()

            obs = next_obs
            step += 1

            if step % eval_freq == 0:
                elapsed = time.time() - start_time
                val_metrics, _, _ = evaluate_tstr_dataset(
                    episodes_dict=val_real_episodes,
                    agent=agent,
                    window_size=window_size,
                    initial_capital=10_000.0,
                    title="Validation",
                )
                pf = val_metrics["Profit Factor"]
                ret = val_metrics["Total Return (%)"]
                trades_cnt = val_metrics["Total Trades"]
                score = (pf * 10.0) + ret if trades_cnt >= 20 else (pf - 10.0)

                print(
                    f"Step [{step:6d}/{total_timesteps:6d}] | "
                    f"Eps: {agent.epsilon:.3f} | "
                    f"Real Val PnL: ${val_metrics['Net P&L ($)']:+7.2f} | "
                    f"PF: {pf:4.2f} | "
                    f"WR: {val_metrics['Win Rate (%)']:4.1f}% | "
                    f"Trades: {trades_cnt:3d} | "
                    f"Time: {elapsed:4.0f}s"
                )

                if score > best_score:
                    best_score = score
                    agent.save(best_model_path)
                    print(f"  ⭐ Saved new best checkpoint (Real Val PF: {pf:.2f}, Ret: {ret:+.2f}%) to {best_model_path}")

        episodes_count += 1

    print(f"\n✅ Training completed in {time.time() - start_time:.1f}s ({episodes_count} episodes processed).")

    # Load best model
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        print(f"📦 Loaded best model from {best_model_path} for final Out-of-Sample evaluation.")

    # 4. Out-of-Sample Evaluation on Real Data (TSTR)
    print("\n" + "=" * 85)
    print("📈 OUT-OF-SAMPLE TSTR EVALUATION ON REAL HISTORICAL DATA")
    print("=" * 85)

    # Eval 1: Full 9-Asset Real Portfolio
    real_metrics, real_equity, real_trades = evaluate_tstr_dataset(
        episodes_dict=real_episodes,
        agent=agent,
        window_size=window_size,
        initial_capital=10_000.0,
        title="Multi-Asset Real Portfolio (9 Symbols)",
    )

    # Eval 2: Real XAUUSD Specialist
    xau_metrics, xau_equity, xau_trades = evaluate_tstr_dataset(
        episodes_dict=xauusd_real_episodes,
        agent=agent,
        window_size=window_size,
        initial_capital=10_000.0,
        title="XAUUSD Real Specialist",
    )

    # Print Summary Table
    summary_df = pd.DataFrame([real_metrics, xau_metrics])
    formatted_df = summary_df.copy()
    formatted_df["Net P&L ($)"] = formatted_df["Net P&L ($)"].apply(lambda x: f"${x:+,.2f}")
    formatted_df["Total Return (%)"] = formatted_df["Total Return (%)"].apply(lambda x: f"{x:+.2f}%")
    formatted_df["Win Rate (%)"] = formatted_df["Win Rate (%)"].apply(lambda x: f"{x:.1f}%")
    formatted_df["Profit Factor"] = formatted_df["Profit Factor"].apply(lambda x: f"{x:.2f}")
    formatted_df["Payoff Ratio (R)"] = formatted_df["Payoff Ratio (R)"].apply(lambda x: f"{x:.2f}")
    formatted_df["Avg Win ($)"] = formatted_df["Avg Win ($)"].apply(lambda x: f"${x:.2f}")
    formatted_df["Avg Loss ($)"] = formatted_df["Avg Loss ($)"].apply(lambda x: f"${x:.2f}")
    formatted_df["Max Drawdown (%)"] = formatted_df["Max Drawdown (%)"].apply(lambda x: f"{x:.2f}%")
    formatted_df["TP Rate (%)"] = formatted_df["TP Rate (%)"].apply(lambda x: f"{x:.1f}%")

    print(formatted_df[[
        "Evaluation", "Total Episodes", "Total Trades", "Net P&L ($)",
        "Total Return (%)", "Win Rate (%)", "Profit Factor", "Payoff Ratio (R)",
        "Avg Win ($)", "Avg Loss ($)", "Max Drawdown (%)", "TP Rate (%)"
    ]].to_string(index=False))

    # Per Symbol Breakdown Table
    if len(real_trades) > 0:
        df_t = pd.DataFrame(real_trades)
        sym_rows = []
        for sym, grp in df_t.groupby("symbol"):
            w = grp[grp["pnl_dollar"] > 0]["pnl_dollar"]
            l = grp[grp["pnl_dollar"] < 0]["pnl_dollar"]
            gw = w.sum() if len(w) > 0 else 0.0
            gl = abs(l.sum()) if len(l) > 0 else 1e-4
            net = gw - gl
            pf = gw / gl
            wr = len(w) / len(grp) * 100.0
            tp_cnt = (grp["reason"] == "TP").sum()
            sym_rows.append({
                "Symbol": sym,
                "Trades": len(grp),
                "Win Rate (%)": f"{wr:.1f}%",
                "Profit Factor": f"{pf:.2f}",
                "Gross Win ($)": f"${gw:+,.2f}",
                "Gross Loss ($)": f"${gl:,.2f}",
                "Net P&L ($)": f"${net:+,.2f}",
                "TP Rate (%)": f"{tp_cnt/len(grp)*100:.1f}%",
            })
        df_sym_summary = pd.DataFrame(sym_rows)
        print("\n📊 Per-Symbol Real Performance Breakdown:")
        print(df_sym_summary.to_string(index=False))

    # 5. Generate Performance Curves Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(real_equity.values, label=f"9-Asset Real Portfolio (PF: {real_metrics['Profit Factor']:.2f}, Ret: {real_metrics['Total Return (%)']:+.2f}%)", color="#00ff88", linewidth=1.8)
    ax1.plot(xau_equity.values, label=f"XAUUSD Real Specialist (PF: {xau_metrics['Profit Factor']:.2f}, Ret: {xau_metrics['Total Return (%)']:+.2f}%)", color="#ffaa00", linewidth=1.8)
    ax1.axhline(10_000.0, color="#888888", linestyle="--", alpha=0.6, label="Base Capital ($10,000)")
    ax1.set_title("TSTR Out-of-Sample Performance on Real Data (Trained on 668k Synthetic Continuity Bars)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Equity ($)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Trade PnL Distribution comparison
    real_pnls = [t["pnl_dollar"] for t in real_trades]
    xau_pnls = [t["pnl_dollar"] for t in xau_trades]
    ax2.hist(real_pnls, bins=50, alpha=0.6, color="#00d4ff", label=f"All Real Trades ({len(real_trades)} trades)")
    if len(xau_trades) > 0:
        ax2.hist(xau_pnls, bins=30, alpha=0.7, color="#ffaa00", label=f"XAUUSD Trades ({len(xau_trades)} trades)")
    ax2.axvline(0.0, color="#ffffff", linestyle="--", alpha=0.8)
    ax2.set_title("Real Trade P&L Distribution ($)", fontsize=12)
    ax2.set_xlabel("P&L per Trade ($)", fontsize=11)
    ax2.set_ylabel("Trade Frequency", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    chart_path = os.path.join(reports_dir, "tstr_synthetic_pooled_backtest.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print(f"\n📊 TSTR Evaluation chart saved to: {chart_path}")
    print("=" * 85)

    return summary_df


if __name__ == "__main__":
    train_and_evaluate_tstr(total_timesteps=40_000, eval_freq=10_000)
