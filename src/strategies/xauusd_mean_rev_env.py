"""
XAUUSD M15 Mean Reversion Trading Environment (Multi-Arm Exit Policies)
-----------------------------------------------------------------------
Custom Gymnasium environment for XAUUSD Forex/CFD supporting:
1. Strict Action Masking at entry (Liquidity Sweep + Session Gating).
2. Modular Exit Policies:
   - 'be_then_trail': Breakeven at +1.0x ATR, Trail at +1.4x ATR (0.5x trail dist), TP 1.8x ATR.
   - 'mean_target': Exit on reversion to Kalman Dynamic Mean (kf_price) / BB Midline or Opposite Sweep.
   - 'agent_after_1atr': Force-hold until profit >= +1.0x ATR, then allow agent discretionary exit.
   - 'fixed_tp_sl': Pure TP (1.8x ATR) / SL (1.0x ATR) without early scratch.
3. Exact step-by-step incremental PnL accounting (zero double counting).
4. Full round-trip spread deduction on real $10,000 equity (0.10 lot sizing).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.strategies.mean_rev_reward import MeanReversionRewardCalculator


class XAUUSDMeanRevEnv(gym.Env):
    """
    Gymnasium Trading Environment for XAUUSD M15 Mean Reversion with Modular Exit Policies.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 12,
        max_steps_per_episode: int = 400,
        exit_policy: str = "be_then_trail",  # 'be_then_trail' | 'mean_target' | 'agent_after_1atr' | 'fixed_tp_sl'
        atr_sl_mult: float = 1.0,            # Tight SL at 1.0x ATR
        atr_tp_mult: float = 1.8,            # Target TP at 1.8x ATR
        be_trigger_atr: float = 1.0,         # Lock Breakeven (+0.1x ATR) when profit >= 1.0x ATR
        trail_trigger_atr: float = 1.4,      # Start trailing only when profit >= 1.4x ATR
        trail_dist_atr: float = 0.5,         # Trail distance behind peak price
        max_hold_bars: int = 32,             # Max 8 hours hold
        lot_size: float = 0.10,              # 0.10 lot = $10 per $1.00 Gold move
        initial_capital: float = 10_000.0,
        reward_calculator: MeanReversionRewardCalculator | None = None,
        is_eval: bool = False,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.max_steps = max_steps_per_episode
        self.exit_policy = exit_policy
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.be_trigger_atr = be_trigger_atr
        self.trail_trigger_atr = trail_trigger_atr
        self.trail_dist_atr = trail_dist_atr
        self.max_hold_bars = max_hold_bars
        self.lot_size = lot_size
        self.initial_capital = initial_capital
        self.is_eval = is_eval

        # Separate raw price columns from model observation features
        self.raw_cols = {
            "open_raw", "high_raw", "low_raw", "close_raw",
            "atr_raw", "spread_raw", "kf_price_raw",
            "long_gate_raw", "short_gate_raw"
        }
        self.feature_cols = [c for c in df.columns if c not in self.raw_cols]
        self.n_features = len(self.feature_cols)

        # Fast numpy arrays
        self.features_np = df[self.feature_cols].values.astype(np.float32)
        self.open_np = df["open_raw"].values.astype(np.float64)
        self.high_np = df["high_raw"].values.astype(np.float64)
        self.low_np = df["low_raw"].values.astype(np.float64)
        self.close_np = df["close_raw"].values.astype(np.float64)
        self.atr_np = df["atr_raw"].values.astype(np.float64)
        self.spread_np = df["spread_raw"].values.astype(np.float64)
        self.kf_price_np = df["kf_price_raw"].values.astype(np.float64)
        self.long_gate_np = df["long_gate_raw"].values.astype(np.float32)
        self.short_gate_np = df["short_gate_raw"].values.astype(np.float32)

        # Observation shape
        obs_dim = (self.window_size * self.n_features) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reward_calc = reward_calculator or MeanReversionRewardCalculator(
            spread_penalty_mult=1.0,
            adverse_penalty_scale=0.8,
            step_hold_penalty=0.0,
        )
        self.np_random = np.random.RandomState()
        self._reset_state()

    def _reset_state(self):
        if self.is_eval:
            self.start_step = self.window_size
            self.episode_length = len(self.df) - self.window_size - 1
        else:
            max_start = len(self.df) - self.max_steps - 2
            if max_start > self.window_size:
                self.start_step = self.np_random.randint(self.window_size, max_start)
            else:
                self.start_step = self.window_size
            self.episode_length = min(self.max_steps, len(self.df) - self.start_step - 1)

        self.current_step = self.start_step
        self.step_count = 0

        # Position tracking: 0 = Flat, 1 = Long, -1 = Short
        self.position = 0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.peak_price = 0.0
        self.be_active = False
        self.trailing_active = False
        self.hold_bars = 0
        self.prev_unrealized_pnl = 0.0

        # Financial tracking
        self.realized_pnl_total = 0.0
        self.trades_history = []
        self.reward_calc.reset()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        self._reset_state()
        return self._get_obs(), {"action_mask": self.action_masks()}

    def action_masks(self) -> np.ndarray:
        """Action mask defining valid transitions at current bar."""
        can_long = bool(self.long_gate_np[self.current_step] > 0.5)
        can_short = bool(self.short_gate_np[self.current_step] > 0.5)

        if self.position == 0:
            return np.array([True, can_long, can_short], dtype=bool)
        elif self.position == 1:
            if self.exit_policy == "agent_after_1atr":
                # Can exit only after profit >= 1.0x ATR
                p = self.close_np[self.current_step]
                atr = max(self.atr_np[self.current_step], 1e-4)
                can_exit = bool((p - self.entry_price) >= 1.0 * atr)
                return np.array([can_exit, True, False], dtype=bool)
            else:
                # Force-hold
                return np.array([False, True, False], dtype=bool)
        else:  # In Short
            if self.exit_policy == "agent_after_1atr":
                p = self.close_np[self.current_step]
                atr = max(self.atr_np[self.current_step], 1e-4)
                can_exit = bool((self.entry_price - p) >= 1.0 * atr)
                return np.array([can_exit, False, True], dtype=bool)
            else:
                return np.array([False, False, True], dtype=bool)

    def _get_obs(self) -> np.ndarray:
        window_feats = self.features_np[
            self.current_step - self.window_size : self.current_step
        ].flatten()

        p = self.close_np[self.current_step]
        atr = max(self.atr_np[self.current_step], 1e-4)
        spread = self.spread_np[self.current_step]

        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = p - self.entry_price
        elif self.position == -1:
            unrealized_pnl = self.entry_price - p

        portfolio_ctx = np.array(
            [
                float(self.position),
                float(unrealized_pnl / atr),
                float(self.hold_bars / self.max_hold_bars),
                float(spread / atr),
            ],
            dtype=np.float32,
        )

        return np.concatenate([window_feats, portfolio_ctx]).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        curr_close = self.close_np[self.current_step]
        curr_high = self.high_np[self.current_step]
        curr_low = self.low_np[self.current_step]
        curr_atr = max(self.atr_np[self.current_step], 1e-4)
        curr_spread = self.spread_np[self.current_step]
        curr_kf = self.kf_price_np[self.current_step]

        realized_pnl_step = 0.0
        exit_reason = None
        pos_change = 0.0
        delta_pnl_total = 0.0

        # 1. Existing Position Management
        if self.position != 0:
            self.hold_bars += 1

            if self.position == 1:  # Long
                self.peak_price = max(self.peak_price, curr_high)
                unrealized_profit_atr = (self.peak_price - self.entry_price) / curr_atr

                # Policy: BE-then-Trail
                if self.exit_policy == "be_then_trail":
                    # Lock Breakeven (+0.1x ATR) at +1.0x ATR profit
                    if not self.be_active and unrealized_profit_atr >= self.be_trigger_atr:
                        self.be_active = True
                        self.sl_price = max(self.sl_price, self.entry_price + 0.1 * curr_atr)

                    # Start trailing at +1.4x ATR profit
                    if not self.trailing_active and unrealized_profit_atr >= self.trail_trigger_atr:
                        self.trailing_active = True

                    if self.trailing_active:
                        new_sl = self.peak_price - (self.trail_dist_atr * curr_atr)
                        self.sl_price = max(self.sl_price, new_sl)

                # Policy: Dynamic Mean Target Exit
                elif self.exit_policy == "mean_target":
                    # Exit when price crosses back above Kalman Mean or reaches TP
                    if curr_close >= curr_kf and (curr_close - self.entry_price) >= 0.5 * curr_atr:
                        exit_price = curr_close - 0.5 * curr_spread
                        trade_realized_pnl = exit_price - self.entry_price
                        exit_reason = "MEAN_REVERTED"

                # Check Exits across all policies
                if exit_reason is None:
                    if curr_high >= self.tp_price:
                        exit_price = self.tp_price - 0.5 * curr_spread
                        trade_realized_pnl = exit_price - self.entry_price
                        exit_reason = "TP"
                    elif curr_low <= self.sl_price:
                        exit_price = self.sl_price - 0.5 * curr_spread
                        trade_realized_pnl = exit_price - self.entry_price
                        exit_reason = "TRAILING_STOP" if self.trailing_active else ("BE_STOP" if self.be_active else "SL")
                    elif self.exit_policy == "agent_after_1atr" and action == 0 and (curr_close - self.entry_price) >= 1.0 * curr_atr:
                        exit_price = curr_close - 0.5 * curr_spread
                        trade_realized_pnl = exit_price - self.entry_price
                        exit_reason = "AGENT_EXIT"
                    elif self.hold_bars >= self.max_hold_bars:
                        exit_price = curr_close - 0.5 * curr_spread
                        trade_realized_pnl = exit_price - self.entry_price
                        exit_reason = "MAX_HOLD"

            elif self.position == -1:  # Short
                self.peak_price = min(self.peak_price, curr_low)
                unrealized_profit_atr = (self.entry_price - self.peak_price) / curr_atr

                # Policy: BE-then-Trail
                if self.exit_policy == "be_then_trail":
                    if not self.be_active and unrealized_profit_atr >= self.be_trigger_atr:
                        self.be_active = True
                        self.sl_price = min(self.sl_price, self.entry_price - 0.1 * curr_atr)

                    if not self.trailing_active and unrealized_profit_atr >= self.trail_trigger_atr:
                        self.trailing_active = True

                    if self.trailing_active:
                        new_sl = self.peak_price + (self.trail_dist_atr * curr_atr)
                        self.sl_price = min(self.sl_price, new_sl)

                # Policy: Dynamic Mean Target Exit
                elif self.exit_policy == "mean_target":
                    if curr_close <= curr_kf and (self.entry_price - curr_close) >= 0.5 * curr_atr:
                        exit_price = curr_close + 0.5 * curr_spread
                        trade_realized_pnl = self.entry_price - exit_price
                        exit_reason = "MEAN_REVERTED"

                # Check Exits across all policies
                if exit_reason is None:
                    if curr_low <= self.tp_price:
                        exit_price = self.tp_price + 0.5 * curr_spread
                        trade_realized_pnl = self.entry_price - exit_price
                        exit_reason = "TP"
                    elif curr_high >= self.sl_price:
                        exit_price = self.sl_price + 0.5 * curr_spread
                        trade_realized_pnl = self.entry_price - exit_price
                        exit_reason = "TRAILING_STOP" if self.trailing_active else ("BE_STOP" if self.be_active else "SL")
                    elif self.exit_policy == "agent_after_1atr" and action == 0 and (self.entry_price - curr_close) >= 1.0 * curr_atr:
                        exit_price = curr_close + 0.5 * curr_spread
                        trade_realized_pnl = self.entry_price - exit_price
                        exit_reason = "AGENT_EXIT"
                    elif self.hold_bars >= self.max_hold_bars:
                        exit_price = curr_close + 0.5 * curr_spread
                        trade_realized_pnl = self.entry_price - exit_price
                        exit_reason = "MAX_HOLD"

            # 2. Process Exit Execution & Exact Incremental PnL
            if exit_reason is not None:
                delta_pnl_total = trade_realized_pnl - self.prev_unrealized_pnl
                realized_dollar = trade_realized_pnl * (self.lot_size * 100.0)

                self.trades_history.append(
                    {
                        "step": self.current_step,
                        "position": self.position,
                        "entry": self.entry_price,
                        "exit": exit_price,
                        "pnl_price": trade_realized_pnl,
                        "pnl_dollar": realized_dollar,
                        "pnl_atr": trade_realized_pnl / curr_atr,
                        "hold_bars": self.hold_bars,
                        "reason": exit_reason,
                    }
                )

                self.realized_pnl_total += realized_dollar
                self.position = 0
                self.hold_bars = 0
                self.be_active = False
                self.trailing_active = False
                self.prev_unrealized_pnl = 0.0
                pos_change = 1.0
                unrealized_pnl_price = 0.0
            else:
                if self.position == 1:
                    unrealized_pnl_price = (curr_close - 0.5 * curr_spread) - self.entry_price
                else:
                    unrealized_pnl_price = self.entry_price - (curr_close + 0.5 * curr_spread)

                delta_pnl_total = unrealized_pnl_price - self.prev_unrealized_pnl
                self.prev_unrealized_pnl = unrealized_pnl_price

        # 3. New Entry Execution (Only when Flat)
        elif self.position == 0:
            mask = self.action_masks()
            target_pos = 0
            if action == 1 and mask[1]:
                target_pos = 1
            elif action == 2 and mask[2]:
                target_pos = -1

            if target_pos != 0:
                self.position = target_pos
                self.hold_bars = 0
                self.be_active = False
                self.trailing_active = False
                pos_change = 1.0

                if target_pos == 1:  # Long
                    self.entry_price = curr_close + 0.5 * curr_spread
                    self.peak_price = curr_high
                    self.sl_price = self.entry_price - (self.atr_sl_mult * curr_atr)
                    self.tp_price = self.entry_price + (self.atr_tp_mult * curr_atr)
                    unrealized_pnl_price = -curr_spread
                else:  # Short
                    self.entry_price = curr_close - 0.5 * curr_spread
                    self.peak_price = curr_low
                    self.sl_price = self.entry_price + (self.atr_sl_mult * curr_atr)
                    self.tp_price = self.entry_price - (self.atr_tp_mult * curr_atr)
                    unrealized_pnl_price = -curr_spread

                delta_pnl_total = unrealized_pnl_price
                self.prev_unrealized_pnl = unrealized_pnl_price
            else:
                unrealized_pnl_price = 0.0
                delta_pnl_total = 0.0
                self.prev_unrealized_pnl = 0.0

        # 4. Calculate Shaped Reward
        reward, rew_info = self.reward_calc.calculate(
            delta_pnl_raw=delta_pnl_total,
            atr_raw=curr_atr,
            position_change=pos_change,
            spread_raw=curr_spread,
            unrealized_pnl_raw=unrealized_pnl_price,
            is_holding=(self.position != 0),
        )

        self.current_step += 1
        self.step_count += 1

        terminated = self.step_count >= self.episode_length or self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        unrealized_dollar = unrealized_pnl_price * (self.lot_size * 100.0)

        info = {
            "step": self.current_step,
            "realized_pnl_total": self.realized_pnl_total,
            "unrealized_dollar": unrealized_dollar,
            "position": self.position,
            "reward_breakdown": rew_info,
            "trades_count": len(self.trades_history),
            "action_mask": self.action_masks(),
        }

        return obs, reward, terminated, truncated, info
