"""
XAUUSD M15 Mean Reversion Trading Environment (With Action Masking)
-------------------------------------------------------------------
Custom Gymnasium environment for XAUUSD Forex/CFD trading with
hard signal gates (Swing sweeps + Extreme Z-scores + ADX regime),
realistic 0.10 lot sizing, ATR Stop Loss / Take Profit, and Differential Sharpe reward.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.strategies.mean_rev_reward import MeanReversionRewardCalculator


class XAUUSDMeanRevEnv(gym.Env):
    """
    Gymnasium Trading Environment for XAUUSD M15 Mean Reversion with Action Masking.

    Action Space:
        0: Flat / Close Position
        1: Long (Buy) — Gate: Z < -1.6, ADX < 30, RSI < 36, Sweep Low / Rejection
        2: Short (Sell) — Gate: Z > +1.6, ADX < 30, RSI > 64, Sweep High / Rejection

    Observation Space:
        Flattened array of [window_size x n_features] + [4 portfolio context features]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 12,
        max_steps_per_episode: int = 400,
        atr_sl_mult: float = 1.2,
        atr_tp_mult: float = 1.8,
        max_hold_bars: int = 20,
        min_hold_bars: int = 2,
        lot_size: float = 0.10,  # 0.10 lot = 10 oz ($10 per $1.00 gold move)
        initial_capital: float = 10_000.0,
        reward_calculator: MeanReversionRewardCalculator | None = None,
        is_eval: bool = False,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.max_steps = max_steps_per_episode
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_hold_bars = max_hold_bars
        self.min_hold_bars = min_hold_bars
        self.lot_size = lot_size
        self.initial_capital = initial_capital
        self.is_eval = is_eval

        # Separate raw columns from model observation features
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
        self.long_gate_np = df["long_gate_raw"].values.astype(np.float32)
        self.short_gate_np = df["short_gate_raw"].values.astype(np.float32)

        # Observation shape
        obs_dim = (self.window_size * self.n_features) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reward_calc = reward_calculator or MeanReversionRewardCalculator(
            spread_penalty_mult=1.5,
            adverse_penalty_scale=1.0,
            step_hold_penalty=0.001,
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
        """
        Action mask defining valid transitions at current bar:
        - Flat (0): Always valid.
        - Long (1): Valid if long_gate is active (or currently holding Long).
        - Short (2): Valid if short_gate is active (or currently holding Short).
        """
        can_long = bool(self.long_gate_np[self.current_step] > 0.5)
        can_short = bool(self.short_gate_np[self.current_step] > 0.5)

        if self.position == 0:
            return np.array([True, can_long, can_short], dtype=bool)
        elif self.position == 1:
            # In Long: can close (0), keep holding (1), or flip to Short if short_gate is active
            return np.array([True, True, can_short], dtype=bool)
        else:  # In Short
            # In Short: can close (0), flip to Long if long_gate is active, or keep holding (2)
            return np.array([True, can_long, True], dtype=bool)

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
        # Enforce Action Masking
        mask = self.action_masks()
        if not mask[action]:
            # Invalid action: stay flat or hold position
            action = 0 if self.position == 0 else (1 if self.position == 1 else 2)

        target_pos = 0 if action == 0 else (1 if action == 1 else -1)

        curr_close = self.close_np[self.current_step]
        curr_high = self.high_np[self.current_step]
        curr_low = self.low_np[self.current_step]
        curr_atr = max(self.atr_np[self.current_step], 1e-4)
        curr_spread = self.spread_np[self.current_step]

        realized_pnl_step = 0.0
        exit_reason = None

        # 1. SL / TP / Max Hold logic for existing position
        if self.position != 0:
            self.hold_bars += 1

            if self.position == 1:  # Long
                if curr_low <= self.sl_price:
                    realized_pnl_step = self.sl_price - self.entry_price
                    exit_reason = "SL"
                    target_pos = 0
                elif curr_high >= self.tp_price:
                    realized_pnl_step = self.tp_price - self.entry_price
                    exit_reason = "TP"
                    target_pos = 0
                elif self.hold_bars >= self.max_hold_bars:
                    realized_pnl_step = curr_close - self.entry_price
                    exit_reason = "MAX_HOLD"
                    target_pos = 0

            elif self.position == -1:  # Short
                if curr_high >= self.sl_price:
                    realized_pnl_step = self.entry_price - self.sl_price
                    exit_reason = "SL"
                    target_pos = 0
                elif curr_low <= self.tp_price:
                    realized_pnl_step = self.entry_price - self.tp_price
                    exit_reason = "TP"
                    target_pos = 0
                elif self.hold_bars >= self.max_hold_bars:
                    realized_pnl_step = self.entry_price - curr_close
                    exit_reason = "MAX_HOLD"
                    target_pos = 0

        # Prevent early close before min_hold_bars unless SL/TP hit
        if self.position != 0 and exit_reason is None and target_pos == 0:
            if self.hold_bars < self.min_hold_bars:
                target_pos = self.position  # keep holding

        # 2. Position Transitions
        pos_change = abs(target_pos - self.position)

        if pos_change > 0:
            # If closing manually without SL/TP
            if self.position != 0 and exit_reason is None:
                if self.position == 1:
                    realized_pnl_step = curr_close - self.entry_price
                else:
                    realized_pnl_step = self.entry_price - curr_close
                exit_reason = "MANUAL_CLOSE"

            if exit_reason is not None:
                dollar_pnl = realized_pnl_step * (self.lot_size * 100.0)  # 0.1 lot = $10/pt
                self.trades_history.append(
                    {
                        "step": self.current_step,
                        "position": self.position,
                        "entry": self.entry_price,
                        "exit": curr_close,
                        "pnl_price": realized_pnl_step,
                        "pnl_dollar": dollar_pnl,
                        "pnl_atr": realized_pnl_step / curr_atr,
                        "hold_bars": self.hold_bars,
                        "reason": exit_reason,
                    }
                )
                self.realized_pnl_total += dollar_pnl
                self.position = 0
                self.hold_bars = 0
                self.prev_unrealized_pnl = 0.0

            # Open new position
            if target_pos != 0:
                self.position = target_pos
                self.hold_bars = 0
                if target_pos == 1:  # Long (Ask)
                    self.entry_price = curr_close + 0.5 * curr_spread
                    self.sl_price = self.entry_price - (self.atr_sl_mult * curr_atr)
                    self.tp_price = self.entry_price + (self.atr_tp_mult * curr_atr)
                else:  # Short (Bid)
                    self.entry_price = curr_close - 0.5 * curr_spread
                    self.sl_price = self.entry_price + (self.atr_sl_mult * curr_atr)
                    self.tp_price = self.entry_price - (self.atr_tp_mult * curr_atr)

        # 3. Compute Unrealized PnL
        unrealized_pnl_price = 0.0
        if self.position == 1:
            unrealized_pnl_price = curr_close - self.entry_price
        elif self.position == -1:
            unrealized_pnl_price = self.entry_price - curr_close

        unrealized_dollar = unrealized_pnl_price * (self.lot_size * 100.0)
        delta_unrealized = unrealized_pnl_price - self.prev_unrealized_pnl
        self.prev_unrealized_pnl = unrealized_pnl_price

        delta_pnl_total = realized_pnl_step + delta_unrealized

        # 4. Calculate Shaped Reward
        reward, rew_info = self.reward_calc.calculate(
            delta_pnl_raw=delta_pnl_total,
            atr_raw=curr_atr,
            position_change=pos_change,
            spread_raw=curr_spread,
            unrealized_pnl_raw=unrealized_pnl_price,
            is_holding=(self.position != 0),
        )

        # Advance step
        self.current_step += 1
        self.step_count += 1

        terminated = self.step_count >= self.episode_length or self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "step": self.current_step,
            "realized_pnl_step": realized_pnl_step,
            "realized_pnl_total": self.realized_pnl_total,
            "unrealized_dollar": unrealized_dollar,
            "position": self.position,
            "pos_change": pos_change,
            "reward_breakdown": rew_info,
            "trades_count": len(self.trades_history),
            "action_mask": self.action_masks(),
        }

        return obs, reward, terminated, truncated, info
