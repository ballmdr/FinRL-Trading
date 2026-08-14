"""
Multi-Asset Multi-Episode Gymnasium Trading Environment (FinRL-Trading)
-----------------------------------------------------------------------
Handles episode-based training across 9 symbols and 5 market stress regimes:
1. Samples random episodes during training (1,152 synthetic episodes).
2. Deterministic sequential replay during validation / Out-of-Sample testing on real data.
3. Fixed Target R:R Execution (1.8x ATR Take Profit / 1.0x ATR Stop Loss).
4. Exact incremental PnL accounting and round-trip spread deduction.
5. Symbol-aware lot sizing and pip valuations for Gold, Forex, and Crypto.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.strategies.mean_rev_reward import MeanReversionRewardCalculator


# Symbol lot specifications ($ value per 1 unit price move per 0.10 standard lot)
SYMBOL_VALUE_PER_POINT = {
    "XAUUSD": 10.0,      # 0.10 lot Gold: $1.00 move = $10.00
    "BTCUSD": 0.10,      # 0.10 lot BTC: $1.00 move = $0.10 ($100 move = $10.00)
    "EURUSD": 1000.0,    # 0.10 lot EURUSD: 0.0001 (1 pip) = $1.00 (multiplier = 10,000 for standard lot, 1000 for 0.10)
    "GBPUSD": 1000.0,
    "AUDUSD": 1000.0,
    "NZDUSD": 1000.0,
    "USDCAD": 1000.0,
    "USDCHF": 1000.0,
    "USDJPY": 1000.0,    # 0.10 lot JPY: 0.01 = $1.00 approx
}


def compute_fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute vector ATR for an episode array."""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = max(high[0] - low[0], 1e-4)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
            1e-4,
        )
    # Exponential smoothing
    alpha = 1.0 / period
    atr = np.zeros(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (alpha * tr[i]) + ((1.0 - alpha) * atr[i - 1])
    return atr


class PooledEpisodeEnv(gym.Env):
    """
    Episode-based Gymnasium Trading Environment for Multi-Asset Synthetic & Real Datasets.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        episodes_dict: dict[str, dict[str, np.ndarray]],
        window_size: int = 8,
        atr_sl_mult: float = 1.0,
        atr_tp_mult: float = 1.8,
        max_hold_bars: int = 36,
        initial_capital: float = 10_000.0,
        reward_calculator: MeanReversionRewardCalculator | None = None,
        is_eval: bool = False,
        smc_gating: bool = True,
    ):
        super().__init__()
        self.episodes = episodes_dict
        self.episode_keys = list(episodes_dict.keys())
        self.window_size = window_size
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_hold_bars = max_hold_bars
        self.initial_capital = initial_capital
        self.is_eval = is_eval
        self.smc_gating = smc_gating

        # Extract feature dimensions from first episode
        sample_ep = self.episodes[self.episode_keys[0]]
        self.n_features = sample_ep["features"].shape[1]

        # Observation shape: (window_size * n_features) + 4 portfolio context
        obs_dim = (self.window_size * self.n_features) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0: Flat, 1: Long, 2: Short

        self.reward_calc = reward_calculator or MeanReversionRewardCalculator(
            spread_penalty_mult=1.0,
            adverse_penalty_scale=0.8,
            step_hold_penalty=0.0,
        )

        self.np_random = np.random.RandomState()
        self.eval_ep_idx = 0
        self._current_ep_key = self.episode_keys[0]

        # Financial tracking
        self.realized_pnl_total = 0.0
        self.all_trades_history = []

    def _select_episode(self) -> dict[str, np.ndarray]:
        if self.is_eval:
            self._current_ep_key = self.episode_keys[self.eval_ep_idx % len(self.episode_keys)]
            self.eval_ep_idx += 1
        else:
            self._current_ep_key = self.np_random.choice(self.episode_keys)

        return self.episodes[self._current_ep_key]

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        self.curr_ep = self._select_episode()
        self.symbol = self.curr_ep["symbol"]
        self.point_value = SYMBOL_VALUE_PER_POINT.get(self.symbol, 1000.0)

        self.features_np = self.curr_ep["features"]
        self.open_np = self.curr_ep["open"]
        self.high_np = self.curr_ep["high"]
        self.low_np = self.curr_ep["low"]
        self.close_np = self.curr_ep["close"]
        self.spread_np = self.curr_ep["spread"]
        self.ep_len = len(self.close_np)

        # Precompute ATR for current episode
        self.atr_np = compute_fast_atr(self.high_np, self.low_np, self.close_np, period=14)

        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.hold_bars = 0
        self.prev_unrealized_pnl = 0.0
        self.ep_trades = []
        self.reward_calc.reset()

        obs = self._get_obs()
        info = {
            "episode_id": self._current_ep_key,
            "symbol": self.symbol,
            "action_mask": self.action_masks(),
        }
        return obs, info

    def action_masks(self) -> np.ndarray:
        """When flat, action mask enforces SMC setup gates; when in position, hold (Force Hold)."""
        if self.position == 1:
            return np.array([False, True, False], dtype=bool)
        elif self.position == -1:
            return np.array([False, False, True], dtype=bool)

        if not self.smc_gating:
            return np.array([True, True, True], dtype=bool)

        curr_feats = self.features_np[self.current_step]
        fvg = curr_feats[14]
        b_ob = curr_feats[15]
        s_ob = curr_feats[16]
        liq = curr_feats[18]

        can_long = (b_ob > 0.0) or (fvg > 0.0) or (liq > 0.3)
        can_short = (s_ob > 0.0) or (fvg < 0.0) or (liq < -0.3)

        return np.array([True, bool(can_long), bool(can_short)], dtype=bool)

    def _get_obs(self) -> np.ndarray:
        window_feats = self.features_np[
            self.current_step - self.window_size : self.current_step
        ].flatten()

        p = self.close_np[self.current_step]
        atr = max(self.atr_np[self.current_step], 1e-5)
        spread = self.spread_np[self.current_step]

        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = p - self.entry_price
        elif self.position == -1:
            unrealized_pnl = self.entry_price - p

        portfolio_ctx = np.array(
            [
                float(self.position),
                float(np.clip(unrealized_pnl / atr, -5.0, 5.0)),
                float(np.clip(self.hold_bars / self.max_hold_bars, 0.0, 1.0)),
                float(np.clip(spread / atr, 0.0, 5.0)),
            ],
            dtype=np.float32,
        )

        return np.concatenate([window_feats, portfolio_ctx]).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        curr_close = self.close_np[self.current_step]
        curr_high = self.high_np[self.current_step]
        curr_low = self.low_np[self.current_step]
        curr_atr = max(self.atr_np[self.current_step], 1e-5)
        curr_spread = max(self.spread_np[self.current_step], 1e-5)

        exit_reason = None
        pos_change = 0.0
        delta_pnl_price = 0.0

        # 1. Existing Position Management (Fixed Target 1.8x TP / 1.0x SL)
        if self.position != 0:
            self.hold_bars += 1

            if self.position == 1:  # Long
                if curr_high >= self.tp_price:
                    exit_price = self.tp_price - (0.5 * curr_spread)
                    trade_realized_pnl = exit_price - self.entry_price
                    exit_reason = "TP"
                elif curr_low <= self.sl_price:
                    exit_price = self.sl_price - (0.5 * curr_spread)
                    trade_realized_pnl = exit_price - self.entry_price
                    exit_reason = "SL"
                elif self.hold_bars >= self.max_hold_bars:
                    exit_price = curr_close - (0.5 * curr_spread)
                    trade_realized_pnl = exit_price - self.entry_price
                    exit_reason = "MAX_HOLD"

            elif self.position == -1:  # Short
                if curr_low <= self.tp_price:
                    exit_price = self.tp_price + (0.5 * curr_spread)
                    trade_realized_pnl = self.entry_price - exit_price
                    exit_reason = "TP"
                elif curr_high >= self.sl_price:
                    exit_price = self.sl_price + (0.5 * curr_spread)
                    trade_realized_pnl = self.entry_price - exit_price
                    exit_reason = "SL"
                elif self.hold_bars >= self.max_hold_bars:
                    exit_price = curr_close + (0.5 * curr_spread)
                    trade_realized_pnl = self.entry_price - exit_price
                    exit_reason = "MAX_HOLD"

            # Process exit
            if exit_reason is not None:
                delta_pnl_price = trade_realized_pnl - self.prev_unrealized_pnl
                realized_dollar = trade_realized_pnl * self.point_value

                trade_record = {
                    "episode_id": self._current_ep_key,
                    "symbol": self.symbol,
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
                self.ep_trades.append(trade_record)
                self.all_trades_history.append(trade_record)
                self.realized_pnl_total += realized_dollar

                self.position = 0
                self.hold_bars = 0
                self.prev_unrealized_pnl = 0.0
                pos_change = 1.0
                unrealized_pnl_price = 0.0
            else:
                # Still holding: Incremental unrealized price change
                if self.position == 1:
                    unrealized_pnl_price = (curr_close - 0.5 * curr_spread) - self.entry_price
                else:
                    unrealized_pnl_price = self.entry_price - (curr_close + 0.5 * curr_spread)

                delta_pnl_price = unrealized_pnl_price - self.prev_unrealized_pnl
                self.prev_unrealized_pnl = unrealized_pnl_price

        # 2. New Entry Execution (Only when Flat)
        elif self.position == 0:
            if action in (1, 2):
                target_pos = 1 if action == 1 else -1
                self.position = target_pos
                self.hold_bars = 0
                pos_change = 1.0

                if target_pos == 1:  # Long
                    self.entry_price = curr_close + (0.5 * curr_spread)
                    self.sl_price = self.entry_price - (self.atr_sl_mult * curr_atr)
                    self.tp_price = self.entry_price + (self.atr_tp_mult * curr_atr)
                    unrealized_pnl_price = -curr_spread
                else:  # Short
                    self.entry_price = curr_close - (0.5 * curr_spread)
                    self.sl_price = self.entry_price + (self.atr_sl_mult * curr_atr)
                    self.tp_price = self.entry_price - (self.atr_tp_mult * curr_atr)
                    unrealized_pnl_price = -curr_spread

                delta_pnl_price = unrealized_pnl_price
                self.prev_unrealized_pnl = unrealized_pnl_price
            else:
                unrealized_pnl_price = 0.0
                delta_pnl_price = 0.0
                self.prev_unrealized_pnl = 0.0

        # 3. Calculate Reward
        reward, rew_info = self.reward_calc.calculate(
            delta_pnl_raw=delta_pnl_price,
            atr_raw=curr_atr,
            position_change=pos_change,
            spread_raw=curr_spread,
            unrealized_pnl_raw=unrealized_pnl_price,
            is_holding=(self.position != 0),
        )

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.ep_len - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "episode_id": self._current_ep_key,
            "symbol": self.symbol,
            "realized_pnl_total": self.realized_pnl_total,
            "trades_count": len(self.ep_trades),
            "action_mask": self.action_masks(),
        }

        return obs, reward, terminated, truncated, info
