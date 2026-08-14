"""
Mean Reversion Reward Shaping Module
------------------------------------
Implements Differential Sharpe Ratio, Transaction Cost Friction,
and Adverse Excursion (Drawdown) Penalty for XAUUSD M15 DRL Trading.
"""
from __future__ import annotations
import numpy as np


class DifferentialSharpeReward:
    """
    Online Differential Sharpe Ratio (Moody & Saffell, 2001) calculator.
    Provides step-by-step reward that directly maximizes the portfolio Sharpe Ratio.
    """

    def __init__(self, decay: float = 0.05, eps: float = 1e-6):
        self.decay = decay  # Exponential decay rate (eta)
        self.eps = eps
        self.a = 0.0  # 1st moment estimate (E[R])
        self.b = 0.0  # 2nd moment estimate (E[R^2])
        self.initialized = False

    def reset(self):
        self.a = 0.0
        self.b = 0.0
        self.initialized = False

    def step(self, step_return: float) -> float:
        """
        Calculate the differential change in Sharpe Ratio given a step return.
        """
        r = float(step_return)
        if not self.initialized:
            self.a = r
            self.b = r ** 2
            self.initialized = True
            return 0.0

        delta_a = r - self.a
        delta_b = (r ** 2) - self.b

        var = self.b - (self.a ** 2)
        if var < self.eps:
            var = self.eps

        # Differential Sharpe Ratio formula
        diff_sharpe = (self.b * delta_a - 0.5 * self.a * delta_b) / (var ** 1.5 + self.eps)

        # Update running moments
        self.a += self.decay * delta_a
        self.b += self.decay * delta_b

        # Clip reward for numerical stability in RL training
        return float(np.clip(diff_sharpe, -5.0, 5.0))


class MeanReversionRewardCalculator:
    """
    Composite reward calculator combining:
    1. Incremental PnL (normalized by ATR)
    2. Differential Sharpe component
    3. Transaction cost / Spread friction
    4. Adverse Excursion (Drawdown breakout) penalty
    5. Holding time friction
    """

    def __init__(
        self,
        use_differential_sharpe: bool = True,
        sharpe_weight: float = 0.5,
        pnl_weight: float = 1.0,
        spread_penalty_mult: float = 1.0,
        adverse_mdd_threshold_atr: float = 1.5,
        adverse_penalty_scale: float = 0.5,
        step_hold_penalty: float = 0.002,
    ):
        self.use_differential_sharpe = use_differential_sharpe
        self.sharpe_weight = sharpe_weight
        self.pnl_weight = pnl_weight
        self.spread_penalty_mult = spread_penalty_mult
        self.adverse_mdd_threshold_atr = adverse_mdd_threshold_atr
        self.adverse_penalty_scale = adverse_penalty_scale
        self.step_hold_penalty = step_hold_penalty

        self.diff_sharpe = DifferentialSharpeReward(decay=0.02)

    def reset(self):
        self.diff_sharpe.reset()

    def calculate(
        self,
        delta_pnl_raw: float,
        atr_raw: float,
        position_change: float,
        spread_raw: float,
        unrealized_pnl_raw: float,
        is_holding: bool,
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate total step reward and breakdown.
        """
        atr = max(atr_raw, 1e-4)

        # 1. Normalized step PnL change
        delta_pnl_atr = delta_pnl_raw / atr

        # 2. Transaction cost penalty (Spread on entry/reversal)
        cost_penalty = 0.0
        if position_change > 0.0:
            turnover_cost = position_change * (spread_raw * self.spread_penalty_mult)
            cost_penalty = -(turnover_cost / atr)

        # 3. Adverse Excursion / Runaway Drawdown Penalty
        # If holding a losing position that moves against us by more than threshold ATR
        adverse_penalty = 0.0
        unrealized_loss_atr = -unrealized_pnl_raw / atr
        if unrealized_loss_atr > self.adverse_mdd_threshold_atr:
            excess_loss = unrealized_loss_atr - self.adverse_mdd_threshold_atr
            adverse_penalty = -self.adverse_penalty_scale * (excess_loss ** 1.5)

        # 4. Holding time penalty (discourage idle holding)
        hold_penalty = -self.step_hold_penalty if is_holding else 0.0

        # 5. Differential Sharpe
        sharpe_rew = 0.0
        if self.use_differential_sharpe:
            sharpe_rew = self.diff_sharpe.step(delta_pnl_atr)

        total_reward = (
            self.pnl_weight * delta_pnl_atr
            + self.sharpe_weight * sharpe_rew
            + cost_penalty
            + adverse_penalty
            + hold_penalty
        )

        breakdown = {
            "delta_pnl_atr": delta_pnl_atr,
            "sharpe_reward": sharpe_rew,
            "cost_penalty": cost_penalty,
            "adverse_penalty": adverse_penalty,
            "hold_penalty": hold_penalty,
            "total_reward": float(np.clip(total_reward, -10.0, 10.0)),
        }

        return breakdown["total_reward"], breakdown
