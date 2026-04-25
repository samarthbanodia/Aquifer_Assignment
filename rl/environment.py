"""
Gymnasium environment for single-pair spread trading.

State  (8 floats): z-score, z-velocity, z-acceleration, VIX, correlation,
                   current position, holding duration, unrealized P&L
Action (discrete 3): 0=FLAT  1=LONG_SPREAD  2=SHORT_SPREAD
Reward: daily P&L in dollars (transaction costs deducted on every trade)

One episode = one pair's preprocessed daily history.
Agent learns WHEN to enter/exit, not how much (fixed notional).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Tuple, Optional


class PairSpreadEnv(gym.Env):

    metadata = {"render_modes": []}

    # action → internal position signal
    _ACTION_MAP = {0: 0, 1: 1, 2: -1}

    def __init__(
        self,
        episode_data: pd.DataFrame,
        notional: float = 10_000,
        one_way_cost_pct: float = 0.0005,   # 0.05% each way = 0.10% round-trip
        max_holding_days: int = 60,
    ):
        super().__init__()
        self.data = episode_data.reset_index(drop=True)
        self.notional = notional
        self.cost = one_way_cost_pct * notional
        self.max_holding = max_holding_days
        self.n = len(self.data)

        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
        self._reset_internals()

    # ── internals ──────────────────────────────────────────────────────────

    def _reset_internals(self):
        self.t = 0
        self.position = 0        # -1, 0, +1
        self.holding_days = 0
        self.unrealized_pnl = 0.0
        self.daily_pnls: list[float] = []

    def _obs(self) -> np.ndarray:
        if self.t >= self.n:
            return np.zeros(8, dtype=np.float32)
        row = self.data.iloc[self.t]
        obs = np.array([
            np.clip(row.get("zscore",    0.0),          -5,  5),
            np.clip(row.get("z_vel",     0.0),          -3,  3),
            np.clip(row.get("z_accel",   0.0),          -2,  2),
            np.clip(row.get("vix",      20.0) / 30.0,   0,   5),
            np.clip(row.get("corr_20d",  0.9),          -1,  1),
            float(self.position),
            np.clip(self.holding_days / self.max_holding, 0, 1),
            np.clip(self.unrealized_pnl / self.notional, -1, 1),
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0)

    # ── gym interface ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internals()
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        new_pos = self._ACTION_MAP[int(action)]
        row = self.data.iloc[self.t]
        spread_ret = float(row.get("spread_return", 0.0))   # daily spread change
        z_now = float(row.get("zscore", 0.0))
        reward = 0.0

        # Dead-zone: only allow entries when spread is sufficiently dislocated
        if new_pos != 0 and self.position == 0 and abs(z_now) < 1.5:
            new_pos = 0

        # ── close existing ─────────────────────────────────────────────────
        if self.position != 0 and new_pos != self.position:
            reward += self.unrealized_pnl - self.cost
            self.daily_pnls.append(self.unrealized_pnl - self.cost)
            self.position = 0
            self.unrealized_pnl = 0.0
            self.holding_days = 0

        # ── open new ───────────────────────────────────────────────────────
        if new_pos != 0 and self.position == 0:
            reward -= self.cost
            self.position = new_pos
            self.holding_days = 0
            self.unrealized_pnl = 0.0

        # ── mark-to-market ────────────────────────────────────────────────
        if self.position != 0:
            daily_pnl = self.position * spread_ret * self.notional
            self.unrealized_pnl += daily_pnl
            reward += daily_pnl
            self.holding_days += 1
            # force close if held too long
            if self.holding_days >= self.max_holding:
                reward += self.unrealized_pnl - self.cost - self.cost  # close cost
                self.daily_pnls.append(self.unrealized_pnl - self.cost)
                self.position = 0
                self.unrealized_pnl = 0.0
                self.holding_days = 0
        else:
            self.daily_pnls.append(0.0)

        self.t += 1
        terminated = self.t >= self.n - 1

        # force close at end of episode
        if terminated and self.position != 0:
            reward += self.unrealized_pnl - self.cost
            self.daily_pnls.append(self.unrealized_pnl - self.cost)

        # normalise reward to fraction-of-notional for training stability
        return self._obs(), float(reward) / self.notional, terminated, False, {}

    def episode_sharpe(self) -> float:
        if len(self.daily_pnls) < 5:
            return 0.0
        arr = np.array(self.daily_pnls)
        std = arr.std()
        return float(arr.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0
