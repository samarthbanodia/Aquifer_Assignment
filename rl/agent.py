"""
RL agent inference wrapper.

Wraps the trained PPO model with the same observation-building logic
used in the gymnasium environment so it can be called directly from
the backtesting engine with raw market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


class RLAgent:
    """
    Stateful inference wrapper for a trained PPO model.

    Maintains per-pair state (position, holding_days, unrealized_pnl)
    so it can be called once per trading day without needing the full history.
    """

    _ACTION_TO_SIGNAL = {0: 0, 1: 1, 2: -1}

    def __init__(self, model: PPO, vec_env: VecNormalize | None = None):
        self.model = model
        self.vec_env = vec_env   # kept for obs normalization during inference
        # per-pair state: pair_name → {position, holding_days, unrealized_pnl}
        self._state: dict[str, dict] = {}

    def _get_or_init(self, pair: str) -> dict:
        if pair not in self._state:
            self._state[pair] = {
                "position": 0,
                "holding_days": 0,
                "unrealized_pnl": 0.0,
            }
        return self._state[pair]

    def build_obs(
        self,
        pair: str,
        zscore: float,
        z_vel: float,
        z_accel: float,
        vix: float,
        corr_20d: float,
    ) -> np.ndarray:
        st = self._get_or_init(pair)
        notional = 10_000.0
        obs = np.array([
            np.clip(zscore,                         -5,  5),
            np.clip(z_vel,                          -3,  3),
            np.clip(z_accel,                        -2,  2),
            np.clip(vix / 30.0,                      0,  5),
            np.clip(corr_20d,                       -1,  1),
            float(st["position"]),
            np.clip(st["holding_days"] / 60.0,       0,  1),
            np.clip(st["unrealized_pnl"] / notional,-1,  1),
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0)

    def predict(self, pair: str, obs: np.ndarray) -> int:
        """Return signal: -1, 0, or +1."""
        action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
        # action may be a 1-element array or scalar depending on SB3 version
        a = int(np.asarray(action).flat[0])
        return self._ACTION_TO_SIGNAL[a]

    def update_state(
        self,
        pair: str,
        new_signal: int,
        spread_return: float,
        notional: float = 10_000.0,
    ):
        """
        Update per-pair state after an action is taken.
        Called by the backtest engine after each trade decision.
        """
        st = self._get_or_init(pair)
        old_pos = st["position"]

        # close
        if old_pos != 0 and new_signal != old_pos:
            st["unrealized_pnl"] = 0.0
            st["holding_days"] = 0
            st["position"] = 0

        # open
        if new_signal != 0 and st["position"] == 0:
            st["position"] = new_signal
            st["holding_days"] = 0
            st["unrealized_pnl"] = 0.0

        # mark-to-market
        if st["position"] != 0:
            st["unrealized_pnl"] += st["position"] * spread_return * notional
            st["holding_days"] += 1

    def reset_pair(self, pair: str):
        self._state.pop(pair, None)
