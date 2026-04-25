"""
Position sizing methods.

Two complementary approaches:
  1. KellySizer     — fraction of Kelly-optimal bet, derived from win-rate and P/L ratio
  2. VolTargetSizer — size to hit a target daily P&L volatility (used by most quant shops)
"""

import numpy as np
import pandas as pd


class KellySizer:
    """
    Half-Kelly criterion for pairs trading.

    Kelly fraction f* = (p·b - q) / b
      where:
        p = win rate
        b = avg_win / avg_loss  (payoff ratio)
        q = 1 - p

    We use HALF-Kelly as a safety margin to reduce variance.
    """

    def __init__(self, kelly_fraction: float = 0.5):
        self.kelly_fraction = kelly_fraction

    def compute_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Returns the fraction of capital to allocate per trade."""
        if avg_loss == 0 or win_rate <= 0:
            return 0.0
        b = avg_win / abs(avg_loss)
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b
        return max(0.0, min(1.0, kelly * self.kelly_fraction))

    def position_size(
        self,
        capital: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Return number of shares to trade."""
        frac = self.compute_fraction(win_rate, avg_win, avg_loss)
        notional = capital * frac
        return notional / price if price > 0 else 0.0


class VolTargetSizer:
    """
    Volatility-targeting position sizer.

    Sizes each pair so that the expected contribution to portfolio daily P&L
    volatility equals `target_daily_vol_pct * capital / n_pairs`.

    Formula:
        shares = (target_notional) / (pair_price * pair_daily_vol)

    This is equivalent to risk-parity across pairs.
    """

    def __init__(
        self,
        target_annual_vol: float = 0.10,
        max_pair_weight: float = 0.10,
    ):
        self.target_annual_vol = target_annual_vol
        self.max_pair_weight = max_pair_weight
        self._daily_vol = target_annual_vol / np.sqrt(252)

    def pair_notional(
        self,
        capital: float,
        spread_returns: pd.Series,
        n_active_pairs: int,
    ) -> float:
        """
        Compute target notional for one pair.
        Spread_returns = daily % changes of the spread series.
        """
        if len(spread_returns.dropna()) < 10:
            return capital * self.max_pair_weight / max(n_active_pairs, 1)

        spread_vol = spread_returns.std()  # daily
        if spread_vol <= 0:
            return 0.0

        # Each pair targets equal vol contribution
        per_pair_vol = self._daily_vol / max(np.sqrt(n_active_pairs), 1)
        notional = capital * per_pair_vol / spread_vol

        # Cap at max_pair_weight
        return min(notional, capital * self.max_pair_weight)
