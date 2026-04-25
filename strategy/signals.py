"""
Signal generation — computes spread, z-score, and entry/exit signals.

Signal conventions:
  +1  →  Long the spread  (long A, short B)
  -1  →  Short the spread (short A, long B)
   0  →  Flat
"""

import numpy as np
import pandas as pd

from strategy.cointegration import CointResult


def compute_spread(
    prices: pd.DataFrame,
    coint_result: CointResult,
) -> pd.Series:
    """
    Compute the log-price spread:
        spread = log(A) - β·log(B) - α
    """
    a, b = coint_result.pair
    log_a = np.log(prices[a])
    log_b = np.log(prices[b])
    spread = log_a - coint_result.hedge_ratio * log_b - coint_result.intercept
    spread.name = f"{a}/{b}_spread"
    return spread


def compute_zscore(
    spread: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling z-score: (spread - μ) / σ over `window` days."""
    mu = spread.rolling(window, min_periods=window // 2).mean()
    sigma = spread.rolling(window, min_periods=window // 2).std()
    z = (spread - mu) / sigma.replace(0, np.nan)
    z.name = spread.name.replace("_spread", "_zscore")
    return z


def generate_position_signals(
    zscore: pd.Series,
    entry: float = 2.0,
    exit_: float = 0.5,
    stop: float = 3.5,
) -> pd.Series:
    """
    Stateful signal generator that respects entry/exit/stop logic.

    Returns a series of {-1, 0, +1} aligned with `zscore`.
    """
    signals = pd.Series(0, index=zscore.index, dtype=float)
    position = 0

    for i, z in enumerate(zscore):
        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            if z < -entry:
                position = 1           # open long spread
            elif z > entry:
                position = -1          # open short spread

        elif position == 1:
            if z >= -exit_ or z <= -stop:
                position = 0           # exit (target reached or stop hit)

        elif position == -1:
            if z <= exit_ or z >= stop:
                position = 0           # exit

        signals.iloc[i] = position

    return signals


class PairSignalEngine:
    """
    Combines cointegration result with rolling z-score and signal generation.
    Handles dynamic hedge-ratio re-estimation at rebalance dates.
    """

    def __init__(
        self,
        coint_result: CointResult,
        zscore_window: int,
        entry: float = 2.0,
        exit_: float = 0.5,
        stop: float = 3.5,
    ):
        self.coint = coint_result
        self.zscore_window = zscore_window
        self.entry = entry
        self.exit_ = exit_
        self.stop = stop

    def run(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Given a price DataFrame, return a DataFrame with columns:
            spread, zscore, signal
        """
        spread = compute_spread(prices, self.coint)
        zscore = compute_zscore(spread, self.zscore_window)
        signals = generate_position_signals(zscore, self.entry, self.exit_, self.stop)

        result = pd.DataFrame({
            "spread": spread,
            "zscore": zscore,
            "signal": signals,
        })
        return result

    def current_signal(self, prices: pd.DataFrame) -> dict:
        """
        Return the most recent signal for live scanning.
        """
        df = self.run(prices)
        last_row = df.dropna(subset=["zscore"]).iloc[-1]
        a, b = self.coint.pair
        z = last_row["zscore"]

        # Confidence score: how extreme is z relative to entry threshold?
        abs_z = abs(z)
        if abs_z >= self.entry:
            confidence = min(1.0, (abs_z - self.entry) / (self.stop - self.entry) + 0.5)
        else:
            confidence = max(0.0, abs_z / self.entry * 0.5)

        edge_bps = max(0.0, (abs_z - self.entry) * 10)   # rough bps estimate

        return {
            "pair": f"{a}/{b}",
            "ticker_a": a,
            "ticker_b": b,
            "hedge_ratio": round(self.coint.hedge_ratio, 4),
            "half_life_days": round(self.coint.half_life, 1),
            "zscore": round(float(z), 4),
            "signal": int(last_row["signal"]),
            "direction": (
                "LONG_SPREAD" if last_row["signal"] == 1
                else "SHORT_SPREAD" if last_row["signal"] == -1
                else "FLAT"
            ),
            "theoretical_edge_bps": round(edge_bps, 2),
            "confidence": round(confidence, 4),
            "spread": round(float(last_row["spread"]), 6),
        }
