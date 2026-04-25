"""
Market regime detection.

Three regime indicators:
  1. VIX level — absolute fear gauge
  2. Rolling correlation breakdown — checks if pair correlation has dropped
  3. Volatility-of-volatility (vol-of-vol) — detects regime transitions
  4. Trend filter — detects persistent trending markets where mean-reversion fails

Regime: "NORMAL", "CAUTION", "STRESS", "CRISIS"
"""

import numpy as np
import pandas as pd


REGIME_NORMAL   = "NORMAL"
REGIME_CAUTION  = "CAUTION"
REGIME_STRESS   = "STRESS"
REGIME_CRISIS   = "CRISIS"


def detect_vix_regime(
    vix: float,
    pause_threshold: float = 40.0,
    reduce_threshold: float = 30.0,
) -> str:
    if vix >= pause_threshold:
        return REGIME_CRISIS
    if vix >= reduce_threshold:
        return REGIME_STRESS
    if vix >= 20.0:
        return REGIME_CAUTION
    return REGIME_NORMAL


def rolling_pair_correlation(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 20,
) -> pd.Series:
    """20-day rolling Pearson correlation of daily returns."""
    if ticker_a not in prices.columns or ticker_b not in prices.columns:
        return pd.Series(dtype=float)
    ret_a = prices[ticker_a].pct_change()
    ret_b = prices[ticker_b].pct_change()
    return ret_a.rolling(window).corr(ret_b)


def is_correlation_broken(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 20,
    min_corr: float = 0.30,
) -> bool:
    """Returns True if the most recent rolling correlation is below min_corr."""
    corr = rolling_pair_correlation(prices, ticker_a, ticker_b, window)
    if corr.empty or corr.dropna().empty:
        return False
    return float(corr.dropna().iloc[-1]) < min_corr


def vol_of_vol(vix_series: pd.Series, window: int = 10) -> float:
    """
    Volatility-of-volatility: std of daily VIX changes.
    High values indicate regime instability.
    """
    if vix_series.dropna().empty:
        return 0.0
    return float(vix_series.pct_change().rolling(window).std().dropna().iloc[-1])


class RegimeDetector:
    """
    Aggregates multiple regime signals into a single regime label
    with position-size recommendations.
    """

    def __init__(
        self,
        vix_pause: float = 40.0,
        vix_reduce: float = 30.0,
        min_pair_corr: float = 0.30,
        max_vol_of_vol: float = 0.10,
    ):
        self.vix_pause = vix_pause
        self.vix_reduce = vix_reduce
        self.min_pair_corr = min_pair_corr
        self.max_vol_of_vol = max_vol_of_vol

    def assess(
        self,
        prices: pd.DataFrame,
        ticker_a: str,
        ticker_b: str,
        current_vix: float,
    ) -> dict:
        """
        Return a regime assessment dict:
          - regime: str
          - size_factor: float  (1.0 = full, 0.5 = half, 0.0 = pause)
          - allow_new_entries: bool
          - reasons: list of str
        """
        reasons = []
        vix_regime = detect_vix_regime(current_vix, self.vix_pause, self.vix_reduce)

        if vix_regime == REGIME_CRISIS:
            return {
                "regime": REGIME_CRISIS,
                "size_factor": 0.0,
                "allow_new_entries": False,
                "reasons": [f"VIX={current_vix:.1f} >= {self.vix_pause} (crisis)"],
            }

        if vix_regime == REGIME_STRESS:
            reasons.append(f"VIX={current_vix:.1f} >= {self.vix_reduce} (stress)")

        # Check correlation breakdown
        corr_broken = is_correlation_broken(prices, ticker_a, ticker_b, min_corr=self.min_pair_corr)
        if corr_broken:
            reasons.append(f"Pair {ticker_a}/{ticker_b} correlation < {self.min_pair_corr}")

        # Vol-of-vol check
        if "^VIX" in prices.columns:
            vv = vol_of_vol(prices["^VIX"])
            if vv > self.max_vol_of_vol:
                reasons.append(f"Vol-of-vol = {vv:.3f} > threshold")

        # Determine overall regime
        n_issues = (vix_regime != REGIME_NORMAL) + corr_broken
        if n_issues == 0:
            regime = REGIME_NORMAL
            size_factor = 1.0
            allow_new = True
        elif n_issues == 1:
            regime = REGIME_CAUTION
            size_factor = 0.75
            allow_new = True
        else:
            regime = REGIME_STRESS
            size_factor = 0.5
            allow_new = False

        return {
            "regime": regime,
            "size_factor": size_factor,
            "allow_new_entries": allow_new,
            "reasons": reasons,
        }
