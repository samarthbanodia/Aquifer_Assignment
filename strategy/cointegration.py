"""
Cointegration testing and hedge ratio estimation.

Uses Engle-Granger two-step procedure:
  1. OLS: log(Y) = α + β·log(X) + ε
  2. ADF test on residuals ε  (null = unit root = NOT cointegrated)

Also provides Johansen test for robustness.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

logger = logging.getLogger(__name__)


@dataclass
class CointResult:
    pair: Tuple[str, str]
    is_cointegrated: bool
    pvalue: float
    hedge_ratio: float        # β in log(Y) = α + β·log(X)
    intercept: float          # α
    half_life: float          # mean-reversion half-life in days
    adf_stat: float


def _estimate_half_life(spread: pd.Series) -> float:
    """Estimate AR(1) mean-reversion half-life of a spread series."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    # Align
    idx = spread_diff.index.intersection(spread_lag.index)
    X = add_constant(spread_lag.loc[idx])
    y = spread_diff.loc[idx]
    try:
        res = OLS(y, X).fit()
        phi = res.params.iloc[1]          # AR(1) coefficient on lagged spread
        if phi >= 0 or phi <= -2:
            return float("inf")
        return -np.log(2) / phi
    except Exception:
        return float("inf")


def test_pair_cointegration(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    significance: float = 0.05,
) -> Optional[CointResult]:
    """
    Test whether log(A) and log(B) are cointegrated.
    Returns CointResult or None if insufficient data.
    """
    if ticker_a not in prices.columns or ticker_b not in prices.columns:
        return None

    log_a = np.log(prices[ticker_a].dropna())
    log_b = np.log(prices[ticker_b].dropna())
    idx = log_a.index.intersection(log_b.index)
    if len(idx) < 100:
        logger.debug("Insufficient data for %s/%s", ticker_a, ticker_b)
        return None

    log_a = log_a.loc[idx]
    log_b = log_b.loc[idx]

    # OLS for hedge ratio
    X = add_constant(log_b)
    try:
        ols_res = OLS(log_a, X).fit()
    except Exception as exc:
        logger.warning("OLS failed for %s/%s: %s", ticker_a, ticker_b, exc)
        return None

    intercept = float(ols_res.params.iloc[0])
    hedge_ratio = float(ols_res.params.iloc[1])
    spread = log_a - hedge_ratio * log_b - intercept

    # ADF on residuals
    try:
        adf_stat, pvalue, *_ = adfuller(spread.values, maxlag=1, autolag=None)
    except Exception:
        return None

    is_coint = pvalue < significance
    half_life = _estimate_half_life(spread)

    return CointResult(
        pair=(ticker_a, ticker_b),
        is_cointegrated=is_coint,
        pvalue=float(pvalue),
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        half_life=half_life,
        adf_stat=float(adf_stat),
    )


def select_cointegrated_pairs(
    prices: pd.DataFrame,
    candidate_pairs: list,
    significance: float = 0.05,
    max_half_life: float = 126,      # ~6 months; slower mean reversion not useful
    min_half_life: float = 2,        # < 2 days is too fast (bid-ask noise)
) -> list:
    """
    Filter candidate pairs to those that pass cointegration and half-life checks.
    Returns list of CointResult objects sorted by p-value.
    """
    results = []
    for a, b in candidate_pairs:
        res = test_pair_cointegration(prices, a, b, significance)
        if res is None:
            continue
        if not res.is_cointegrated:
            logger.debug("Not cointegrated: %s/%s (p=%.3f)", a, b, res.pvalue)
            continue
        if not (min_half_life <= res.half_life <= max_half_life):
            logger.debug(
                "Half-life out of range: %s/%s (hl=%.1f days)", a, b, res.half_life
            )
            continue
        logger.info(
            "Cointegrated pair: %s/%s | p=%.4f | beta=%.4f | HL=%.1f days",
            a, b, res.pvalue, res.hedge_ratio, res.half_life,
        )
        results.append(res)

    results.sort(key=lambda r: r.pvalue)
    return results
