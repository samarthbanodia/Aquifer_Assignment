"""
Data loader — downloads and caches daily OHLCV data via yfinance.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("results/data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{ticker}_{start}_{end}.csv"


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> Optional[pd.Series]:
    """Return daily adjusted-close prices for a single ticker."""
    path = _cache_path(ticker, start, end)
    if use_cache and path.exists():
        s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        return s

    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning("No data for %s", ticker)
            return None
        s = raw["Close"].squeeze()
        s.name = ticker
        s.to_csv(path)
        return s
    except Exception as exc:
        logger.error("Failed to download %s: %s", ticker, exc)
        return None


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted-close prices for a list of tickers.
    Returns a DataFrame with tickers as columns, dates as index.
    Drops any ticker that has more than 5% missing values.
    """
    frames = {}
    for ticker in tickers:
        s = download_ticker(ticker, start, end, use_cache)
        if s is not None:
            frames[ticker] = s

    if not frames:
        raise RuntimeError("No data downloaded for any ticker.")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Drop tickers with > 5% missing
    threshold = 0.05 * len(df)
    bad = [c for c in df.columns if df[c].isna().sum() > threshold]
    if bad:
        logger.warning("Dropping tickers with excessive missing data: %s", bad)
        df.drop(columns=bad, inplace=True)

    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


def get_latest_prices(
    tickers: List[str],
    lookback_days: int = 300,
) -> pd.DataFrame:
    """
    Fetch the most recent `lookback_days` of daily prices.
    Bypasses cache so we always get fresh data.
    """
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    return download_prices(tickers, start, end, use_cache=False)


def get_all_tickers(pairs: list) -> List[str]:
    """Flatten a list of (A, B) pairs into a unique ticker list."""
    tickers = set()
    for a, b in pairs:
        tickers.add(a)
        tickers.add(b)
    return sorted(tickers)
