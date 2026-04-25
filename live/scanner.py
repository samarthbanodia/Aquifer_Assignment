"""
Live signal scanner — runs continuously, scans all pairs every SCAN_INTERVAL_SEC seconds,
and logs signals to both a human-readable log file and a structured CSV.

Signal log columns:
    timestamp, pair, ticker_a, ticker_b, direction, zscore,
    hedge_ratio, half_life_days, theoretical_edge_bps, confidence,
    spread, regime, vix
"""

import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from data.loader import get_latest_prices, get_all_tickers
from strategy.cointegration import select_cointegrated_pairs
from strategy.signals import PairSignalEngine
from risk.regime import RegimeDetector

# ── Logging setup ──────────────────────────────────────────────────────────────

Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.SIGNAL_LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def _init_csv(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "pair", "ticker_a", "ticker_b",
                "direction", "zscore", "hedge_ratio", "half_life_days",
                "theoretical_edge_bps", "confidence", "spread",
                "regime", "vix", "cointegration_pvalue",
            ])
            writer.writeheader()


def _append_csv(path: str, row: dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


class LiveScanner:
    """
    Scans all configured pairs for cointegration and active z-score signals.
    Logs any signal with confidence >= MIN_CONFIDENCE.
    """

    def __init__(self):
        self.regime_detector = RegimeDetector(
            vix_pause=config.VIX_PAUSE_THRESHOLD,
            vix_reduce=config.VIX_REDUCE_THRESHOLD,
        )
        _init_csv(config.SIGNAL_CSV_FILE)
        self._scan_count = 0
        self._last_signals: dict = {}    # tracks last logged state per pair

    # ── helpers ────────────────────────────────────────────────────────────────

    def _get_vix(self, prices: pd.DataFrame) -> float:
        if "^VIX" in prices.columns:
            v = prices["^VIX"].dropna()
            return float(v.iloc[-1]) if not v.empty else 20.0
        return 20.0

    def _fetch_prices(self) -> Optional[pd.DataFrame]:
        all_tickers = get_all_tickers(config.ETF_PAIRS) + config.REGIME_TICKERS
        try:
            prices = get_latest_prices(all_tickers, lookback_days=config.LIVE_LOOKBACK_DAYS)
            return prices
        except Exception as exc:
            logger.error("Failed to fetch prices: %s", exc)
            return None

    def _scan_pair(
        self,
        prices: pd.DataFrame,
        coint_result,
        vix: float,
    ) -> Optional[dict]:
        """Return a signal dict if a signal exists for this pair, else None."""
        a, b = coint_result.pair

        if a not in prices.columns or b not in prices.columns:
            return None

        engine = PairSignalEngine(
            coint_result,
            config.ZSCORE_WINDOW,
            config.ENTRY_ZSCORE,
            config.EXIT_ZSCORE,
            config.STOP_ZSCORE,
        )

        try:
            sig = engine.current_signal(prices[[a, b]])
        except Exception as exc:
            logger.debug("Signal error for %s/%s: %s", a, b, exc)
            return None

        if abs(sig["zscore"]) < config.EXIT_ZSCORE:
            return None   # no active signal

        regime_info = self.regime_detector.assess(prices, a, b, vix)

        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "pair": sig["pair"],
            "ticker_a": a,
            "ticker_b": b,
            "direction": sig["direction"],
            "zscore": sig["zscore"],
            "hedge_ratio": sig["hedge_ratio"],
            "half_life_days": sig["half_life_days"],
            "theoretical_edge_bps": sig["theoretical_edge_bps"],
            "confidence": sig["confidence"],
            "spread": sig["spread"],
            "regime": regime_info["regime"],
            "vix": round(vix, 2),
            "cointegration_pvalue": round(coint_result.pvalue, 5),
        }
        return row

    # ── main scan loop ─────────────────────────────────────────────────────────

    def scan_once(self) -> list:
        """Run a single scan across all pairs. Returns list of signal dicts."""
        self._scan_count += 1
        logger.info("=== Scan #%d at %s ===", self._scan_count, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

        prices = self._fetch_prices()
        if prices is None:
            logger.warning("Skipping scan — data fetch failed")
            return []

        vix = self._get_vix(prices)
        logger.info("VIX: %.2f | Price matrix: %d rows x %d cols", vix, len(prices), len(prices.columns))

        # Run cointegration on recent history
        coint_results = select_cointegrated_pairs(
            prices.iloc[-config.COINTEGRATION_LOOKBACK :],
            config.ETF_PAIRS,
            significance=0.05,
        )
        logger.info("Cointegrated pairs found: %d", len(coint_results))

        signals = []
        for cr in coint_results:
            row = self._scan_pair(prices, cr, vix)
            if row is None:
                continue
            if row["confidence"] < config.MIN_CONFIDENCE:
                continue

            # Only log if signal is new or changed
            pair_key = row["pair"]
            last = self._last_signals.get(pair_key, {})
            if last.get("direction") != row["direction"] or row["direction"] != "FLAT":
                signals.append(row)
                _append_csv(config.SIGNAL_CSV_FILE, row)
                self._last_signals[pair_key] = row

                logger.info(
                    "SIGNAL | %-12s | %-12s | z=%+.3f | edge=%.1fbps | conf=%.2f | regime=%s",
                    row["pair"],
                    row["direction"],
                    row["zscore"],
                    row["theoretical_edge_bps"],
                    row["confidence"],
                    row["regime"],
                )

        if not signals:
            logger.info("No actionable signals this scan.")

        return signals

    def run(self, max_scans: Optional[int] = None) -> None:
        """
        Continuous scanning loop.
        Runs indefinitely (or until max_scans) with SCAN_INTERVAL_SEC between scans.
        Designed to run stably for 48+ hours.
        """
        logger.info("Live Scanner started. Interval: %ds | Log: %s",
                    config.SCAN_INTERVAL_SEC, config.SIGNAL_LOG_FILE)
        logger.info("Universe: %d pairs | VIX pause: %.0f | VIX reduce: %.0f",
                    len(config.ETF_PAIRS), config.VIX_PAUSE_THRESHOLD, config.VIX_REDUCE_THRESHOLD)

        scan_num = 0
        while True:
            try:
                self.scan_once()
            except KeyboardInterrupt:
                logger.info("Scanner stopped by user after %d scans.", self._scan_count)
                break
            except Exception as exc:
                logger.error("Unexpected error in scan loop: %s", exc, exc_info=True)

            scan_num += 1
            if max_scans is not None and scan_num >= max_scans:
                logger.info("Reached max_scans=%d, stopping.", max_scans)
                break

            logger.info("Sleeping %ds until next scan...", config.SCAN_INTERVAL_SEC)
            try:
                time.sleep(config.SCAN_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("Scanner stopped by user.")
                break
