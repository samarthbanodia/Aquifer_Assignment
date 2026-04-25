"""
Backtesting engine for the cointegration-based pairs strategy.

Methodology:
  - Rolling formation window (252 days) to test cointegration and estimate β
  - Re-estimate β every REBALANCE_FREQ days to avoid stale hedge ratios
  - Z-score computed over trailing 60-day window
  - Position sizing: equal-risk contribution across active pairs
  - Full transaction-cost accounting on every trade

Position units:
  - We trade the SPREAD, which means:
      Long spread  (+1): buy N_a shares of A, short N_b shares of B
      Short spread (-1): short N_a shares of A, buy N_b shares of B
  - N is sized so each pair risks TARGET_PAIR_NOTIONAL per trade
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest.costs import CostModel, DEFAULT_COST_MODEL
from strategy.cointegration import select_cointegrated_pairs, test_pair_cointegration
from strategy.signals import PairSignalEngine, compute_spread, compute_zscore, generate_position_signals

logger = logging.getLogger(__name__)


class PairPosition:
    """Tracks the state of a single pair trade."""

    def __init__(self, name: str, target_notional: float, ticker_a: str = "", ticker_b: str = ""):
        self.name = name
        self.target_notional = target_notional
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.shares_a: float = 0.0
        self.shares_b: float = 0.0
        self.signal: int = 0
        self.entry_date = None
        self.entry_prices: Dict[str, float] = {}
        self.cum_cost: float = 0.0
        self.cooldown_until: int = 0   # day_idx after which we may re-enter


class BacktestEngine:
    """
    Walk-forward backtester for a universe of ETF pairs.
    """

    def __init__(
        self,
        pairs: list,
        initial_capital: float,
        cointegration_lookback: int = 252,
        zscore_window: int = 60,
        rebalance_freq: int = 63,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
        max_pair_weight: float = 0.10,
        portfolio_stop_dd: float = 0.10,
        vix_pause: float = 40.0,
        vix_reduce: float = 30.0,
        cost_model: CostModel = DEFAULT_COST_MODEL,
    ):
        self.pairs = pairs
        self.initial_capital = initial_capital
        self.coint_lookback = cointegration_lookback
        self.zscore_window = zscore_window
        self.rebalance_freq = rebalance_freq
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_pair_weight = max_pair_weight
        self.portfolio_stop_dd = portfolio_stop_dd
        self.vix_pause = vix_pause
        self.vix_reduce = vix_reduce
        self.cost_model = cost_model

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_vix(self, prices: pd.DataFrame, date) -> float:
        if "^VIX" in prices.columns and date in prices.index:
            v = prices.loc[date, "^VIX"]
            return float(v) if not np.isnan(v) else 20.0
        return 20.0

    def _trade_pair(
        self,
        pos: PairPosition,
        new_signal: int,
        prices_today: pd.Series,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float,
        size_factor: float,
        trade_log: list,
        date,
    ) -> float:
        """Execute a signal change, return net P&L change (after costs) for today."""
        pnl = 0.0
        price_a = prices_today[ticker_a]
        price_b = prices_today[ticker_b]

        if np.isnan(price_a) or np.isnan(price_b):
            return 0.0

        old_signal = pos.signal

        # ── Close existing position ──────────────────────────────────────
        if old_signal != 0 and new_signal != old_signal:
            cost_a = self.cost_model.total_cost(price_a, pos.shares_a)
            cost_b = self.cost_model.total_cost(price_b, pos.shares_b)
            pos.cum_cost += cost_a + cost_b

            pnl_a = pos.shares_a * (price_a - pos.entry_prices.get(ticker_a, price_a))
            pnl_b = pos.shares_b * (price_b - pos.entry_prices.get(ticker_b, price_b))
            close_pnl = pnl_a + pnl_b - cost_a - cost_b
            pnl += close_pnl

            trade_log.append({
                "date": date,
                "pair": pos.name,
                "action": "CLOSE",
                "old_signal": old_signal,
                "pnl": close_pnl,
                "cost": cost_a + cost_b,
            })

            pos.shares_a = 0.0
            pos.shares_b = 0.0
            pos.signal = 0

        # ── Open new position (only if flat) ────────────────────────────
        if new_signal != 0 and pos.signal == 0:
            notional = pos.target_notional * size_factor
            # Size so that notional/2 goes into each leg (dollar-neutral)
            shares_a = (notional / 2.0) / price_a
            # Hedge-ratio-adjusted B leg
            shares_b = (notional / 2.0) / price_b * hedge_ratio

            if new_signal == 1:     # Long spread: buy A, sell B
                pos.shares_a = +shares_a
                pos.shares_b = -shares_b
            else:                   # Short spread: sell A, buy B
                pos.shares_a = -shares_a
                pos.shares_b = +shares_b

            cost_a = self.cost_model.total_cost(price_a, pos.shares_a)
            cost_b = self.cost_model.total_cost(price_b, pos.shares_b)
            pos.cum_cost += cost_a + cost_b
            pnl -= (cost_a + cost_b)

            pos.signal = new_signal
            pos.entry_date = date
            pos.entry_prices = {ticker_a: price_a, ticker_b: price_b}

            trade_log.append({
                "date": date,
                "pair": pos.name,
                "action": "OPEN",
                "signal": new_signal,
                "shares_a": round(pos.shares_a, 2),
                "shares_b": round(pos.shares_b, 2),
                "price_a": price_a,
                "price_b": price_b,
                "cost": cost_a + cost_b,
            })

        return pnl

    def _mark_to_market(
        self,
        positions: Dict[str, PairPosition],
        prices_today: pd.Series,
        pair_meta: dict,
    ) -> float:
        """Sum unrealised P&L across all open positions."""
        total = 0.0
        for name, pos in positions.items():
            if pos.signal == 0:
                continue
            # Use ticker info stored directly on position (survives pair_meta rebuild)
            a = pos.ticker_a or (pair_meta.get(name, {}).get("tickers", (None, None))[0])
            b = pos.ticker_b or (pair_meta.get(name, {}).get("tickers", (None, None))[1])
            if not a or not b:
                continue
            pa = prices_today.get(a, np.nan)
            pb = prices_today.get(b, np.nan)
            if np.isnan(pa) or np.isnan(pb):
                continue
            total += pos.shares_a * (pa - pos.entry_prices.get(a, pa))
            total += pos.shares_b * (pb - pos.entry_prices.get(b, pb))
        return total

    # ── main run ─────────────────────────────────────────────────────────────

    def run(self, prices: pd.DataFrame) -> dict:
        """
        Execute the backtest over the full price history.
        Returns dict with keys: portfolio, trades, pair_returns.
        """
        dates = prices.index
        capital = self.initial_capital
        portfolio_values = []
        trade_log = []
        pair_meta: Dict[str, dict] = {}    # stores hedge ratios, engines per pair
        positions: Dict[str, PairPosition] = {}

        peak_value = capital
        strategy_paused = False

        for day_idx, date in enumerate(dates):
            prices_today = prices.loc[date]

            # ── Periodic rebalance: re-test cointegration ─────────────────
            if day_idx >= self.coint_lookback and day_idx % self.rebalance_freq == 0:
                window_prices = prices.iloc[
                    max(0, day_idx - self.coint_lookback) : day_idx
                ]
                active_coint = select_cointegrated_pairs(
                    window_prices, self.pairs, significance=0.10
                )
                # Rebuild pair_meta
                pair_meta = {}
                for cr in active_coint:
                    a, b = cr.pair
                    name = f"{a}/{b}"
                    pair_meta[name] = {
                        "tickers": (a, b),
                        "coint": cr,
                        "engine": PairSignalEngine(
                            cr, self.zscore_window, self.entry_z, self.exit_z, self.stop_z
                        ),
                    }
                    if name not in positions:
                        notional_per_pair = min(
                            capital * self.max_pair_weight,
                            capital / max(len(pair_meta), 1),
                        )
                        positions[name] = PairPosition(name, notional_per_pair, a, b)

                logger.info(
                    "Day %d (%s): %d cointegrated pairs found",
                    day_idx, date.date(), len(pair_meta),
                )

            if not pair_meta:
                portfolio_values.append(capital)
                continue

            # ── Risk filters ──────────────────────────────────────────────
            vix = self._get_vix(prices, date)
            size_factor = 1.0
            if vix >= self.vix_pause:
                strategy_paused = True
            elif vix >= self.vix_reduce:
                size_factor = 0.5

            # Check portfolio drawdown
            mtm = self._mark_to_market(positions, prices_today, pair_meta)
            current_value = capital + mtm
            peak_value = max(peak_value, current_value)
            dd = (peak_value - current_value) / peak_value
            if dd >= self.portfolio_stop_dd:
                strategy_paused = True
            if dd < self.portfolio_stop_dd * 0.5:
                strategy_paused = False   # resume when recovered halfway

            # ── Generate signals and trade ────────────────────────────────
            day_pnl = 0.0

            # Close positions for pairs that dropped out of pair_meta
            for name, pos in positions.items():
                if pos.signal != 0 and name not in pair_meta:
                    a, b = pos.ticker_a, pos.ticker_b
                    if a and b:
                        pnl = self._trade_pair(
                            pos, 0, prices_today, a, b,
                            1.0, 1.0, trade_log, date,
                        )
                        day_pnl += pnl
                        capital += pnl

            for name, meta in pair_meta.items():
                a, b = meta["tickers"]
                if a not in prices.columns or b not in prices.columns:
                    continue

                pos = positions[name]

                # Compute current z-score incrementally using rolling window
                history_start = max(0, day_idx - self.zscore_window - 5)
                window = prices.iloc[history_start : day_idx + 1]
                if len(window) < self.zscore_window:
                    continue

                try:
                    from strategy.signals import compute_spread, compute_zscore
                    spread = compute_spread(window[[a, b]], meta["coint"])
                    zscore = compute_zscore(spread, self.zscore_window)
                    z = float(zscore.iloc[-1]) if not zscore.empty else float("nan")
                except Exception:
                    continue

                if np.isnan(z):
                    continue

                # Incremental signal logic — using actual pos.signal state
                old_signal = pos.signal
                if old_signal == 0:
                    if day_idx < pos.cooldown_until:
                        new_signal = 0   # in cooldown after a stop-loss
                    elif z < -self.entry_z:
                        new_signal = 1
                    elif z > self.entry_z:
                        new_signal = -1
                    else:
                        new_signal = 0
                elif old_signal == 1:
                    if z >= -self.exit_z:
                        new_signal = 0   # take profit / exit
                    elif z <= -self.stop_z:
                        new_signal = 0   # stop-loss
                        pos.cooldown_until = day_idx + 20
                    else:
                        new_signal = 1   # hold
                else:  # old_signal == -1
                    if z <= self.exit_z:
                        new_signal = 0
                    elif z >= self.stop_z:
                        new_signal = 0
                        pos.cooldown_until = day_idx + 20
                    else:
                        new_signal = -1

                if strategy_paused and new_signal != 0 and pos.signal == 0:
                    new_signal = 0   # don't open new positions while paused

                # Correlation filter: block new entries if pair relationship has broken
                if new_signal != 0 and pos.signal == 0:
                    recent = prices.iloc[max(0, day_idx - 20) : day_idx + 1]
                    if len(recent) >= 10:
                        corr = recent[a].pct_change().corr(recent[b].pct_change())
                        if not np.isnan(corr) and corr < 0.80:
                            new_signal = 0   # correlation breakdown — skip entry

                pnl = self._trade_pair(
                    pos, new_signal, prices_today, a, b,
                    meta["coint"].hedge_ratio, size_factor,
                    trade_log, date,
                )
                day_pnl += pnl
                capital += pnl

            portfolio_values.append(capital)

        portfolio = pd.Series(portfolio_values, index=dates, name="portfolio_value")
        trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        return {"portfolio": portfolio, "trades": trades}
