"""
run_rl_backtest.py  —  Run and compare Classical vs RL strategy.

The RL agent replaces the fixed z-score thresholds with a learned policy.
Everything else (cointegration selection, position sizing, costs) is identical.

Usage:
    python run_rl_backtest.py

Outputs:
    results/rl_backtest_portfolio.csv
    results/rl_backtest_trades.csv
    results/rl_backtest_metrics.json
    results/rl_vs_classical.png
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import config
from data.loader import download_prices, get_all_tickers
from backtest.costs import DEFAULT_COST_MODEL
from backtest.metrics import calculate_metrics, print_metrics
from strategy.cointegration import select_cointegrated_pairs
from strategy.signals import compute_spread, compute_zscore
from rl.trainer import load_model, build_pair_dataset
from rl.agent import RLAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── RL Backtest Engine ─────────────────────────────────────────────────────────

class RLBacktestEngine:
    """
    Same walk-forward framework as BacktestEngine but uses the RL agent
    for position decisions instead of fixed z-score thresholds.
    """

    def __init__(
        self,
        agent: RLAgent,
        pairs: list,
        initial_capital: float,
        cointegration_lookback: int = 252,
        zscore_window: int = 60,
        rebalance_freq: int = 63,
        max_pair_weight: float = 0.20,
        portfolio_stop_dd: float = 0.10,
        vix_pause: float = 40.0,
        vix_reduce: float = 30.0,
    ):
        self.agent = agent
        self.pairs = pairs
        self.initial_capital = initial_capital
        self.coint_lookback = cointegration_lookback
        self.zscore_window = zscore_window
        self.rebalance_freq = rebalance_freq
        self.max_pair_weight = max_pair_weight
        self.portfolio_stop_dd = portfolio_stop_dd
        self.vix_pause = vix_pause
        self.vix_reduce = vix_reduce
        self.cost_model = DEFAULT_COST_MODEL

    def run(self, prices: pd.DataFrame) -> dict:
        from backtest.engine import PairPosition

        dates = prices.index
        capital = self.initial_capital
        portfolio_values = []
        trade_log = []
        pair_meta: dict = {}
        positions: dict = {}
        peak_value = capital
        strategy_paused = False

        # per-pair rolling observation history
        pair_z_history: dict[str, list] = {}

        for day_idx, date in enumerate(dates):
            prices_today = prices.loc[date]

            # ── Quarterly rebalance ──────────────────────────────────────
            if day_idx >= self.coint_lookback and day_idx % self.rebalance_freq == 0:
                window_prices = prices.iloc[max(0, day_idx - self.coint_lookback): day_idx]
                active_coint = select_cointegrated_pairs(
                    window_prices, self.pairs, significance=0.10
                )
                new_meta = {}
                for cr in active_coint:
                    a, b = cr.pair
                    name = f"{a}/{b}"
                    new_meta[name] = {"tickers": (a, b), "coint": cr}
                    if name not in positions:
                        notional = min(
                            capital * self.max_pair_weight,
                            capital / max(len(new_meta), 1),
                        )
                        positions[name] = PairPosition(name, notional, a, b)

                # Close positions for pairs that fell out of cointegration
                for name, pos in positions.items():
                    if pos.signal != 0 and name not in new_meta:
                        a, b = pos.ticker_a, pos.ticker_b
                        pa = prices_today.get(a, np.nan)
                        pb = prices_today.get(b, np.nan)
                        if not (np.isnan(pa) or np.isnan(pb)):
                            cost = self.cost_model.total_cost(pa, pos.shares_a) + \
                                   self.cost_model.total_cost(pb, pos.shares_b)
                            pnl_a = pos.shares_a * (pa - pos.entry_prices.get(a, pa))
                            pnl_b = pos.shares_b * (pb - pos.entry_prices.get(b, pb))
                            capital += pnl_a + pnl_b - cost
                        pos.signal = 0
                        pos.shares_a = pos.shares_b = 0.0
                        pos.unrealized_pnl = 0.0 if hasattr(pos, "unrealized_pnl") else None

                pair_meta = new_meta
                logger.info("Day %d (%s): %d cointegrated pairs", day_idx, date.date(), len(pair_meta))

            if not pair_meta:
                portfolio_values.append(capital)
                continue

            # ── Risk filters ─────────────────────────────────────────────
            vix = float(prices.loc[date, "^VIX"]) if "^VIX" in prices.columns else 20.0
            size_factor = 0.5 if vix >= self.vix_reduce else 1.0
            if vix >= self.vix_pause:
                strategy_paused = True
            elif vix < self.vix_reduce:
                strategy_paused = False

            # ── Per-pair RL decisions ─────────────────────────────────────
            for name, meta in pair_meta.items():
                a, b = meta["tickers"]
                if a not in prices.columns or b not in prices.columns:
                    continue

                pos = positions[name]
                coint = meta["coint"]

                # compute current z-score and its derivatives
                hist_start = max(0, day_idx - self.zscore_window - 10)
                window = prices.iloc[hist_start: day_idx + 1]
                if len(window) < self.zscore_window:
                    continue

                try:
                    spread_series = compute_spread(window[[a, b]], coint)
                    zscore_series = compute_zscore(spread_series, self.zscore_window)
                except Exception:
                    continue

                if zscore_series.dropna().empty:
                    continue

                z_vals = zscore_series.dropna()
                z = float(z_vals.iloc[-1])
                z_vel = float(z_vals.diff(5).iloc[-1]) / 5 if len(z_vals) >= 6 else 0.0
                z_accel = float(z_vals.diff(5).diff(1).iloc[-1]) if len(z_vals) >= 7 else 0.0

                # spread return for agent state update
                spread_vals = spread_series.dropna()
                std_val = float(spread_vals.rolling(self.zscore_window).std().iloc[-1])
                spread_ret = float(spread_vals.diff().iloc[-1]) / (std_val if std_val > 0 else 1)

                # 20-day rolling correlation
                ret_a = np.log(prices[a]).diff()
                ret_b = np.log(prices[b]).diff()
                corr_series = ret_a.iloc[max(0, day_idx-20):day_idx+1].corr(
                    ret_b.iloc[max(0, day_idx-20):day_idx+1]
                )
                corr = float(corr_series) if not np.isnan(corr_series) else 0.9

                # Build observation and get RL action
                obs = self.agent.build_obs(name, z, z_vel, z_accel, vix, corr)
                new_signal = self.agent.predict(name, obs)

                # Override: don't open new if paused, apply correlation filter
                if strategy_paused and new_signal != 0 and pos.signal == 0:
                    new_signal = 0
                if new_signal != 0 and pos.signal == 0 and corr < 0.80:
                    new_signal = 0

                # Update agent internal state
                self.agent.update_state(name, new_signal, spread_ret)

                # Execute trade (reuse engine logic)
                pa = float(prices_today.get(a, np.nan))
                pb = float(prices_today.get(b, np.nan))
                if np.isnan(pa) or np.isnan(pb):
                    continue

                old_signal = pos.signal

                # close
                if old_signal != 0 and new_signal != old_signal:
                    cost_a = self.cost_model.total_cost(pa, pos.shares_a)
                    cost_b = self.cost_model.total_cost(pb, pos.shares_b)
                    pnl = (pos.shares_a * (pa - pos.entry_prices.get(a, pa)) +
                           pos.shares_b * (pb - pos.entry_prices.get(b, pb)) -
                           cost_a - cost_b)
                    capital += pnl
                    trade_log.append({"date": date, "pair": name, "action": "CLOSE",
                                      "pnl": pnl, "cost": cost_a + cost_b})
                    pos.shares_a = pos.shares_b = 0.0
                    pos.signal = 0

                # open
                if new_signal != 0 and pos.signal == 0:
                    notional = pos.target_notional * size_factor
                    shares_a = (notional / 2) / pa
                    shares_b = (notional / 2) / pb * coint.hedge_ratio
                    if new_signal == 1:
                        pos.shares_a, pos.shares_b = +shares_a, -shares_b
                    else:
                        pos.shares_a, pos.shares_b = -shares_a, +shares_b
                    cost_a = self.cost_model.total_cost(pa, pos.shares_a)
                    cost_b = self.cost_model.total_cost(pb, pos.shares_b)
                    capital -= (cost_a + cost_b)
                    pos.signal = new_signal
                    pos.entry_prices = {a: pa, b: pb}
                    trade_log.append({"date": date, "pair": name, "action": "OPEN",
                                      "signal": new_signal, "cost": cost_a + cost_b})

            portfolio_values.append(capital)

        portfolio = pd.Series(portfolio_values, index=dates, name="rl_portfolio_value")
        trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        return {"portfolio": portfolio, "trades": trades}


# ── comparison plot ────────────────────────────────────────────────────────────

def plot_comparison(classical: pd.Series, rl: pd.Series, m_classical: dict, m_rl: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Classical Stat Arb vs RL-Enhanced Strategy", fontsize=13, fontweight="bold")

    # NAV comparison
    ax = axes[0]
    ax.plot(classical.index, (classical / classical.iloc[0] - 1) * 100,
            label=f"Classical (Sharpe {m_classical['sharpe_ratio']:.2f})", color="#2196F3")
    ax.plot(rl.index, (rl / rl.iloc[0] - 1) * 100,
            label=f"RL Agent  (Sharpe {m_rl['sharpe_ratio']:.2f})", color="#4CAF50")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Cumulative Return (%)")
    ax.set_ylabel("Return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.legend()

    # Metrics bar chart
    ax = axes[1]
    labels = ["Total\nReturn %", "Sharpe", "Win Rate %", "Profit\nFactor"]
    c_vals = [m_classical["total_return_pct"], m_classical["sharpe_ratio"],
              m_classical["win_rate_pct"], min(m_classical["profit_factor"], 5)]
    r_vals = [m_rl["total_return_pct"], m_rl["sharpe_ratio"],
              m_rl["win_rate_pct"], min(m_rl["profit_factor"], 5)]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, c_vals, w, label="Classical", color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, r_vals, w, label="RL Agent",  color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Key Metrics Comparison")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out = "results/rl_vs_classical.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Comparison plot saved: %s", out)
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    Path("results").mkdir(exist_ok=True)

    # load RL model
    model = load_model()
    agent = RLAgent(model)

    # load test data
    all_tickers = get_all_tickers(config.ETF_PAIRS) + config.REGIME_TICKERS
    logger.info("Loading test data: %s to %s", config.BACKTEST_START, config.BACKTEST_END)
    prices = download_prices(all_tickers, config.BACKTEST_START, config.BACKTEST_END)

    # run RL backtest
    logger.info("Running RL backtest...")
    rl_engine = RLBacktestEngine(
        agent=agent,
        pairs=config.ETF_PAIRS,
        initial_capital=config.INITIAL_CAPITAL,
        cointegration_lookback=config.COINTEGRATION_LOOKBACK,
        zscore_window=config.ZSCORE_WINDOW,
        rebalance_freq=config.REBALANCE_FREQ,
        max_pair_weight=config.MAX_PAIR_WEIGHT,
        portfolio_stop_dd=config.PORTFOLIO_STOP_DD,
        vix_pause=config.VIX_PAUSE_THRESHOLD,
        vix_reduce=config.VIX_REDUCE_THRESHOLD,
    )
    rl_results = rl_engine.run(prices)
    rl_portfolio = rl_results["portfolio"]
    rl_trades = rl_results["trades"]

    rl_metrics = calculate_metrics(rl_portfolio, rl_trades, risk_free_rate=0.0)

    print("\n=== RL AGENT BACKTEST ===")
    print_metrics(rl_metrics)

    rl_portfolio.to_csv("results/rl_backtest_portfolio.csv")
    if not rl_trades.empty:
        rl_trades.to_csv("results/rl_backtest_trades.csv", index=False)
    with open("results/rl_backtest_metrics.json", "w") as f:
        json.dump(rl_metrics, f, indent=2)

    # load classical results for comparison
    try:
        classical_portfolio = pd.read_csv(
            "results/backtest_portfolio.csv", index_col=0, parse_dates=True
        ).squeeze()
        classical_trades = pd.read_csv("results/backtest_trades.csv")
        classical_metrics = calculate_metrics(
            classical_portfolio, classical_trades, risk_free_rate=0.0
        )

        print("\n=== CLASSICAL STRATEGY ===")
        print_metrics(classical_metrics)

        print("\n=== IMPROVEMENT SUMMARY ===")
        for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                    "win_rate_pct", "profit_factor", "num_trades"]:
            c = classical_metrics[key]
            r = rl_metrics[key]
            delta = r - c
            arrow = "+" if delta > 0 else ""
            print(f"  {key:<22}  Classical: {c:>7.3f}   RL: {r:>7.3f}   ({arrow}{delta:.3f})")

        plot_comparison(classical_portfolio, rl_portfolio, classical_metrics, rl_metrics)

    except FileNotFoundError:
        logger.warning("Classical results not found — run run_backtest.py first.")

    logger.info("RL backtest complete.")


if __name__ == "__main__":
    main()
