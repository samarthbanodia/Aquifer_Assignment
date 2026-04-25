"""
run_backtest.py — entry point for the full backtest.

Usage:
    python run_backtest.py

Outputs:
    results/backtest_portfolio.csv
    results/backtest_trades.csv
    results/backtest_metrics.json
    results/backtest_plots.png
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
import seaborn as sns

import config
from data.loader import download_prices, get_all_tickers
from backtest.engine import BacktestEngine
from backtest.metrics import calculate_metrics, print_metrics
from backtest.costs import DEFAULT_COST_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def plot_results(portfolio: pd.Series, trades: pd.DataFrame, metrics: dict) -> None:
    """Generate a 4-panel performance chart."""
    sns.set_theme(style="darkgrid", font_scale=0.9)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Global Arbitrage Signal Engine — Backtest Results\n"
        f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
        f"CAGR: {metrics['cagr_pct']:.1f}% | "
        f"Max DD: {metrics['max_drawdown_pct']:.1f}% | "
        f"Win Rate: {metrics['win_rate_pct']:.1f}%",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Portfolio NAV ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(portfolio.index, portfolio.values / 1e6, color="#2196F3", linewidth=1.5, label="Strategy NAV")
    ax.axhline(config.INITIAL_CAPITAL / 1e6, color="gray", linestyle="--", linewidth=0.8, label="Initial Capital")
    ax.set_title("Portfolio NAV ($M)")
    ax.set_ylabel("Value ($M)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.legend(fontsize=8)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    returns = portfolio.pct_change().dropna()
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax() * 100
    ax.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.7, label="Drawdown")
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # ── Panel 3: Rolling 63-day Sharpe ───────────────────────────────────────
    ax = axes[1, 0]
    daily_rf = (1.05) ** (1 / 252) - 1
    excess = returns - daily_rf
    roll_sharpe = excess.rolling(63).mean() / returns.rolling(63).std() * np.sqrt(252)
    ax.plot(roll_sharpe.index, roll_sharpe.values, color="#4CAF50", linewidth=1.2, label="63-day Rolling Sharpe")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color="orange", linestyle=":", linewidth=0.8, label="Sharpe=1.0")
    ax.set_title("Rolling 63-day Sharpe Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.legend(fontsize=8)

    # ── Panel 4: Monthly Returns Heatmap ─────────────────────────────────────
    ax = axes[1, 1]
    monthly = (
        portfolio.resample("ME").last().pct_change().dropna() * 100
    )
    monthly_df = monthly.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][: len(pivot.columns)]

    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", center=0,
        annot=True, fmt=".1f", linewidths=0.3, linecolor="gray",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8, "label": "Return (%)"},
    )
    ax.set_title("Monthly Returns (%)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = "results/backtest_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", out)
    plt.close()


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    # ── 1. Download Data ──────────────────────────────────────────────────────
    logger.info("Downloading historical data: %s to %s", config.BACKTEST_START, config.BACKTEST_END)
    all_tickers = get_all_tickers(config.ETF_PAIRS) + config.REGIME_TICKERS
    prices = download_prices(all_tickers, config.BACKTEST_START, config.BACKTEST_END)
    logger.info("Loaded %d trading days x %d tickers", len(prices), len(prices.columns))

    # ── 2. Run Backtest ───────────────────────────────────────────────────────
    logger.info("Starting backtest engine...")
    engine = BacktestEngine(
        pairs=config.ETF_PAIRS,
        initial_capital=config.INITIAL_CAPITAL,
        cointegration_lookback=config.COINTEGRATION_LOOKBACK,
        zscore_window=config.ZSCORE_WINDOW,
        rebalance_freq=config.REBALANCE_FREQ,
        entry_z=config.ENTRY_ZSCORE,
        exit_z=config.EXIT_ZSCORE,
        stop_z=config.STOP_ZSCORE,
        max_pair_weight=config.MAX_PAIR_WEIGHT,
        portfolio_stop_dd=config.PORTFOLIO_STOP_DD,
        vix_pause=config.VIX_PAUSE_THRESHOLD,
        vix_reduce=config.VIX_REDUCE_THRESHOLD,
        cost_model=DEFAULT_COST_MODEL,
    )

    results = engine.run(prices)
    portfolio = results["portfolio"]
    trades = results["trades"]

    # ── 3. Compute Metrics ────────────────────────────────────────────────────
    # Market-neutral strategies measure alpha vs zero (cash earns T-bills separately)
    metrics = calculate_metrics(portfolio, trades, risk_free_rate=0.00)
    print_metrics(metrics)

    # ── 4. Save Results ───────────────────────────────────────────────────────
    portfolio.to_csv("results/backtest_portfolio.csv", header=True)
    logger.info("Portfolio saved: results/backtest_portfolio.csv")

    if not trades.empty:
        trades.to_csv("results/backtest_trades.csv", index=False)
        logger.info("Trades saved: results/backtest_trades.csv  (%d rows)", len(trades))

    with open("results/backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: results/backtest_metrics.json")

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    try:
        plot_results(portfolio, trades, metrics)
    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    logger.info("Backtest complete.")


if __name__ == "__main__":
    main()
