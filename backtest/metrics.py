"""
Performance metrics for the backtest.
All metrics computed from daily portfolio values.
"""

import numpy as np
import pandas as pd


def calculate_metrics(
    portfolio: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Compute comprehensive performance metrics.

    Args:
        portfolio: daily portfolio NAV series
        trades:    trade log DataFrame (from BacktestEngine)
        risk_free_rate: annualised risk-free rate (e.g. 0.05 = 5%)

    Returns:
        dict of metrics
    """
    returns = portfolio.pct_change().dropna()
    annual_factor = 252
    daily_rf = (1 + risk_free_rate) ** (1 / annual_factor) - 1

    # ── Returns ─────────────────────────────────────────────────────────────
    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    cagr = (1 + total_return) ** (annual_factor / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(annual_factor)

    # ── Sharpe / Sortino / Calmar ────────────────────────────────────────────
    excess = returns - daily_rf
    sharpe = (excess.mean() / returns.std()) * np.sqrt(annual_factor) if returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(annual_factor) if len(downside) > 0 else 1e-9
    sortino = (returns.mean() - daily_rf) * annual_factor / downside_vol

    # ── Drawdown ─────────────────────────────────────────────────────────────
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # ── Longest drawdown period ───────────────────────────────────────────────
    in_dd = drawdown < 0
    dd_lengths = []
    current = 0
    for v in in_dd:
        if v:
            current += 1
        else:
            if current:
                dd_lengths.append(current)
            current = 0
    max_dd_duration = max(dd_lengths) if dd_lengths else 0

    # ── Trade-level metrics ──────────────────────────────────────────────────
    win_rate = 0.0
    avg_holding = 0.0
    total_turnover = 0.0
    avg_trade_pnl = 0.0
    profit_factor = 0.0
    num_trades = 0

    if not trades.empty and "action" in trades.columns:
        close_trades = trades[trades["action"] == "CLOSE"].copy()
        num_trades = len(close_trades)

        if num_trades > 0 and "pnl" in close_trades.columns:
            winners = close_trades[close_trades["pnl"] > 0]
            win_rate = len(winners) / num_trades
            avg_trade_pnl = close_trades["pnl"].mean()

            gross_profit = winners["pnl"].sum()
            gross_loss = abs(close_trades[close_trades["pnl"] < 0]["pnl"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Holding period: sequential match per pair (each OPEN gets the next CLOSE)
        trade_seq = trades[trades["action"].isin(["OPEN", "CLOSE"])].copy()
        trade_seq["date"] = pd.to_datetime(trade_seq["date"])
        holding_samples = []
        for pair_name, grp in trade_seq.groupby("pair"):
            grp = grp.sort_values("date").reset_index(drop=True)
            last_open = None
            for _, row in grp.iterrows():
                if row["action"] == "OPEN":
                    last_open = row["date"]
                elif row["action"] == "CLOSE" and last_open is not None:
                    holding_samples.append((row["date"] - last_open).days)
                    last_open = None
        if holding_samples:
            avg_holding = float(pd.Series(holding_samples).mean())

        if "cost" in trades.columns:
            total_turnover = trades["cost"].sum()

    return {
        "total_return_pct":   round(total_return * 100, 2),
        "cagr_pct":           round(cagr * 100, 2),
        "annual_vol_pct":     round(annual_vol * 100, 2),
        "sharpe_ratio":       round(sharpe, 3),
        "sortino_ratio":      round(sortino, 3),
        "calmar_ratio":       round(calmar, 3),
        "max_drawdown_pct":   round(max_dd * 100, 2),
        "max_dd_duration_days": max_dd_duration,
        "win_rate_pct":       round(win_rate * 100, 1),
        "num_trades":         num_trades,
        "avg_trade_pnl":      round(avg_trade_pnl, 2),
        "avg_holding_days":   round(avg_holding, 1),
        "profit_factor":      round(profit_factor, 2),
        "total_cost_usd":     round(total_turnover, 2),
        "final_nav":          round(portfolio.iloc[-1], 2),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics table."""
    print("\n" + "=" * 55)
    print("  BACKTEST PERFORMANCE REPORT")
    print("=" * 55)
    rows = [
        ("Total Return",       f"{metrics['total_return_pct']:>8.2f} %"),
        ("CAGR",               f"{metrics['cagr_pct']:>8.2f} %"),
        ("Annual Volatility",  f"{metrics['annual_vol_pct']:>8.2f} %"),
        ("Sharpe Ratio",       f"{metrics['sharpe_ratio']:>8.3f}"),
        ("Sortino Ratio",      f"{metrics['sortino_ratio']:>8.3f}"),
        ("Calmar Ratio",       f"{metrics['calmar_ratio']:>8.3f}"),
        ("Max Drawdown",       f"{metrics['max_drawdown_pct']:>8.2f} %"),
        ("Max DD Duration",    f"{metrics['max_dd_duration_days']:>8d} days"),
        ("Win Rate",           f"{metrics['win_rate_pct']:>8.1f} %"),
        ("# Trades",           f"{metrics['num_trades']:>8d}"),
        ("Avg Trade P&L",      f"${metrics['avg_trade_pnl']:>8.2f}"),
        ("Avg Holding",        f"{metrics['avg_holding_days']:>8.1f} days"),
        ("Profit Factor",      f"{metrics['profit_factor']:>8.2f}"),
        ("Total Costs",        f"${metrics['total_cost_usd']:>8,.2f}"),
        ("Final NAV",          f"${metrics['final_nav']:>10,.2f}"),
    ]
    for label, value in rows:
        print(f"  {label:<22} {value}")
    print("=" * 55 + "\n")
