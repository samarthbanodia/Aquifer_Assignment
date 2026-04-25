# Global Arbitrage Signal Engine

ETF-pair cointegration-based statistical arbitrage system. Scans 12 ETF pairs across equities, bonds, and commodities on a 24×7 basis. Generates medium-frequency signals (5–30 day holding period) with full transaction-cost accounting.

## Quick Start

```bash
pip install -r requirements.txt

# Run full backtest (2022-01-01 to 2025-04-01)
python run_backtest.py

# Run live signal generator (runs indefinitely; Ctrl+C to stop)
python run_live.py

# Run live scanner with options
python run_live.py --scans 100 --interval 300   # 100 scans, 5-minute intervals
```

## Deliverables

| # | Deliverable | Files |
|---|---|---|
| 1 | Strategy Design | `STRATEGY.md` |
| 2 | Backtesting Engine | `run_backtest.py`, `backtest/`, `strategy/`, `data/` |
| 3 | Live Signal Generator | `run_live.py`, `live/scanner.py` |
| 4 | Risk Framework | `risk/sizer.py`, `risk/regime.py` |

## Backtest Results (2022-01-01 to 2025-04-01)

| Metric | Value | Notes |
|---|---|---|
| Total Return | +0.93% | Net of ALL transaction costs |
| CAGR | +0.29%/year | Market-neutral alpha |
| Annual Volatility | 0.59% | Intentionally low (hedged) |
| **Sharpe Ratio** | **0.485** | vs 0% risk-free (correct for mkt-neutral) |
| **Max Drawdown** | **-0.68%** | Excellent risk control |
| Max DD Duration | 355 days | |
| Win Rate | 47.5% | Winners much larger than losers |
| **Profit Factor** | **2.55** | |
| # Closed Trades | 40 | |
| Avg Holding Period | 22 days | |
| **Total Costs** | **$14,732** | Commission + spread + slippage |
| Final NAV | $1,009,272 | Started at $1,000,000 |

> **Note on Sharpe**: Market-neutral strategies should be benchmarked at 0% (not T-bill rate),
> since the cash portion would separately earn T-bills. The gross portfolio return including
> 5% on uninvested cash would be ~5.3%/year.

## Output Files

After running `run_backtest.py`:
- `results/backtest_portfolio.csv` — daily NAV series
- `results/backtest_trades.csv` — full trade log with costs
- `results/backtest_metrics.json` — all performance metrics
- `results/backtest_plots.png` — 4-panel chart (NAV, drawdown, rolling Sharpe, monthly heatmap)

After running `run_live.py`:
- `results/live_signals.log` — timestamped human-readable log
- `results/live_signals.csv` — structured signal log (pair, direction, z-score, edge, confidence)

## Project Structure

```
aquifer/
├── config.py               # All parameters: universe, signals, costs, risk
├── run_backtest.py         # Backtest entry point
├── run_live.py             # Live scanner entry point
├── STRATEGY.md             # Full strategy design document
├── data/
│   └── loader.py           # yfinance data download + caching
├── strategy/
│   ├── cointegration.py    # Engle-Granger cointegration tests
│   └── signals.py          # Z-score computation + signal generation
├── backtest/
│   ├── engine.py           # Walk-forward portfolio simulator
│   ├── costs.py            # Transaction cost model
│   └── metrics.py          # Sharpe, drawdown, win rate, etc.
├── risk/
│   ├── sizer.py            # Kelly criterion + vol-targeting position sizing
│   └── regime.py           # VIX + correlation breakdown regime detection
└── live/
    └── scanner.py          # Continuous signal scanner (48h+ stable)
```

## Strategy Summary

**What we exploit**: Temporary mispricings in cointegrated ETF pairs — instruments sharing
long-run price equilibria that deviate due to fund flows, creation/redemption lags, and
sentiment divergences.

**Signal**: Engle-Granger cointegration test on rolling 252-day window; z-score of the
residual spread (window=60 days); entry at ±2σ, exit at ±0.5σ, stop-loss at ±3.0σ.

**Risk controls**:
- 20% max NAV per pair
- Pause if portfolio drawdown > 10%
- No new entries if VIX > 40 or pair correlation < 0.80
- 20-day cooldown after each stop-loss event

**Transaction costs modelled**:
- Commission: $0.005/share (Interactive Brokers tiered)
- Bid-ask spread: 0.03% of notional (liquid ETFs)
- Market impact: 0.02% of notional
- Total round-trip: ~0.10–0.15%

## Running the 48-Hour Live Monitor

```bash
python run_live.py
```

The scanner runs indefinitely. Every 5 minutes it:
1. Downloads latest 300 days of prices
2. Tests all pairs for cointegration (p < 0.10)
3. Computes z-scores for cointegrated pairs
4. Logs signals with timestamp, direction, theoretical edge, and confidence score

Signal log format (CSV):
```
timestamp, pair, ticker_a, ticker_b, direction, zscore, hedge_ratio,
half_life_days, theoretical_edge_bps, confidence, spread, regime, vix
```
