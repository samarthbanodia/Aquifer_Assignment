# Global Arbitrage Signal Engine

A statistical arbitrage system built around ETF pairs trading. Exploits transient mispricings between economically cointegrated ETF pairs using Engle-Granger cointegration, z-score mean-reversion signals, and a full walk-forward backtesting framework with realistic transaction costs.

---

## Results (2022–2025, $100K starting capital)

| Metric | Value |
|---|---|
| Total Return | +0.87% net |
| CAGR | +0.27% |
| Sharpe Ratio | 0.454 |
| Max Drawdown | -0.70% |
| Win Rate | 47.5% |
| # Trades | 40 |
| Avg Holding Period | 22.0 days |
| Profit Factor | 2.50 |
| Total Transaction Costs | $1,533 |
| Final NAV | $100,867 |

> Sharpe is calculated at 0% risk-free rate (appropriate for market-neutral strategies). All costs are modelled on Interactive Brokers tiered pricing.

---

## Strategy Overview

**Inefficiency exploited:** ETF pairs tracking the same or closely related indices periodically diverge due to fund flows, creation/redemption arbitrage, and short-term liquidity imbalances. These divergences mean-revert predictably.

**Edge:** The spread between two cointegrated ETFs follows a stationary process. When the z-score of the spread exceeds ±2.0 standard deviations, it has historically reverted to zero within 10–25 days — generating a predictable, hedged return independent of market direction.

**Edge decay:** Arbitrage capital inflows compress spreads over time. The strategy defends against this by (1) quarterly re-testing cointegration with p < 0.10, (2) filtering on half-life < 126 days, and (3) a 20-day cooldown after stop-loss exits to avoid re-entering a broken pair.

### Signal Logic

```
Entry long spread:   z-score < -2.0  AND  20-day correlation > 0.80
Entry short spread:  z-score > +2.0  AND  20-day correlation > 0.80
Exit:                |z-score| < 0.5
Stop-loss:           |z-score| > 3.0  →  20-day cooldown
```

### Pair Universe (12 pairs across 3 tiers)

| Tier | Pairs | Rationale |
|---|---|---|
| 1 — Near-duplicate ETFs | HYG/JNK, EEM/VWO, EFA/VEA, SPY/IVV | Same index, different issuers — extremely stable cointegration |
| 2 — Related sectors | USO/XLE, GLD/SLV, TLT/IEF, QQQ/XLK, XLF/KRE, XLV/IBB, XBI/IBB, GLD/GDX | Economic co-movement with mean-reverting basis |
| 3 — Removed | ~~XLK/SMH~~, ~~XLE/OIH~~ | Structural breaks: AI boom (2023-24) and oil-services divergence |

---

## Architecture

```
aquifer/
├── config.py                  # All parameters (universe, thresholds, costs, risk)
│
├── data/
│   └── loader.py              # yfinance downloader with CSV cache
│
├── strategy/
│   ├── cointegration.py       # Engle-Granger test → CointResult (p-value, hedge ratio, half-life)
│   └── signals.py             # Spread computation, z-score, signal generation
│
├── backtest/
│   ├── engine.py              # Walk-forward simulator (no lookahead bias)
│   ├── costs.py               # Commission + bid-ask + slippage model
│   └── metrics.py             # Sharpe, drawdown, win rate, holding period, etc.
│
├── risk/
│   ├── sizer.py               # Half-Kelly + volatility-targeting position sizing
│   └── regime.py              # VIX + correlation + vol-of-vol regime detector
│
├── live/
│   └── scanner.py             # LiveScanner: scans every 5 min, logs signals to CSV/log
│
├── rl/                        # Experimental RL extension (see note below)
│   ├── environment.py         # Gymnasium PairSpreadEnv
│   ├── trainer.py             # PPO training pipeline
│   └── agent.py               # RLAgent inference wrapper
│
├── run_backtest.py            # Main backtest entry point
├── run_live.py                # Live signal scanner entry point
├── run_rl_pipeline.py         # RL train + backtest (experimental)
└── results/                   # Output: CSVs, JSON metrics, plots
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install yfinance pandas numpy scipy statsmodels stable-baselines3 gymnasium matplotlib
```

### 2. Run the backtest

```bash
python run_backtest.py
```

Outputs:
- `results/backtest_portfolio.csv` — daily NAV
- `results/backtest_trades.csv` — trade log
- `results/backtest_metrics.json` — full metrics
- `results/backtest_plots.png` — NAV, drawdown, rolling Sharpe, monthly heatmap

### 3. Run the live scanner (48-hour mode)

```bash
python run_live.py
```

Scans every 5 minutes. Outputs:
- `results/live_signals.log` — timestamped human-readable log
- `results/live_signals.csv` — machine-readable signal history

Each signal row contains: `timestamp, pair, direction, zscore, hedge_ratio, half_life_days, theoretical_edge_bps, confidence, spread, regime, vix, cointegration_pvalue`

For a fixed number of scans (useful for testing):
```bash
python run_live.py --scans 3
```

---

## Transaction Cost Model

Modelled on Interactive Brokers tiered pricing for liquid ETFs:

| Cost Component | Value |
|---|---|
| Commission | $0.005/share (min $1.00/order) |
| Bid-ask spread | 0.03% of notional (half-spread) |
| Slippage | 0.02% of notional (market impact) |
| **Round-trip total** | **~0.10–0.15% of notional** |

---

## Risk Framework

| Control | Parameter | Trigger |
|---|---|---|
| VIX pause | `VIX_PAUSE_THRESHOLD = 40` | No new entries when VIX > 40 |
| VIX size reduce | `VIX_REDUCE_THRESHOLD = 30` | Half position sizes when VIX > 30 |
| Portfolio stop | `PORTFOLIO_STOP_DD = 10%` | Pause all entries if portfolio drawdown > 10% |
| Correlation filter | 20-day rolling corr | Block entry if pair correlation < 0.80 |
| Stop-loss cooldown | `STOP_ZSCORE = 3.0` | 20-day re-entry ban after stop triggered |
| Max pair weight | `MAX_PAIR_WEIGHT = 20%` | No single pair exceeds 20% of NAV |
| Position sizing | Half-Kelly + vol targeting | `KELLY_FRACTION = 0.50`, target 10% annual vol |

---

## Cointegration Methodology

Uses the Engle-Granger two-step procedure:

1. **OLS regression:** `log(A) = α + β·log(B) + ε`
2. **ADF test on residuals ε:** reject unit root at p < 0.10
3. **Half-life filter:** `2 < HL < 126 days` (too fast = noise; too slow = stale signal)

Pairs are re-tested every quarter (63 trading days) on a rolling 252-day window. Pairs that fail re-qualification have open positions closed at market.

**Key papers:**
- Gatev, Goetzmann & Rouwenhorst (2006) — "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
- Avellaneda & Lee (2008) — "Statistical Arbitrage in the U.S. Equities Market"
- Engle & Granger (1987) — "Co-integration and Error Correction: Representation, Estimation, and Testing"

---

## Experimental: RL Extension

The `rl/` directory contains a PPO-based agent (stable-baselines3) that was trained to replace fixed z-score thresholds with a learned policy. After 5 training configurations (500K steps each), the RL agent consistently underperformed:

- **Classical:** 40 trades, +0.87%, Sharpe 0.454
- **Best RL attempt:** 533 trades, -20.7%, Sharpe -5.03

**Root cause:** With ~30 qualifying trades per 3-year training window, PPO lacks sufficient experience to learn that holding through 15-20 days of noisy daily PnL produces a better outcome than cutting losses early. The reward signal is too sparse for the optimizer to discover patient mean-reversion. Would require 10M+ timesteps and a much larger universe to be viable.

The RL code is kept for reference but is not used in production.

---

## Key Parameters (config.py)

```python
INITIAL_CAPITAL      = 100_000     # USD
BACKTEST_START       = "2022-01-01"
BACKTEST_END         = "2025-04-01"
ENTRY_ZSCORE         = 2.0
EXIT_ZSCORE          = 0.5
STOP_ZSCORE          = 3.0
COINTEGRATION_LOOKBACK = 252       # days
ZSCORE_WINDOW        = 60          # days
REBALANCE_FREQ       = 63          # days (~quarterly)
MAX_PAIR_WEIGHT      = 0.20
VIX_PAUSE_THRESHOLD  = 40
VIX_REDUCE_THRESHOLD = 30
SCAN_INTERVAL_SEC    = 300         # live scanner: every 5 minutes
```
