"""
Global Arbitrage Signal Engine — Configuration
All parameters for strategy, backtest, costs, and risk.
"""

# ─── Trading Universe ────────────────────────────────────────────────────────
# Pairs chosen for strong economic co-movement and stable cointegration.
# (leg_A, leg_B)  —  we trade spread = log(A) - β·log(B)
ETF_PAIRS = [
    # ── Tier-1: Near-duplicate ETF pairs (same index, different issuers) ──────
    # These have stable cointegration due to shared benchmark.
    # Lower edge-per-trade but very reliable mean reversion.
    ("HYG", "JNK"),    # iShares vs SPDR high-yield bond — same universe, tiny diffs
    ("EEM", "VWO"),    # iShares vs Vanguard EM — same MSCI index, slight drift
    ("EFA", "VEA"),    # iShares vs Vanguard developed-ex-US
    ("SPY", "IVV"),    # Two S&P 500 ETFs — creation/redemption basis trades

    # ── Tier-2: Related-sector pairs with economic cointegration ──────────────
    # Stronger edge-per-trade, higher half-life risk.
    ("USO", "XLE"),    # Oil commodity ETF vs energy-sector equity ETF
    ("GLD", "SLV"),    # Gold vs silver — precious-metals ratio mean reversion
    ("TLT", "IEF"),    # 20yr vs 7-10yr Treasuries — yield-curve duration spread
    ("QQQ", "XLK"),    # Nasdaq-100 vs tech-sector SPDR — weight-adjusted drift
    ("XLF", "KRE"),    # Broad financials vs regional-bank sub-sector
    ("XLV", "IBB"),    # Healthcare sector vs biotech — sub-sector premium/discount
    ("XBI", "IBB"),    # Two biotech ETFs — different index construction
    ("GLD", "GDX"),    # Gold spot vs gold-miners — equity-beta leverage on gold

    # ── Tier-3: Removed — structural-break risk too high ─────────────────────
    # XLK/SMH removed: 2023-24 AI boom caused permanent semiconductor outperformance
    # XLE/OIH removed: oil-services vs integrated-energy structural divergence
]

# Additional macro instruments for the regime filter
REGIME_TICKERS = ["^VIX", "^GSPC", "GLD", "TLT", "^TNX"]

# ─── Backtest Parameters ─────────────────────────────────────────────────────
BACKTEST_START = "2022-01-01"
BACKTEST_END   = "2025-04-01"
INITIAL_CAPITAL = 100_000            # USD

# ─── Signal Parameters ───────────────────────────────────────────────────────
COINTEGRATION_LOOKBACK = 252         # trading days for coint test
ZSCORE_WINDOW          = 60          # rolling window for z-score
ENTRY_ZSCORE           = 2.0         # open position when |z| crosses this
EXIT_ZSCORE            = 0.5         # close position when |z| falls below this
STOP_ZSCORE            = 3.0         # hard stop when |z| exceeds this
REBALANCE_FREQ         = 63          # re-test cointegration every ~quarter

# ─── Transaction Cost Parameters ─────────────────────────────────────────────
# Modelled on Interactive Brokers tiered pricing for liquid ETFs
COMMISSION_PER_SHARE   = 0.005       # $0.005/share each way
MIN_COMMISSION         = 1.00        # minimum per order
BID_ASK_SPREAD_PCT     = 0.0003      # 0.03% half-spread (liquid ETFs)
SLIPPAGE_PCT           = 0.0002      # 0.02% market-impact estimate
# Total round-trip drag estimate: ~0.1–0.15% of notional per trade

# ─── Risk Parameters ─────────────────────────────────────────────────────────
MAX_PAIR_WEIGHT        = 0.20        # max 20% NAV in any single pair
PORTFOLIO_STOP_DD      = 0.10        # pause ALL entries if drawdown > 10%
PAIR_STOP_DD           = 0.05        # pause a pair if its own drawdown > 5%
VIX_PAUSE_THRESHOLD    = 40          # no new entries when VIX > 40
VIX_REDUCE_THRESHOLD   = 30          # halve sizes when VIX > 30
KELLY_FRACTION         = 0.50        # half-Kelly safety factor
TARGET_ANNUAL_VOL      = 0.10        # 10% annual portfolio volatility target

# ─── Live Scanner Parameters ─────────────────────────────────────────────────
SCAN_INTERVAL_SEC      = 300         # every 5 minutes
LIVE_LOOKBACK_DAYS     = 300         # days of history fetched each scan
SIGNAL_LOG_FILE        = "results/live_signals.log"
SIGNAL_CSV_FILE        = "results/live_signals.csv"
MIN_CONFIDENCE         = 0.50        # minimum confidence score to log signal
