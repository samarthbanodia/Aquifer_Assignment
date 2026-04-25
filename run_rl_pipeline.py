"""
run_rl_pipeline.py  —  Train PPO then immediately run RL backtest in one process.

Trains on 2019-2021 data, tests on 2022-2025 (same window as classical backtest).
Prints side-by-side comparison and saves results/rl_vs_classical.png.

Usage:
    python run_rl_pipeline.py
    python run_rl_pipeline.py --steps 500000
"""

import argparse
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from data.loader import download_prices, get_all_tickers
from backtest.costs import DEFAULT_COST_MODEL
from backtest.engine import PairPosition
from backtest.metrics import calculate_metrics, print_metrics
from strategy.cointegration import select_cointegrated_pairs
from strategy.signals import compute_spread, compute_zscore
from rl.trainer import collect_training_datasets, make_env_fn, TRAIN_START, TRAIN_END
from rl.agent import RLAgent
from rl.environment import PairSpreadEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

Path("results").mkdir(exist_ok=True)


# ── Training ───────────────────────────────────────────────────────────────────

def train_agent(total_timesteps: int) -> tuple[PPO, VecNormalize]:
    logger.info("Downloading training data: %s to %s", TRAIN_START, TRAIN_END)
    all_tickers = get_all_tickers(config.ETF_PAIRS) + config.REGIME_TICKERS
    prices = download_prices(all_tickers, TRAIN_START, TRAIN_END)

    datasets = collect_training_datasets(prices)
    if not datasets:
        raise RuntimeError("No training datasets found.")

    n_envs = min(len(datasets), 4)
    env_fns = [make_env_fn(datasets, i, cost_multiplier=10.0) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    model = PPO(
        "MlpPolicy", vec_env, verbose=1,
        n_steps=512, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    logger.info("Training PPO for %d timesteps...", total_timesteps)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    logger.info("Training complete.")
    return model, vec_env


# ── RL Backtest ────────────────────────────────────────────────────────────────

def run_rl_backtest(agent: RLAgent, prices: pd.DataFrame) -> dict:
    dates = prices.index
    capital = config.INITIAL_CAPITAL
    portfolio_values = []
    trade_log = []
    pair_meta: dict = {}
    positions: dict = {}
    peak_value = capital
    strategy_paused = False

    for day_idx, date in enumerate(dates):
        prices_today = prices.loc[date]

        # Quarterly cointegration rebalance
        if day_idx >= config.COINTEGRATION_LOOKBACK and day_idx % config.REBALANCE_FREQ == 0:
            window = prices.iloc[max(0, day_idx - config.COINTEGRATION_LOOKBACK): day_idx]
            active = select_cointegrated_pairs(window, config.ETF_PAIRS, significance=0.10)
            new_meta = {}
            for cr in active:
                a, b = cr.pair
                name = f"{a}/{b}"
                new_meta[name] = {"tickers": (a, b), "coint": cr}
                if name not in positions:
                    notional = min(capital * config.MAX_PAIR_WEIGHT,
                                   capital / max(len(new_meta), 1))
                    positions[name] = PairPosition(name, notional, a, b)

            # Close pairs that lost cointegration
            for name, pos in positions.items():
                if pos.signal != 0 and name not in new_meta:
                    a, b = pos.ticker_a, pos.ticker_b
                    pa, pb = prices_today.get(a, np.nan), prices_today.get(b, np.nan)
                    if not (np.isnan(pa) or np.isnan(pb)):
                        cost = (DEFAULT_COST_MODEL.total_cost(pa, pos.shares_a) +
                                DEFAULT_COST_MODEL.total_cost(pb, pos.shares_b))
                        pnl = (pos.shares_a * (pa - pos.entry_prices.get(a, pa)) +
                               pos.shares_b * (pb - pos.entry_prices.get(b, pb)) - cost)
                        capital += pnl
                    pos.signal = pos.shares_a = pos.shares_b = 0
            pair_meta = new_meta
            logger.info("Day %d (%s): %d cointegrated pairs", day_idx, date.date(), len(pair_meta))

        if not pair_meta:
            portfolio_values.append(capital)
            continue

        vix = float(prices.loc[date, "^VIX"]) if "^VIX" in prices.columns else 20.0
        size_factor = 0.5 if vix >= config.VIX_REDUCE_THRESHOLD else 1.0
        if vix >= config.VIX_PAUSE_THRESHOLD:
            strategy_paused = True
        if vix < config.VIX_REDUCE_THRESHOLD:
            strategy_paused = False

        for name, meta in pair_meta.items():
            a, b = meta["tickers"]
            if a not in prices.columns or b not in prices.columns:
                continue

            pos = positions[name]
            coint = meta["coint"]

            hist_start = max(0, day_idx - config.ZSCORE_WINDOW - 10)
            win = prices.iloc[hist_start: day_idx + 1]
            if len(win) < config.ZSCORE_WINDOW:
                continue

            try:
                spread_s = compute_spread(win[[a, b]], coint)
                zscore_s = compute_zscore(spread_s, config.ZSCORE_WINDOW)
            except Exception:
                continue

            z_vals = zscore_s.dropna()
            if z_vals.empty:
                continue

            z = float(z_vals.iloc[-1])
            z_vel = float(z_vals.diff(5).iloc[-1]) / 5 if len(z_vals) >= 6 else 0.0
            z_accel = float(z_vals.diff(5).diff(1).iloc[-1]) if len(z_vals) >= 7 else 0.0

            spread_std = float(spread_s.dropna().rolling(config.ZSCORE_WINDOW).std().iloc[-1])
            spread_ret = float(spread_s.diff().iloc[-1]) / (spread_std if spread_std > 0 else 1.0)

            ret_a = np.log(prices[a]).diff().iloc[max(0, day_idx-20): day_idx+1]
            ret_b = np.log(prices[b]).diff().iloc[max(0, day_idx-20): day_idx+1]
            corr = float(ret_a.corr(ret_b)) if len(ret_a) >= 5 else 0.9
            if np.isnan(corr):
                corr = 0.9

            obs = agent.build_obs(name, z, z_vel, z_accel, vix, corr)
            new_signal = agent.predict(name, obs)

            if strategy_paused and new_signal != 0 and pos.signal == 0:
                new_signal = 0
            if new_signal != 0 and pos.signal == 0 and corr < 0.80:
                new_signal = 0

            agent.update_state(name, new_signal, spread_ret)

            pa = float(prices_today.get(a, np.nan))
            pb = float(prices_today.get(b, np.nan))
            if np.isnan(pa) or np.isnan(pb):
                continue

            old_signal = pos.signal

            if old_signal != 0 and new_signal != old_signal:
                ca = DEFAULT_COST_MODEL.total_cost(pa, pos.shares_a)
                cb = DEFAULT_COST_MODEL.total_cost(pb, pos.shares_b)
                pnl = (pos.shares_a * (pa - pos.entry_prices.get(a, pa)) +
                       pos.shares_b * (pb - pos.entry_prices.get(b, pb)) - ca - cb)
                capital += pnl
                trade_log.append({"date": date, "pair": name, "action": "CLOSE",
                                   "pnl": pnl, "cost": ca + cb})
                pos.shares_a = pos.shares_b = 0.0
                pos.signal = 0

            if new_signal != 0 and pos.signal == 0:
                notional = pos.target_notional * size_factor
                sa = (notional / 2) / pa
                sb = (notional / 2) / pb * abs(coint.hedge_ratio)
                pos.shares_a = +sa if new_signal == 1 else -sa
                pos.shares_b = -sb if new_signal == 1 else +sb
                ca = DEFAULT_COST_MODEL.total_cost(pa, pos.shares_a)
                cb = DEFAULT_COST_MODEL.total_cost(pb, pos.shares_b)
                capital -= (ca + cb)
                pos.signal = new_signal
                pos.entry_prices = {a: pa, b: pb}
                trade_log.append({"date": date, "pair": name, "action": "OPEN",
                                   "signal": new_signal, "cost": ca + cb})

        portfolio_values.append(capital)

    portfolio = pd.Series(portfolio_values, index=dates, name="rl_portfolio")
    trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    return {"portfolio": portfolio, "trades": trades}


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_comparison(cl_nav: pd.Series, rl_nav: pd.Series, m_cl: dict, m_rl: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Classical Stat Arb vs RL-Enhanced Strategy (2022-2025)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(cl_nav.index, (cl_nav / cl_nav.iloc[0] - 1) * 100,
            color="#2196F3", linewidth=1.5,
            label=f"Classical  Sharpe={m_cl['sharpe_ratio']:.2f}  DD={m_cl['max_drawdown_pct']:.2f}%")
    ax.plot(rl_nav.index, (rl_nav / rl_nav.iloc[0] - 1) * 100,
            color="#4CAF50", linewidth=1.5,
            label=f"RL Agent   Sharpe={m_rl['sharpe_ratio']:.2f}  DD={m_rl['max_drawdown_pct']:.2f}%")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Cumulative Return (%)")
    ax.set_ylabel("Return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.legend(fontsize=8)

    ax = axes[1]
    metrics_to_show = ["total_return_pct", "sharpe_ratio", "win_rate_pct", "profit_factor"]
    labels = ["Total Return %", "Sharpe", "Win Rate %", "Profit Factor"]
    c_vals = [m_cl[k] for k in metrics_to_show]
    r_vals = [m_rl[k] for k in metrics_to_show]
    x = np.arange(len(labels))
    w = 0.35
    bars_c = ax.bar(x - w/2, c_vals, w, label="Classical", color="#2196F3", alpha=0.85)
    bars_r = ax.bar(x + w/2, r_vals, w, label="RL Agent",  color="#4CAF50", alpha=0.85)
    ax.bar_label(bars_c, fmt="%.2f", fontsize=7, padding=2)
    ax.bar_label(bars_r, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Key Metrics Comparison")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()

    plt.tight_layout()
    out = "results/rl_vs_classical.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Comparison plot saved: %s", out)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    args = p.parse_args()

    # 1. Train
    model, vec_env = train_agent(args.steps)
    agent = RLAgent(model, vec_env)

    # 2. Load test prices
    all_tickers = get_all_tickers(config.ETF_PAIRS) + config.REGIME_TICKERS
    logger.info("Loading test data: %s to %s", config.BACKTEST_START, config.BACKTEST_END)
    prices = download_prices(all_tickers, config.BACKTEST_START, config.BACKTEST_END)

    # 3. Run RL backtest
    logger.info("Running RL backtest on test set...")
    rl_result = run_rl_backtest(agent, prices)
    rl_nav   = rl_result["portfolio"]
    rl_trades = rl_result["trades"]
    rl_metrics = calculate_metrics(rl_nav, rl_trades, risk_free_rate=0.0)

    print("\n=== RL AGENT BACKTEST (2022-2025) ===")
    print_metrics(rl_metrics)

    rl_nav.to_csv("results/rl_backtest_portfolio.csv")
    if not rl_trades.empty:
        rl_trades.to_csv("results/rl_backtest_trades.csv", index=False)
    with open("results/rl_backtest_metrics.json", "w") as f:
        json.dump(rl_metrics, f, indent=2)

    # 4. Compare with classical
    cl_nav = pd.read_csv("results/backtest_portfolio.csv",
                         index_col=0, parse_dates=True).squeeze()
    cl_trades = pd.read_csv("results/backtest_trades.csv")
    cl_metrics = calculate_metrics(cl_nav, cl_trades, risk_free_rate=0.0)

    print("\n=== CLASSICAL STRATEGY (2022-2025) ===")
    print_metrics(cl_metrics)

    print("\n=== IMPROVEMENT SUMMARY ===")
    print(f"  {'Metric':<24} {'Classical':>10} {'RL Agent':>10} {'Delta':>10}")
    print("  " + "-" * 56)
    for key in ["total_return_pct", "cagr_pct", "sharpe_ratio",
                "max_drawdown_pct", "win_rate_pct", "profit_factor",
                "num_trades", "avg_holding_days", "total_cost_usd"]:
        c = cl_metrics[key]
        r = rl_metrics[key]
        d = r - c
        sign = "+" if d >= 0 else ""
        print(f"  {key:<24} {c:>10.3f} {r:>10.3f} {sign}{d:>9.3f}")

    plot_comparison(cl_nav, rl_nav, cl_metrics, rl_metrics)
    logger.info("RL pipeline complete.")


if __name__ == "__main__":
    main()
