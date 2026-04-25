"""
RL training pipeline.

1. Downloads 2019-2022 prices (pre-test training window).
2. For every pair that was cointegrated in at least one 252-day rolling window,
   builds a feature-enriched DataFrame.
3. Trains a PPO agent across all pairs (multi-episode curriculum).
4. Saves model + VecNormalize stats to results/.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from data.loader import download_prices, get_all_tickers
from strategy.cointegration import test_pair_cointegration
from strategy.signals import compute_spread, compute_zscore
from rl.environment import PairSpreadEnv

logger = logging.getLogger(__name__)

TRAIN_START = "2019-01-01"
TRAIN_END   = "2021-12-31"
MODEL_PATH  = "results/rl_model"
VECNORM_PATH = "results/rl_vecnorm.pkl"


# ── feature engineering ────────────────────────────────────────────────────────

def build_pair_dataset(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    zscore_window: int = 60,
) -> pd.DataFrame | None:
    """Build a fully-featured episode DataFrame for one pair."""
    result = test_pair_cointegration(prices, ticker_a, ticker_b, significance=0.10)
    if result is None or not result.is_cointegrated:
        return None

    spread = compute_spread(prices[[ticker_a, ticker_b]], result)
    zscore = compute_zscore(spread, zscore_window)

    # z-score velocity and acceleration
    z_vel   = zscore.diff(5).fillna(0) / 5
    z_accel = z_vel.diff(1).fillna(0)

    # spread daily return (normalized by spread std over window)
    spread_std   = spread.rolling(zscore_window, min_periods=zscore_window // 2).std()
    spread_return = spread.diff().fillna(0) / spread_std.replace(0, np.nan).fillna(1)

    # 20-day rolling return correlation
    ret_a = np.log(prices[ticker_a]).diff()
    ret_b = np.log(prices[ticker_b]).diff()
    corr  = ret_a.rolling(20).corr(ret_b).fillna(0.9)

    # VIX
    vix = prices["^VIX"] if "^VIX" in prices.columns else pd.Series(20.0, index=prices.index)

    df = pd.DataFrame({
        "zscore":        zscore,
        "z_vel":         z_vel,
        "z_accel":       z_accel,
        "spread_return": spread_return,
        "vix":           vix,
        "corr_20d":      corr,
    }, index=prices.index).dropna()

    return df if len(df) >= zscore_window + 10 else None


def collect_training_datasets(prices: pd.DataFrame) -> List[pd.DataFrame]:
    """
    For each pair in the universe that shows cointegration on the training data,
    build one episode DataFrame. Also build sliding sub-windows for data augmentation.
    """
    datasets = []
    for a, b in config.ETF_PAIRS:
        if a not in prices.columns or b not in prices.columns:
            continue
        df = build_pair_dataset(prices, a, b)
        if df is None or len(df) < 100:
            continue

        logger.info("Training pair: %s/%s  (%d days)", a, b, len(df))
        datasets.append(df)

        # Sliding-window augmentation: 3 overlapping sub-episodes
        n = len(df)
        if n > 300:
            half = n // 2
            datasets.append(df.iloc[:half + 30])
            datasets.append(df.iloc[half - 30:])

    logger.info("Total training datasets: %d", len(datasets))
    return datasets


# ── environment factory ────────────────────────────────────────────────────────

def make_env_fn(datasets: List[pd.DataFrame], idx: int, cost_multiplier: float = 1.0):
    """Return a callable that creates a PairSpreadEnv for episode `idx`."""
    def _make():
        data = datasets[idx % len(datasets)]
        return PairSpreadEnv(data, one_way_cost_pct=0.0005 * cost_multiplier)
    return _make


# ── train ──────────────────────────────────────────────────────────────────────

def train(total_timesteps: int = 300_000) -> PPO:
    Path("results").mkdir(exist_ok=True)

    logger.info("Downloading training data: %s to %s", TRAIN_START, TRAIN_END)
    all_tickers = get_all_tickers(config.ETF_PAIRS) + config.REGIME_TICKERS
    prices = download_prices(all_tickers, TRAIN_START, TRAIN_END)

    datasets = collect_training_datasets(prices)
    if not datasets:
        raise RuntimeError("No training datasets found — check pair universe.")

    # Create N parallel envs (one per dataset, cycle through them)
    n_envs = min(len(datasets), 4)
    # 10x cost penalty during training forces the agent to be selective
    # (real backtest still uses actual costs)
    env_fns = [make_env_fn(datasets, i, cost_multiplier=10.0) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # entropy bonus encourages exploration
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    logger.info("Training PPO for %d timesteps...", total_timesteps)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    logger.info("Model saved: %s", MODEL_PATH)
    logger.info("VecNorm saved: %s", VECNORM_PATH)

    return model


# ── load ───────────────────────────────────────────────────────────────────────

def load_model() -> PPO:
    """Load saved PPO model, patching SB3's torch.load to use weights_only=False."""
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.zip — run: python run_rl_train.py"
        )
    # SB3 save_util imports torch as `th`; patch that local reference
    import stable_baselines3.common.save_util as sb3_util
    _orig = sb3_util.th.load
    sb3_util.th.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        model = PPO.load(MODEL_PATH)
    finally:
        sb3_util.th.load = _orig
    logger.info("RL model loaded from %s", MODEL_PATH)
    return model
