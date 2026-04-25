"""
run_rl_train.py  —  Train the PPO agent on 2019-2022 data.

Usage:
    python run_rl_train.py               # 300K timesteps (default)
    python run_rl_train.py --steps 500000
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    args = p.parse_args()

    from rl.trainer import train
    model = train(total_timesteps=args.steps)
    print(f"\nTraining complete. Model saved to results/rl_model.zip")


if __name__ == "__main__":
    main()
