"""
run_live.py — entry point for the continuous live signal generator.

Usage:
    python run_live.py                    # runs indefinitely (48h+)
    python run_live.py --scans 10         # run 10 scans then exit
    python run_live.py --interval 60      # override scan interval to 60s

Logs are written to:
    results/live_signals.log   (human-readable)
    results/live_signals.csv   (structured, one row per signal)
"""

import argparse
import logging
import sys

from live.scanner import LiveScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser(description="Live Arbitrage Signal Generator")
    p.add_argument("--scans", type=int, default=None, help="Max number of scans (default: unlimited)")
    p.add_argument("--interval", type=int, default=None, help="Override scan interval in seconds")
    return p.parse_args()


def main():
    args = parse_args()

    import config
    if args.interval is not None:
        config.SCAN_INTERVAL_SEC = args.interval
        logging.getLogger(__name__).info("Scan interval overridden to %ds", args.interval)

    scanner = LiveScanner()

    try:
        scanner.run(max_scans=args.scans)
    except KeyboardInterrupt:
        print("\nScanner stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
