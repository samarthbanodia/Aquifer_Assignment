"""
Realistic transaction cost model.

Components:
  1. Commission  — $0.005/share, min $1.00 per order
  2. Bid-ask spread — 0.03% half-spread (paid on entry AND exit)
  3. Market impact / slippage — 0.02% of notional per trade
"""

from dataclasses import dataclass


@dataclass
class CostModel:
    commission_per_share: float = 0.005
    min_commission: float = 1.00
    bid_ask_half_spread_pct: float = 0.0003    # 3 bps
    slippage_pct: float = 0.0002               # 2 bps

    def commission(self, shares: float) -> float:
        return max(self.min_commission, abs(shares) * self.commission_per_share)

    def spread_cost(self, notional: float) -> float:
        """Full bid-ask spread paid when crossing the market."""
        return abs(notional) * self.bid_ask_half_spread_pct * 2

    def slippage(self, notional: float) -> float:
        return abs(notional) * self.slippage_pct

    def total_cost(self, price: float, shares: float) -> float:
        """Total one-way transaction cost in dollars."""
        notional = price * abs(shares)
        return self.commission(shares) + self.spread_cost(notional) + self.slippage(notional)

    def round_trip_cost(self, price: float, shares: float) -> float:
        """Cost for opening AND closing a position (both legs)."""
        return 2 * self.total_cost(price, shares)

    def cost_pct(self, price: float, shares: float) -> float:
        """One-way cost as a fraction of notional."""
        notional = price * abs(shares)
        if notional == 0:
            return 0.0
        return self.total_cost(price, shares) / notional


DEFAULT_COST_MODEL = CostModel()
