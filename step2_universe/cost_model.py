"""
Cost model utilities for the C2O strategy.

This module provides reusable cost functions for the portfolio/backtest step.

It covers:
    - Commission cost
    - Base round-trip trading cost
    - Borrow cost for short positions
    - Market impact cost as a function of traded notional / ADV20

The functions are intentionally simple, deterministic, and easy to audit.
"""

import numpy as np
import pandas as pd


TRADING_DAYS = 252

# From project guide / config convention:
# Total round-trip cost = commission + slippage.
ROUNDTRIP_BPS = 4.0

# Borrow schedule from coursework Section 6.
BORROW_FEE_BPS_PA = {
    "A": 40,
    "B": 200,
    "C": 800,
}


def annual_bps_to_daily_rate(annual_bps: float) -> float:
    """
    Convert annual bps to daily decimal rate.

    Example:
        200 bps p.a. -> 0.02 / 252 per day
    """
    return annual_bps / 10000 / TRADING_DAYS


def borrow_fee_daily_from_tier(borrow_tier: str) -> float:
    """
    Return daily borrow fee rate for a borrow tier.

    Tier A: 40 bps p.a.
    Tier B: 200 bps p.a.
    Tier C: 800 bps p.a.
    """
    annual_bps = BORROW_FEE_BPS_PA.get(borrow_tier, 40)
    return annual_bps_to_daily_rate(annual_bps)


def borrow_cost_dollars(short_notional: float, borrow_tier: str) -> float:
    """
    Compute one-day borrow cost in dollars for a short position.

    Parameters:
        short_notional:
            Gross short notional in dollars. Use positive value.
        borrow_tier:
            A, B, or C.

    Returns:
        Dollar borrow cost for one holding day.
    """
    daily_rate = borrow_fee_daily_from_tier(borrow_tier)
    return abs(short_notional) * daily_rate


def base_trading_cost_bps(turnover_notional: float, gross_notional: float) -> float:
    """
    Compute base trading cost in bps using the round-trip cost assumption.

    This is a simple helper:
        cost dollars = turnover_notional * ROUNDTRIP_BPS / 10000

    It returns the cost expressed as bps of gross_notional.

    Parameters:
        turnover_notional:
            Dollar value traded.
        gross_notional:
            Portfolio gross notional, used as denominator.

    Returns:
        Cost in basis points of gross_notional.
    """
    if gross_notional <= 0:
        return 0.0

    cost_dollars = turnover_notional * ROUNDTRIP_BPS / 10000
    return cost_dollars / gross_notional * 10000


def market_impact_bps(
    order_notional: float,
    adv20: float,
    impact_coefficient: float = 10.0,
    max_impact_bps: float = 50.0,
) -> float:
    """
    Estimate market impact in bps using a square-root style model.

    Formula:
        impact_bps = impact_coefficient * sqrt(order_notional / ADV20)

    Interpretation:
        The larger the trade relative to ADV20, the larger the impact.

    Parameters:
        order_notional:
            Dollar value traded in one stock on one day.
        adv20:
            20-day average dollar volume.
        impact_coefficient:
            Scaling parameter. Default 10.0 is deliberately conservative.
        max_impact_bps:
            Cap extreme impact estimates.

    Returns:
        Estimated market impact in bps.
    """
    if pd.isna(order_notional) or pd.isna(adv20) or adv20 <= 0:
        return np.nan

    participation = abs(order_notional) / adv20
    impact = impact_coefficient * np.sqrt(participation)

    return min(impact, max_impact_bps)


def trading_cost_dollars(
    order_notional: float,
    adv20: float,
    include_market_impact: bool = True,
) -> float:
    """
    Compute trading cost in dollars for one stock trade.

    Components:
        1. Base round-trip trading cost
        2. Optional market impact

    Parameters:
        order_notional:
            Dollar value traded.
        adv20:
            20-day average dollar volume.
        include_market_impact:
            Whether to add market impact cost.

    Returns:
        Trading cost in dollars.
    """
    base_cost = abs(order_notional) * ROUNDTRIP_BPS / 10000

    if not include_market_impact:
        return base_cost

    impact = market_impact_bps(order_notional, adv20)

    if pd.isna(impact):
        impact_cost = 0.0
    else:
        impact_cost = abs(order_notional) * impact / 10000

    return base_cost + impact_cost


def total_short_holding_cost_dollars(
    short_notional: float,
    borrow_tier: str,
    order_notional: float = 0.0,
    adv20: float | None = None,
    include_market_impact: bool = True,
) -> float:
    """
    Combine borrow and trading costs for a short position.

    This function is designed for Step 5 portfolio/backtest to call.

    Parameters:
        short_notional:
            Gross short exposure held overnight.
        borrow_tier:
            A/B/C borrow tier.
        order_notional:
            Dollar value traded when entering/exiting/rebalancing.
        adv20:
            20-day average dollar volume.
        include_market_impact:
            Whether to include market impact.

    Returns:
        Total dollar cost.
    """
    borrow_cost = borrow_cost_dollars(short_notional, borrow_tier)

    if adv20 is None:
        trading_cost = abs(order_notional) * ROUNDTRIP_BPS / 10000
    else:
        trading_cost = trading_cost_dollars(
            order_notional=order_notional,
            adv20=adv20,
            include_market_impact=include_market_impact,
        )

    return borrow_cost + trading_cost


if __name__ == "__main__":
    # Simple sanity checks.
    print("Borrow daily rate Tier A:", borrow_fee_daily_from_tier("A"))
    print("Borrow daily rate Tier B:", borrow_fee_daily_from_tier("B"))
    print("Borrow daily rate Tier C:", borrow_fee_daily_from_tier("C"))

    print("Borrow cost on $1m Tier B:", borrow_cost_dollars(1_000_000, "B"))
    print(
        "Market impact bps for $1m trade on $100m ADV:",
        market_impact_bps(order_notional=1_000_000, adv20=100_000_000),
    )
    print(
        "Trading cost dollars:",
        trading_cost_dollars(order_notional=1_000_000, adv20=100_000_000),
    )