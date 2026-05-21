"""
Build cost schedule table.

Output:
    step2_universe/cost_schedule.csv

This file documents the cost assumptions used by the Universe & Costs module.
The portfolio/backtest step can read this table or import cost_model.py directly.
"""

from pathlib import Path

import pandas as pd

from cost_model import (
    ROUNDTRIP_BPS,
    BORROW_FEE_BPS_PA,
    annual_bps_to_daily_rate,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "step2_universe"


def main():
    rows = []

    rows.append(
        {
            "cost_component": "base_roundtrip_trading_cost",
            "tier": "",
            "rate_bps": ROUNDTRIP_BPS,
            "rate_daily_decimal": "",
            "applies_to": "turnover_notional",
            "frequency": "per round trip",
            "notes": "Commission + slippage base assumption",
        }
    )

    rows.append(
        {
            "cost_component": "market_impact",
            "tier": "",
            "rate_bps": "",
            "rate_daily_decimal": "",
            "applies_to": "order_notional / ADV20",
            "frequency": "per trade",
            "notes": "impact_bps = 10 * sqrt(order_notional / ADV20), capped at 50 bps",
        }
    )

    for tier, annual_bps in BORROW_FEE_BPS_PA.items():
        rows.append(
            {
                "cost_component": "borrow_fee",
                "tier": tier,
                "rate_bps": annual_bps,
                "rate_daily_decimal": annual_bps_to_daily_rate(annual_bps),
                "applies_to": "gross_short_notional",
                "frequency": "daily",
                "notes": "Annual borrow bps divided by 252 trading days",
            }
        )

    cost_schedule = pd.DataFrame(rows)

    output_path = OUT_DIR / "cost_schedule.csv"
    cost_schedule.to_csv(output_path, index=False)

    print("Saved cost schedule to:", output_path)
    print(cost_schedule)


if __name__ == "__main__":
    main()