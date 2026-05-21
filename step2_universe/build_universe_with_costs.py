"""
Combine eligibility table with borrow costs.

Inputs:
    step2_universe/eligibility_table.parquet
    step3_borrow/daily_borrow_tiers.parquet

Output:
    step2_universe/universe_with_costs.parquet

This is the main output for the Universe & Costs Lead.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEP2_DIR = ROOT / "step2_universe"
STEP3_DIR = ROOT / "step3_borrow"


def main():
    eligibility = pd.read_parquet(STEP2_DIR / "eligibility_table.parquet")
    borrow = pd.read_parquet(STEP3_DIR / "daily_borrow_tiers.parquet")

    eligibility["date"] = pd.to_datetime(eligibility["date"])
    borrow["date"] = pd.to_datetime(borrow["date"])

    universe = eligibility.merge(
        borrow[
            [
                "date",
                "stock_id",
                "borrow_tier",
                "borrow_fee_bps_pa",
                "borrow_fee_daily",
                "dsi",
                "dtcn",
                "ddtcn",
            ]
        ],
        on=["date", "stock_id"],
        how="left",
    )

    # If no borrow data exists, default to Tier A.
    universe["borrow_tier"] = universe["borrow_tier"].fillna("A")
    universe["borrow_fee_bps_pa"] = universe["borrow_fee_bps_pa"].fillna(40)
    universe["borrow_fee_daily"] = universe["borrow_fee_daily"].fillna(
        40 / 10000 / 252
    )

    universe = universe[
        [
            "date",
            "stock_id",
            "ticker",
            "is_sp500",
            "adv20",
            "market_cap",
            "is_eligible",
            "borrow_tier",
            "borrow_fee_bps_pa",
            "borrow_fee_daily",
            "dsi",
            "dtcn",
            "ddtcn",
        ]
    ].sort_values(["date", "stock_id"])

    output_path = STEP2_DIR / "universe_with_costs.parquet"
    universe.to_parquet(output_path, index=False)

    print("Saved universe with costs to:", output_path)
    print("Output shape:", universe.shape)
    print("Date range:", universe["date"].min(), "->", universe["date"].max())
    print()
    print("Eligible ratio:", universe["is_eligible"].mean())
    print()
    print("Borrow tier distribution overall:")
    print(universe["borrow_tier"].value_counts(normalize=True).sort_index())
    print()
    print("Borrow tier distribution among eligible names:")
    print(
        universe.loc[universe["is_eligible"], "borrow_tier"]
        .value_counts(normalize=True)
        .sort_index()
    )


if __name__ == "__main__":
    main()