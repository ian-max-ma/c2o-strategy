"""
Build daily point-in-time borrow tier table.

Input:
    data/prices.parquet
    step3_borrow/borrow_tiers.parquet

Output:
    step3_borrow/daily_borrow_tiers.parquet

Purpose:
    The raw short-interest data is not daily. It is released periodically.
    Since the strategy trades every day, we forward-fill the latest available
    borrow tier for each stock to every trading date.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "step3_borrow"

TRAIN_END = pd.Timestamp("2024-12-31")


def main():
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    borrow = pd.read_parquet(OUT_DIR / "borrow_tiers.parquet")

    prices["date"] = pd.to_datetime(prices["date"])
    borrow["date"] = pd.to_datetime(borrow["date"])

    # Use prices as the daily trading calendar / stock universe.
    daily = (
        prices[["instrument_id", "ticker", "date"]]
        .drop_duplicates()
        .rename(columns={"instrument_id": "stock_id"})
        .sort_values(["stock_id", "date"])
    )

    # Sort before merge_asof.
    daily = daily.sort_values(["date", "stock_id"]).reset_index(drop=True)
    borrow = borrow.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # For each stock and each trading date, attach the most recent borrow tier
    # available on or before that date. This avoids look-ahead bias.
    daily_borrow = pd.merge_asof(
        daily,
        borrow[
            [
                "stock_id",
                "date",
                "dsi",
                "dtcn",
                "ddtcn",
                "borrow_tier",
                "borrow_fee_bps_pa",
                "borrow_fee_daily",
            ]
        ],
        on="date",
        by="stock_id",
        direction="backward",
    )

    # If a stock has no prior short-interest observation yet, default to Tier A.
    daily_borrow["borrow_tier"] = daily_borrow["borrow_tier"].fillna("A")
    daily_borrow["borrow_fee_bps_pa"] = daily_borrow["borrow_fee_bps_pa"].fillna(40)
    daily_borrow["borrow_fee_daily"] = daily_borrow["borrow_fee_daily"].fillna(
        40 / 10000 / 252
    )

    # Keep development window only for now.
    daily_borrow = daily_borrow[daily_borrow["date"] <= TRAIN_END].copy()

    daily_borrow = daily_borrow[
        [
            "stock_id",
            "ticker",
            "date",
            "dsi",
            "dtcn",
            "ddtcn",
            "borrow_tier",
            "borrow_fee_bps_pa",
            "borrow_fee_daily",
        ]
    ].sort_values(["date", "stock_id"])

    output_path = OUT_DIR / "daily_borrow_tiers.parquet"
    daily_borrow.to_parquet(output_path, index=False)

    print("Saved daily borrow tiers to:", output_path)
    print("Output shape:", daily_borrow.shape)
    print()
    print("Borrow tier distribution:")
    print(daily_borrow["borrow_tier"].value_counts(normalize=True).sort_index())
    print()
    print("Date range:", daily_borrow["date"].min(), "->", daily_borrow["date"].max())
    print("Missing ticker rate:", daily_borrow["ticker"].isna().mean())


if __name__ == "__main__":
    main()