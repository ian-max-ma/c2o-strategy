"""
Build borrow tiers for the short leg.

This script creates a point-in-time borrow tier table using short interest data.
Output:
    step3_borrow/borrow_tiers.parquet
"""

from pathlib import Path

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "step3_borrow"

TRAIN_END = pd.Timestamp("2024-12-31")


def load_data():
    """Load prices and short interest data."""
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    si = pd.read_parquet(DATA_DIR / "short_interest_transfo.parquet")

    prices["date"] = pd.to_datetime(prices["date"])
    si["date"] = pd.to_datetime(si["date"])

    return prices, si


def add_ticker(si, prices):
    """Map stock_id from short interest data to ticker from prices."""
    id_to_ticker = (
        prices[["instrument_id", "ticker", "date"]]
        .dropna(subset=["instrument_id", "ticker"])
        .sort_values("date")
        .drop_duplicates("instrument_id", keep="last")
        .rename(columns={"instrument_id": "stock_id"})
        [["stock_id", "ticker"]]
    )

    si_with_ticker = si.merge(id_to_ticker, on="stock_id", how="left")
    return si_with_ticker


def compute_thresholds(si_with_ticker):
    """
    Compute borrow stress thresholds using only the development window.

    We avoid looking beyond TRAIN_END to prevent look-ahead bias.
    """
    si_train = si_with_ticker[si_with_ticker["date"] <= TRAIN_END].copy()

    thresholds = {
        "dsi_90": si_train["dsi"].quantile(0.90),
        "dsi_975": si_train["dsi"].quantile(0.975),
        "dtcn_90": si_train["dtcn"].quantile(0.90),
        "dtcn_975": si_train["dtcn"].quantile(0.975),
        "ddtcn_90": si_train["ddtcn"].quantile(0.90),
        "ddtcn_975": si_train["ddtcn"].quantile(0.975),
    }

    return thresholds


def assign_borrow_tiers(si_with_ticker, thresholds):
    """
    Assign borrow tiers.

    Tier A: normal borrow, 40 bps p.a.
    Tier B: moderate hard-to-borrow, 200 bps p.a.
    Tier C: severe hard-to-borrow, 800 bps p.a.
    """
    borrow = si_with_ticker.copy()

    moderate_h2b = (
        (borrow["dsi"] >= thresholds["dsi_90"])
        | (borrow["dtcn"] >= thresholds["dtcn_90"])
        | (borrow["ddtcn"] >= thresholds["ddtcn_90"])
    )

    high_h2b = (
        (borrow["dsi"] >= thresholds["dsi_975"])
        | (borrow["dtcn"] >= thresholds["dtcn_975"])
        | (borrow["ddtcn"] >= thresholds["ddtcn_975"])
    )

    borrow["borrow_tier"] = "A"
    borrow.loc[moderate_h2b, "borrow_tier"] = "B"
    borrow.loc[high_h2b, "borrow_tier"] = "C"

    borrow["borrow_fee_bps_pa"] = borrow["borrow_tier"].map(
        {
            "A": 40,
            "B": 200,
            "C": 800,
        }
    )

    borrow["borrow_fee_daily"] = borrow["borrow_fee_bps_pa"] / 10000 / 252

    borrow = borrow[
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
    ].sort_values(["stock_id", "date"])

    return borrow


def main():
    prices, si = load_data()

    # Sanity check: stock_id should match instrument_id
    price_ids = set(prices["instrument_id"].dropna().unique())
    si_ids = set(si["stock_id"].dropna().unique())

    print("prices instrument_id count:", len(price_ids))
    print("short interest stock_id count:", len(si_ids))
    print("matched ids:", len(price_ids & si_ids))
    print("si ids missing from prices:", len(si_ids - price_ids))

    si_with_ticker = add_ticker(si, prices)

    missing_ticker_rate = si_with_ticker["ticker"].isna().mean()
    print("ticker missing rate:", round(missing_ticker_rate, 6))

    thresholds = compute_thresholds(si_with_ticker)
    print("thresholds:")
    for key, value in thresholds.items():
        print(f"  {key}: {value:.6f}")

    borrow = assign_borrow_tiers(si_with_ticker, thresholds)

    print("borrow tier distribution:")
    print(borrow["borrow_tier"].value_counts(normalize=True).sort_index())

    output_path = OUT_DIR / "borrow_tiers.parquet"
    borrow.to_parquet(output_path, index=False)

    print(f"Saved borrow tiers to: {output_path}")
    print("Output shape:", borrow.shape)


if __name__ == "__main__":
    main()
    