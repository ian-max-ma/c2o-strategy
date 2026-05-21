"""
Build daily eligibility table for the tradable universe.

Output:
    step2_universe/eligibility_table.parquet

Initial rule:
    - Stock must be an S&P 500 member on that date
    - Has valid adjusted close and volume
    - Has positive market cap
    - 20-day average dollar volume (ADV20) above threshold
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "step2_universe"

TRAIN_END = pd.Timestamp("2024-12-31")
ADV20_MIN = 20_000_000


def add_sp500_membership(prices, sp500):
    """Add point-in-time S&P 500 membership flag."""
    prices = prices.copy()
    sp500 = sp500.copy()

    prices["date"] = pd.to_datetime(prices["date"])
    sp500["add_date"] = pd.to_datetime(sp500["add_date"])
    sp500["remove_date"] = pd.to_datetime(sp500["remove_date"])

    merged = prices.merge(sp500, on="ticker", how="left")

    merged["is_sp500"] = (
        (merged["date"] >= merged["add_date"])
        & (
            merged["remove_date"].isna()
            | (merged["date"] <= merged["remove_date"])
        )
    )

    return merged


def main():
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    sp500 = pd.read_parquet(DATA_DIR / "sp500_constituents.parquet")

    prices["date"] = pd.to_datetime(prices["date"])

    prices = prices[prices["date"] <= TRAIN_END].copy()

    # Keep only useful columns
    prices = prices[
        [
            "ticker",
            "instrument_id",
            "date",
            "adjusted_close",
            "volume",
            "market_cap",
            "status",
        ]
    ].copy()

    prices = prices.rename(columns={"instrument_id": "stock_id"})

    # Dollar volume = price × shares traded
    prices["dollar_volume"] = prices["adjusted_close"] * prices["volume"]

    prices = prices.sort_values(["stock_id", "date"])

    # 20-day average dollar volume, shifted by 1 day to avoid using today's close/volume
    prices["adv20"] = (
        prices.groupby("stock_id")["dollar_volume"]
        .transform(lambda x: x.rolling(20, min_periods=10).mean().shift(1))
    )

    panel = add_sp500_membership(prices, sp500)

    panel["is_eligible"] = (
        panel["is_sp500"]
        & panel["adjusted_close"].notna()
        & panel["volume"].notna()
        & (panel["volume"] > 0)
        & panel["market_cap"].notna()
        & (panel["market_cap"] > 0)
        & panel["adv20"].notna()
        & (panel["adv20"] >= ADV20_MIN)
    )

    eligibility = panel[
        [
            "date",
            "stock_id",
            "ticker",
            "is_sp500",
            "adv20",
            "market_cap",
            "is_eligible",
        ]
    ].sort_values(["date", "stock_id"])

    output_path = OUT_DIR / "eligibility_table.parquet"
    eligibility.to_parquet(output_path, index=False)

    print("Saved eligibility table to:", output_path)
    print("Output shape:", eligibility.shape)
    print("Date range:", eligibility["date"].min(), "->", eligibility["date"].max())
    print("Eligible ratio:", eligibility["is_eligible"].mean())
    print()
    print("Average eligible names per day:")
    print(eligibility.groupby("date")["is_eligible"].sum().describe())


if __name__ == "__main__":
    main()