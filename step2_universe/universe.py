"""
step2_universe/universe.py

Construct the point-in-time eligible universe:
  - Top 1000 US common equities by market cap at each year-start
  - Frozen for the full calendar year
  - Mid-year exits only via delisting
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import UNIVERSE_SIZE


def build_yearly_universe(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each calendar year, rank stocks by market cap on the last
    trading day of the prior year and take the top UNIVERSE_SIZE names.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain: date, ticker, market_cap.

    Returns
    -------
    pd.DataFrame
        Columns: year, ticker, mcap_rank, mcap_at_selection.
        One row per (year, ticker) in the eligible universe.
    """
    prices = prices.sort_values(["ticker", "date"])
    results = []

    years = sorted(prices["date"].dt.year.unique())

    # precompute first observed date per ticker for the 12-month history check
    first_date = prices.groupby("ticker")["date"].min()

    for year in years:
        # last trading day of year Y-1
        prior_year_data = prices[prices["date"].dt.year == year - 1]
        if prior_year_data.empty:
            continue

        last_day = prior_year_data["date"].max()
        cutoff_12m = last_day - pd.DateOffset(months=12)

        snap = prior_year_data[prior_year_data["date"] == last_day][["ticker", "market_cap"]].copy()
        snap = snap.dropna(subset=["market_cap"])

        # brief 2.2: require at least 12 months of price history before year-start
        snap = snap[snap["ticker"].map(first_date) <= cutoff_12m]

        snap = (
            snap.sort_values("market_cap", ascending=False)
            .head(UNIVERSE_SIZE)
            .reset_index(drop=True)
        )
        snap["mcap_rank"] = snap.index + 1
        snap["year"] = year
        snap = snap.rename(columns={"market_cap": "mcap_at_selection"})
        results.append(snap)

    return pd.concat(results, ignore_index=True)


def mark_eligible(panel: pd.DataFrame, yearly_universe: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean 'in_universe' column to the daily panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Daily panel with date and ticker columns.
    yearly_universe : pd.DataFrame
        Output of build_yearly_universe().

    Returns
    -------
    pd.DataFrame
        Panel with 'in_universe' column added.
    """
    yearly_universe = yearly_universe.copy()
    yearly_universe["year"] = yearly_universe["year"].astype(int)

    panel = panel.copy()
    panel["year"] = panel["date"].dt.year

    panel = panel.merge(
        yearly_universe[["year", "ticker", "mcap_rank"]],
        on=["year", "ticker"],
        how="left",
    )
    panel["in_universe"] = panel["mcap_rank"].notna()
    return panel.drop(columns=["year"])


def universe_summary(yearly_universe: pd.DataFrame) -> pd.DataFrame:
    """
    Report eligible name count per year (Section 2 Q5).
    """
    return (
        yearly_universe.groupby("year")["ticker"]
        .count()
        .rename("n_eligible")
        .reset_index()
    )


def survivorship_bias_diagnostic(
    sp500_constituents: pd.DataFrame,
    prices_tickers: set,
    start_year: int = 2010,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Quantify the survivorship bias in prices.parquet using S&P 500 constituent
    windows as a proxy.

    prices.parquet contains only stocks that survived to the data cutoff (2024).
    Stocks that were S&P 500 members but were acquired or delisted before 2024
    are absent from the dataset entirely, making the panel survivorship-biased.

    This function reports, for each year, how many S&P 500 mid-year removals
    (proxy for delistings/acquisitions) are missing from prices.parquet.

    Parameters
    ----------
    sp500_constituents : pd.DataFrame
        Output of load_sp500_constituents() — columns: ticker, add_date, remove_date.
    prices_tickers : set
        Set of tickers present in prices.parquet (after status dedup).
    start_year, end_year : int
        Year range to report.

    Returns
    -------
    pd.DataFrame
        Columns: year, sp500_midyear_removed, missing_from_prices, example_missing.
    """
    const = sp500_constituents.copy()
    const["add_date"]    = pd.to_datetime(const["add_date"])
    const["remove_date"] = pd.to_datetime(const["remove_date"])

    rows = []
    for year in range(start_year, end_year + 1):
        yr_start = pd.Timestamp(f"{year}-01-01")
        yr_end   = pd.Timestamp(f"{year}-12-31")

        active = const[
            (const["add_date"] <= yr_end) &
            (const["remove_date"].isna() | (const["remove_date"] >= yr_start))
        ]
        removed_mid = active[
            active["remove_date"].notna() &
            (active["remove_date"] >= yr_start) &
            (active["remove_date"] < pd.Timestamp(f"{year}-12-01"))
        ]
        missing = removed_mid[~removed_mid["ticker"].isin(prices_tickers)]
        rows.append({
            "year":                  year,
            "sp500_midyear_removed": len(removed_mid),
            "missing_from_prices":   len(missing),
            "example_tickers":       ", ".join(missing["ticker"].head(4).tolist()),
        })
    return pd.DataFrame(rows).set_index("year")


def mid_year_exits(panel: pd.DataFrame, yearly_universe: pd.DataFrame) -> pd.DataFrame:
    """
    Count names that entered the year-start universe but whose last observed
    trading date is more than 30 calendar days before 31-Dec of that year —
    a proxy for mid-year delistings or acquisitions.

    Brief §2 Q5: "Provide a count of eligible names at each year-start and
    a count of mid-year exits."

    Parameters
    ----------
    panel           : daily panel with date and ticker columns
    yearly_universe : output of build_yearly_universe()

    Returns
    -------
    pd.DataFrame  index=year, columns: n_universe, n_exits, pct_exits
    """
    rows = []
    for year, grp in yearly_universe.groupby("year"):
        tickers = set(grp["ticker"])
        year_panel = panel[
            (panel["date"].dt.year == year) & panel["ticker"].isin(tickers)
        ]
        if year_panel.empty:
            rows.append({"year": year, "n_universe": len(tickers),
                         "n_exits": 0, "pct_exits": 0.0})
            continue
        last_date = year_panel.groupby("ticker")["date"].max()
        year_end  = pd.Timestamp(f"{year}-12-31")
        exits = int((last_date < year_end - pd.Timedelta(days=30)).sum())
        rows.append({
            "year":       year,
            "n_universe": len(tickers),
            "n_exits":    exits,
            "pct_exits":  round(100 * exits / len(tickers), 1),
        })
    return pd.DataFrame(rows).set_index("year")
