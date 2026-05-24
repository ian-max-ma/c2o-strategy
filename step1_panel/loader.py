"""
step1_panel/loader.py

Load raw parquet files from the data directory and return
clean DataFrames ready for panel construction.

All functions enforce the TRAIN_END cutoff — no 2025+ data
enters the development pipeline from this point forward.
"""

import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, TRAIN_START, TRAIN_END


def _load(filename: str) -> pd.DataFrame:
    """Read a parquet file from DATA_DIR."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. "
            "Download data files from the shared Dropbox link and place them in data/."
        )
    return pd.read_parquet(path)


def load_prices(start: "str | None" = None) -> pd.DataFrame:
    """
    Load prices.parquet and enforce the development cutoff.

    Parameters
    ----------
    start : str, optional
        Earliest date to include (YYYY-MM-DD).  Defaults to TRAIN_START.
        Pass an earlier date (e.g. "2009-01-01") when building the yearly
        universe, which needs the last trading day of the prior year.
        TRAIN_END is always enforced regardless of this parameter.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, open, high, low, close, adjusted_close,
        volume, market_cap, status, updated.
        Filtered to start .. TRAIN_END.
    """
    if start is None:
        start = TRAIN_START

    df = _load("prices.parquet")

    date_col = _find_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})

    # ── CRITICAL: TRAIN_END is always enforced — never leak 2025+ ────────────
    df = df[(df["date"] >= start) & (df["date"] <= TRAIN_END)]

    # ── De-duplicate ticker aliases ───────────────────────────────────────────
    # The parquet contains both a "primary" ticker (status='1') and alias tickers
    # (status='0') for the same instrument_id with identical price/market_cap.
    # E.g. MRSH (status='1') and MMC (status='0') share instrument_id 195.
    # Keeping both would double-count large-cap names in the top-1000 ranking.
    if "status" in df.columns:
        df = df[df["status"] == "1"]

    # ── Price sanity: replace placeholder values (e.g. close=1,000,000) ───────
    # These are data-vendor sentinel values, not real prices.
    # Set to NaN so downstream code sees a missing observation rather than
    # a spurious extreme return. Documented for Section 2 Q2 of the report.
    _PRICE_COLS = ["open", "high", "low", "close"]
    _MAX_VALID_PRICE = 100_000
    bad_mask = df[_PRICE_COLS].gt(_MAX_VALID_PRICE).any(axis=1)
    n_bad = bad_mask.sum()
    if n_bad > 0:
        print(f"[load_prices] {n_bad:,} rows with price > {_MAX_VALID_PRICE:,} set to NaN "
              f"({100*n_bad/len(df):.4f}% of stock-days).")
        df.loc[bad_mask, _PRICE_COLS + ["market_cap"]] = float("nan")

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def load_sp500_constituents() -> pd.DataFrame:
    """
    Load historical S&P 500 membership windows.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, add_date, remove_date (NaT if still in index).
        Both date columns are datetime64[ns] as stored in the parquet file.
    """
    return _load("sp500_constituents.parquet")


def load_sp500_tr() -> pd.DataFrame:
    """Load S&P 500 Total Return index (benchmark for QuantStats)."""
    df = _load("sp500_tr.parquet")
    date_col = _find_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})
    df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    return df.sort_values("date").reset_index(drop=True)


def load_earnings_calendar() -> pd.DataFrame:
    """
    Load earnings_calendar.parquet with BMO/AMC timing flags.

    Returns
    -------
    pd.DataFrame
        Columns: stock_id, report_date, strat_trading_date,
                 before_after_market ('before' | 'after'), reporting_time,
                 period, period_end_date.
        stock_id matches instrument_id in prices.parquet.

        strat_trading_date encodes the C2O entry date (when you buy at MOC):
          AMC reports: strat_trading_date = report_date (shift = 0).
                       Entry MOC on report_date; overnight D→D+1 captures AMC news.
          BMO reports: strat_trading_date = report_date - 1 BDay (shift = -1).
                       Entry MOC on D-1; overnight D-1→D open already reflects BMO.
    """
    df = _load("earnings_calendar.parquet")
    # reporting_date is the actual report date; rename for clarity
    if "reporting_date" in df.columns:
        df["reporting_date"] = pd.to_datetime(df["reporting_date"])
        df = df.rename(columns={"reporting_date": "report_date"})
    if "strat_trading_date" in df.columns:
        df["strat_trading_date"] = pd.to_datetime(df["strat_trading_date"])
    return df


def load_short_interest() -> pd.DataFrame:
    """
    Load short_interest_transfo.parquet.

    The `date` column is already the publication/availability date:
    FINRA bi-weekly snapshot + ~8 days publication + 2 days vendor ≈ 10-day lag
    already baked in by the data provider (Section 2.1.3).
    Use as-of merge (forward-fill) when joining onto the daily panel,
    since snapshots are bi-weekly and most trading days have no new entry.
    """
    return _load("short_interest_transfo.parquet")


def load_gics() -> pd.DataFrame:
    """Load GICS sector/industry classification."""
    return _load("gics_info.parquet")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_date_col(df: pd.DataFrame) -> str:
    """Return the name of the date column (handles different naming conventions)."""
    candidates = ["date", "Date", "DATE", "trading_date", "report_date"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first column that looks like a date
    for c in df.columns:
        try:
            pd.to_datetime(df[c].iloc[:5])
            return c
        except Exception:
            continue
    raise ValueError(f"Cannot find a date column in DataFrame with columns: {df.columns.tolist()}")
