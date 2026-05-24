"""
step3_borrow/borrow_filter.py

Build a point-in-time hard-to-borrow (HTB) proxy from short-interest data
and assign borrow tiers (A / B / C) for the cost schedule of Section 6.3.

SI data arrives as bi-weekly FINRA snapshots. Construction (Section 2.1.3):
  - The `date` in short_interest_transfo.parquet is already the availability date
    (snapshot + ~8 days FINRA publication + 2 days vendor delivery ≈ D+10).
  - For panel date t, the value is the most recent snapshot with availability ≤ t-1.
  - Forward-fill between releases so every trading day has a value.

Inputs (from short_interest_transfo.parquet):
    dsi   — short interest ratio (short interest / shares outstanding)
    dtcn  — days-to-cover (short interest / avg daily volume)
    ddtcn — change in days-to-cover across consecutive snapshots

stock_id in the SI file equals instrument_id in prices.parquet.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HTB_MODERATE_SI, HTB_HIGH_SI, HTB_DTC_THRESHOLD


def build_daily_si(panel: pd.DataFrame, si: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill bi-weekly SI snapshots onto the daily panel.

    For panel date t, uses the most recent SI snapshot whose effective
    availability date is ≤ t-1 (Section 2.1.3 — the snapshot cannot be
    used on the same day it arrives; it is available from the next day).

    Parameters
    ----------
    panel : pd.DataFrame
        Daily panel with columns: date (datetime), instrument_id (int).
    si : pd.DataFrame
        Output of load_short_interest() — columns: stock_id, date, dsi, dtcn, ddtcn.
        `date` is already the availability date (snapshot + ~10 calendar days).

    Returns
    -------
    pd.DataFrame
        Panel with columns dsi, dtcn, ddtcn added.
        NaN where no prior SI snapshot exists for that instrument.
    """
    si = si.rename(columns={"stock_id": "instrument_id"}).copy()

    # Shift availability dates forward by 1 calendar day to enforce the
    # "≤ t-1" constraint: a snapshot available on date A is eligible for
    # panel rows on date A+1 or later (not the same day).
    # After the shift, a backward merge_asof on the original panel date
    # finds the correct snapshot: shifted_date ≤ panel_date
    # ↔ original_availability ≤ panel_date - 1  ✓
    si["date"] = si["date"] + pd.Timedelta(days=1)

    # merge_asof requires both DataFrames to be sorted by the merge key (date)
    # globally (not just within groups). Sort by date only; within each
    # instrument_id group the dates are still monotonic after this global sort.
    si = si.sort_values("date")

    # Deduplicate panel keys: if two tickers share the same instrument_id on the
    # same date (ticker-reuse edge case), panel_keys would have duplicate
    # (date, instrument_id) rows. That causes merge_asof to output duplicates,
    # which then fan out through the final left-merge and inflate the panel.
    panel_keys = (
        panel[["date", "instrument_id"]]
        .drop_duplicates(subset=["date", "instrument_id"])
        .sort_values("date")
        .copy()
    )

    daily_si = pd.merge_asof(
        panel_keys,
        si[["instrument_id", "date", "dsi", "dtcn", "ddtcn"]],
        on="date",
        by="instrument_id",
        direction="backward",   # most recent snapshot ≤ panel_date
    )
    # daily_si now has: date, instrument_id, dsi, dtcn, ddtcn
    # Each (date, instrument_id) pair appears exactly once — safe left-merge.

    panel = panel.merge(
        daily_si[["date", "instrument_id", "dsi", "dtcn", "ddtcn"]],
        on=["date", "instrument_id"],
        how="left",
    )
    return panel


def assign_htb_tier(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add htb_tier (A | B | C) and htb_flag columns to the daily panel.

    Requires dsi and dtcn columns (produced by build_daily_si).
    Rows where dsi is NaN (no SI data yet) default to Tier A (General Collateral).

    Tier assignment (thresholds in config.py):
        Tier C — dsi > HTB_HIGH_SI  OR  dtcn > 2 × HTB_DTC_THRESHOLD
        Tier B — dsi > HTB_MODERATE_SI  OR  dtcn > HTB_DTC_THRESHOLD
        Tier A — everything else
    """
    panel = panel.copy()
    panel["htb_tier"] = "A"

    has_si = panel["dsi"].notna()

    tier_b = has_si & (
        (panel["dsi"] > HTB_MODERATE_SI) | (panel["dtcn"] > HTB_DTC_THRESHOLD)
    )
    panel.loc[tier_b, "htb_tier"] = "B"

    tier_c = has_si & (
        (panel["dsi"] > HTB_HIGH_SI) | (panel["dtcn"] > 2 * HTB_DTC_THRESHOLD)
    )
    panel.loc[tier_c, "htb_tier"] = "C"

    panel["htb_flag"] = panel["htb_tier"] != "A"
    return panel


def htb_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-year count of eligible stock-days by borrow tier.
    Answers Section 4 Q3: what fraction of the raw short signal is HTB-affected.
    """
    df = panel[panel["eligible"]].copy()
    df["year"] = df["date"].dt.year
    result = (
        df.groupby(["year", "htb_tier"])
        .size()
        .unstack(fill_value=0)
    )
    # add pct of non-GC days
    total = result.sum(axis=1)
    non_gc = result.get("B", 0) + result.get("C", 0)
    result["pct_non_gc"] = (100 * non_gc / total).round(1)
    return result


def si_coverage_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-year fraction of eligible stock-days that have SI data.
    Shows the coverage gap at the start of the history.
    """
    df = panel[panel["eligible"]].copy()
    df["year"] = df["date"].dt.year
    return (
        df.groupby("year")
        .agg(
            stock_days=("dsi", "count"),
            pct_with_si=("dsi", lambda x: 100 * x.notna().mean()),
        )
        .round(1)
    )


# ── G3: SI plot for a representative name (Brief §2 Q4) ──────────────────────

def plot_si_representative(
    panel: pd.DataFrame,
    instrument_id: int,
    ticker_label: str = "Unknown",
    save_path: "str | None" = None,
) -> None:
    """
    Plot the forward-filled short-interest ratio (dsi) and days-to-cover (dtcn)
    for a single instrument over the full development window.

    Brief §2 Q4: "Plot the resulting series for a representative name."

    The figure shows:
      - The stepped bi-weekly update cadence (forward-fill between releases)
      - The D+10+1 availability lag (no future data enters before it is available)
      - Tier B / Tier C thresholds as reference lines

    Parameters
    ----------
    panel        : daily panel with columns date, instrument_id, dsi, dtcn
    instrument_id: integer instrument_id to plot
    ticker_label : human-readable ticker for the chart title
    save_path    : if provided, save PNG to this path instead of showing
    """
    import matplotlib.pyplot as plt

    grp = panel[panel["instrument_id"] == instrument_id].sort_values("date")
    if grp.empty or grp["dsi"].isna().all():
        print(f"[plot_si_representative] No SI data for instrument_id={instrument_id}.")
        return

    from config import HTB_MODERATE_SI, HTB_HIGH_SI, HTB_DTC_THRESHOLD

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(
        f"Short Interest — {ticker_label} (instrument_id={instrument_id})\n"
        "Forward-filled from bi-weekly FINRA snapshots; availability lag D+10+1",
        fontsize=11,
    )

    ax1.step(grp["date"], grp["dsi"], where="post", color="steelblue", lw=1.5)
    ax1.axhline(HTB_MODERATE_SI, color="darkorange", lw=1.2, ls="--",
                label=f"Tier B threshold ({HTB_MODERATE_SI*100:.0f}%)")
    ax1.axhline(HTB_HIGH_SI,     color="crimson",    lw=1.2, ls="--",
                label=f"Tier C threshold ({HTB_HIGH_SI*100:.0f}%)")
    ax1.set_ylabel("Short Interest / Shares Outstanding (dsi)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.step(grp["date"], grp["dtcn"], where="post", color="seagreen", lw=1.5)
    ax2.axhline(HTB_DTC_THRESHOLD,     color="darkorange", lw=1.2, ls="--",
                label=f"Tier B DTC ({HTB_DTC_THRESHOLD} days)")
    ax2.axhline(2 * HTB_DTC_THRESHOLD, color="crimson",    lw=1.2, ls="--",
                label=f"Tier C DTC ({2*HTB_DTC_THRESHOLD} days)")
    ax2.set_ylabel("Days-to-Cover (dtcn)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_si_representative] Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def borrow_cost_daily(notional: float, tier: str, trading_days: int = 252) -> float:
    """
    Daily borrow cost in dollars for one overnight short position.

    Parameters
    ----------
    notional     : float   gross short notional in dollars
    tier         : str     "A", "B", or "C"
    trading_days : int     convention for annualisation (default 252)

    Returns
    -------
    float   dollar cost for one night's borrow
    """
    from config import BORROW_TIER_A_BPS, BORROW_TIER_B_BPS, BORROW_TIER_C_BPS
    annual_bps = {"A": BORROW_TIER_A_BPS, "B": BORROW_TIER_B_BPS, "C": BORROW_TIER_C_BPS}
    annual_rate = annual_bps[tier] / 10_000
    return notional * annual_rate / trading_days
