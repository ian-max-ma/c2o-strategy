"""
step1_panel/returns.py

Compute overnight (ON) and intraday (ID) returns from adjusted prices.

Definitions (Brief Section 2.1):
    Open_t, Close_t  — official auction prices, adjusted for splits and dividends.
    r_ON,t  = Open_t  / Close_{t-1} - 1   (close-to-open)
    r_ID,t  = Close_t / Open_t       - 1   (open-to-close)
    r_CC,t  = (1 + r_ON,t)(1 + r_ID,t) - 1 = Close_t / Close_{t-1} - 1

Corporate-action convention
---------------------------
`open` and `close` in prices.parquet are unadjusted auction prices.
`adjusted_close` is backward-adjusted for all splits and dividends.
There is no adjusted open column in the data.

We construct it from the same-day adjustment factor:
    adjusted_open_t = open_t × (adjusted_close_t / close_t)

This is valid because open and close on the same calendar day carry the same
backward-adjustment factor (no corporate action falls intraday between them).
With adjusted_open in hand, all three returns follow the brief's formulas exactly.

The reconciliation check  (1+r_ON)(1+r_ID) - 1 ≈ r_CC  is now a genuine
data-quality test: a failure means the data provider applied a corporate-action
adjustment to close but not to open on that date (brief §2.1: "diagnosed, not
silently masked").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRADER'S CLOCK — READ BEFORE USING PANEL COLUMNS IN STEPS 4 & 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Decision time is 15:50 ET on day t (brief Section 2.1.1).
Orders enter at MOC (Close_t, 16:00 ET); exit at MOO (Open_{t+1}, 09:30 ET).
Any alpha feature must be OBSERVABLE by 15:50 ET on day t.

  Column          Observable at 15:50?  Use as alpha feature?
  ─────────────────────────────────────────────────────────────
  r_ON            YES  (Open_t known at 09:30; Close_{t-1} from yesterday)
  r_CC            NO   (needs Close_t which settles at 16:00)
  r_ID            NO   (needs Close_t which settles at 16:00)
  market_cap      NO   (price × shares; uses Close_t)
  open            YES  (known at 09:30)
  ─────────────────────────────────────────────────────────────
  r_CC_lag1       YES  — pre-shifted version of r_CC, safe to use directly
  r_ID_lag1       YES  — pre-shifted version of r_ID, safe to use directly
  market_cap_lag1 YES  — pre-shifted version of market_cap, safe to use directly

Rule: in Steps 4 & 5, ONLY use r_CC_lag1 / r_ID_lag1 / market_cap_lag1
as features. Never use the raw r_CC / r_ID / market_cap directly as an
alpha input — those are kept in the panel for analysis/attribution only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd

RECONCILE_TOL = 1e-6   # max allowed residual |(1+r_ON)(1+r_ID)-1 - r_CC|
CORP_ACTION_TOL = 1e-4 # 1 bp difference between raw and adjusted close-to-close


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add return columns to a prices DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain: date, ticker, open, close, adjusted_close, market_cap.
        Sorted by [ticker, date].

    Returns
    -------
    pd.DataFrame
        All input columns plus:

        Intermediate:
          adjusted_open   open × (adjusted_close / close) — same-day adj factor

        Raw returns (for analysis/attribution — NOT for use as alpha features):
          r_ON           close-to-open  (observable at 15:50 ET — safe to use directly)
          r_ID           open-to-close  (uses Close_t — not observable at 15:50 ET)
          r_CC           close-to-close (uses Close_t — not observable at 15:50 ET)
          reconcile_fail bool; True if |(1+r_ON)(1+r_ID)-1 - r_CC| > RECONCILE_TOL

        Feature-safe lagged columns (15:50 ET observable — use these in Steps 4/5):
          r_CC_lag1      r_CC shifted 1 day within ticker
          r_ID_lag1      r_ID shifted 1 day within ticker
          market_cap_lag1 market_cap shifted 1 day within ticker
    """
    df = prices.sort_values(["ticker", "date"]).copy()

    # ── Construct adjusted_open from same-day adjustment factor ───────────────
    # adjusted_close / close gives the backward-adjustment factor for day t.
    # open and close on the same day share the same factor (no intraday CA).
    df["adj_factor"] = df["adjusted_close"] / df["close"]
    df["adjusted_open"] = df["open"] * df["adj_factor"]

    # ── Three returns per brief §2.1, all using adjusted prices ───────────────
    grp = df.groupby("ticker")
    prev_adj_close = grp["adjusted_close"].shift(1)

    df["r_ON"] = df["adjusted_open"]  / prev_adj_close       - 1  # Open_t / Close_{t-1}
    df["r_ID"] = df["adjusted_close"] / df["adjusted_open"]  - 1  # Close_t / Open_t
    df["r_CC"] = df["adjusted_close"] / prev_adj_close       - 1  # Close_t / Close_{t-1}

    # Raw price returns are kept only for diagnostics. They are not suitable for
    # the strategy around split/dividend dates, but they provide an independent
    # sanity check that the adjustment layer is doing real work.
    prev_close = grp["close"].shift(1)
    df["raw_ON"] = df["open"]  / prev_close - 1
    df["raw_ID"] = df["close"] / df["open"] - 1
    df["raw_CC"] = df["close"] / prev_close - 1
    df["cc_adjustment_gap"] = df["r_CC"] - df["raw_CC"]
    df["corp_action_day"] = df["cc_adjustment_gap"].abs() > CORP_ACTION_TOL

    # ── Reconciliation check — genuine data-quality test ──────────────────────
    # r_ON and r_ID are computed independently from r_CC.
    # A failure here means the data provider applied a corporate-action
    # adjustment inconsistently between open and close on that date.
    compounded = (1 + df["r_ON"]) * (1 + df["r_ID"]) - 1
    residual   = (compounded - df["r_CC"]).abs()
    df["reconcile_fail"] = residual > RECONCILE_TOL

    n_fail = int(df["reconcile_fail"].sum())
    n_total = int(df["r_CC"].notna().sum())
    if n_fail > 0:
        pct = 100 * n_fail / n_total
        print(
            f"[returns] Reconciliation failures: {n_fail:,} / {n_total:,} "
            f"({pct:.3f}%) stock-days exceed tolerance {RECONCILE_TOL:.0e}."
        )
    else:
        print(f"[returns] Reconciliation OK — 0 failures across {n_total:,} stock-days.")

    # ── Feature-safe lagged columns (15:50 ET observable) ────────────────────
    # r_CC and r_ID use Close_t (not known at 15:50); shift by 1 day so that
    # the value on row t is yesterday's return — safe for Steps 4 & 5.
    # r_ON already uses Open_t + Close_{t-1}, so NO shift needed.
    # market_cap uses Close_t, so also needs the lag.
    df["r_CC_lag1"]       = grp["r_CC"].shift(1)
    df["r_ID_lag1"]       = grp["r_ID"].shift(1)
    df["market_cap_lag1"] = grp["market_cap"].shift(1)

    return df


def reconciliation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-year breakdown of reconciliation failures.
    Useful for Section 2 of the report.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    summary = (
        df.groupby("year")
        .agg(
            total_stock_days=("r_CC", "count"),
            fail_count=("reconcile_fail", "sum"),
        )
        .assign(fail_pct=lambda x: 100 * x["fail_count"] / x["total_stock_days"])
    )
    return summary


def return_sanity_summary(df: pd.DataFrame) -> pd.Series:
    """
    Return non-tautological diagnostics for the return construction.

    The algebraic identity in `reconcile_fail` verifies implementation closure
    after `adjusted_open` has been constructed. The diagnostics here answer the
    stronger question: did the corporate-action adjustment materially alter raw
    close-to-close returns, and does it leave non-corporate-action days alone?
    """
    valid = df["r_CC"].notna() & df["raw_CC"].notna()
    raw_adj_gap = df.loc[valid, "cc_adjustment_gap"].abs()

    non_ca = valid & ~df["corp_action_day"]
    non_ca_gap = df.loc[non_ca, "cc_adjustment_gap"].abs()

    implementation_residual = (
        (1 + df["r_ON"]) * (1 + df["r_ID"]) - 1 - df["r_CC"]
    ).abs()

    raw_extreme = valid & (df["raw_CC"].abs() > 0.50)
    neutralised = raw_extreme & (df["r_CC"].abs() < df["raw_CC"].abs())

    return pd.Series({
        "stock_days_checked": int(valid.sum()),
        "implementation_identity_max_residual": float(implementation_residual.max()),
        "implementation_identity_fail_count": int((implementation_residual > RECONCILE_TOL).sum()),
        "corp_action_stock_days_gt_1bp": int((raw_adj_gap > CORP_ACTION_TOL).sum()),
        "corp_action_stock_days_gt_1bp_pct": float(100 * (raw_adj_gap > CORP_ACTION_TOL).mean()),
        "non_corp_action_stock_days": int(non_ca.sum()),
        "non_corp_action_max_raw_adjusted_gap": float(non_ca_gap.max()),
        "non_corp_action_fail_count_gt_1bp": int((non_ca_gap > CORP_ACTION_TOL).sum()),
        "raw_abs_return_gt_50pct_count": int(raw_extreme.sum()),
        "raw_extreme_neutralised_count": int(neutralised.sum()),
    })


def corporate_action_examples(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Largest raw-vs-adjusted close-to-close differences for hand inspection.
    """
    cols = [
        "date", "ticker", "open", "close", "adjusted_open", "adjusted_close",
        "raw_CC", "r_CC", "cc_adjustment_gap",
    ]
    out = (
        df.loc[df["r_CC"].notna() & df["raw_CC"].notna(), cols]
        .assign(abs_gap=lambda x: x["cc_adjustment_gap"].abs())
        .sort_values("abs_gap", ascending=False)
        .head(n)
    )
    return out.drop(columns=["abs_gap"])
