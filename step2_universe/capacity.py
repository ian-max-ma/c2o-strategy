"""
step2_universe/capacity.py

Apply liquidity and risk screens to the eligible universe.

Screens applied point-in-time (no look-ahead):
  1. Price floor: prior close >= MIN_PRICE (remove penny stocks)
  2. Liquidity floor: 20-day average dollar volume >= MIN_ADV_USD
  3. Volatility ceiling: 20-day realised vol (annualised) <= MAX_ANNUAL_VOL
  4. Earnings window: exclude EARNINGS_WINDOW days around report_date
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MIN_PRICE, MIN_ADV_USD, MAX_ANNUAL_VOL, EARNINGS_WINDOW


def apply_capacity_filters(
    panel: pd.DataFrame,
    earnings: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """
    Add boolean filter columns to the daily panel for in-universe rows.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain: date, ticker, close, volume, r_CC, in_universe.
        close and volume are unadjusted (same-day; no cross-day ratio).
    earnings : pd.DataFrame, optional
        Output of load_earnings_calendar() — columns: ticker, report_date.
        If None, the earnings window filter is skipped.

    Returns
    -------
    pd.DataFrame
        Panel with added columns:
          pass_price   — close >= MIN_PRICE
          pass_adv     — 20-day avg dollar volume >= MIN_ADV_USD
          pass_vol     — 20-day realised vol (ann.) <= MAX_ANNUAL_VOL
          pass_earnings— not within EARNINGS_WINDOW days of report_date
          eligible     — in_universe & all pass_* flags
    """
    df = panel.copy().sort_values(["ticker", "date"])

    # ── 1. Price floor ────────────────────────────────────────────────────────
    # Use prior close (observable at 15:50 ET on day t; today's close is not)
    prev_close = df.groupby("ticker")["close"].shift(1)
    df["pass_price"] = prev_close >= MIN_PRICE

    # ── 2. 20-day average dollar volume ──────────────────────────────────────
    df["dollar_vol"] = df["close"] * df["volume"]
    df["adv20"] = (
        df.groupby("ticker")["dollar_vol"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    )
    df["pass_adv"] = df["adv20"] >= MIN_ADV_USD

    # ── 3. 20-day realised volatility ceiling ─────────────────────────────────
    df["vol20"] = (
        df.groupby("ticker")["r_CC"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=10).std() * np.sqrt(252))
    )
    df["pass_vol"] = df["vol20"] <= MAX_ANNUAL_VOL

    # ── 4. Earnings window ────────────────────────────────────────────────────
    if earnings is not None and not earnings.empty:
        # earnings joins via stock_id == instrument_id; use strat_trading_date
        # (already adjusted: AMC → next trading day, BMO → same day)
        date_col = "strat_trading_date" if "strat_trading_date" in earnings.columns else "report_date"
        earnings_clean = earnings[["stock_id", date_col]].dropna().rename(
            columns={"stock_id": "instrument_id", date_col: "earn_date"}
        )

        # Expand each event to ±EARNINGS_WINDOW *trading* days (Mon–Fri).
        # Using BDay avoids the calendar-day bug where a Monday strat_trading_date
        # would skip excluding the preceding Friday (Timedelta(-1) lands on Sunday).
        from pandas.tseries.offsets import BDay
        offsets = range(-EARNINGS_WINDOW, EARNINGS_WINDOW + 1)
        shifted = pd.concat(
            [earnings_clean.assign(date=earnings_clean["earn_date"] + BDay(d))
             for d in offsets]
        )[["instrument_id", "date"]].drop_duplicates()

        shifted["near_earnings"] = True
        df = df.merge(shifted, on=["instrument_id", "date"], how="left")
        df["near_earnings"] = df["near_earnings"].fillna(False).astype(bool)
        df["pass_earnings"] = ~df["near_earnings"]
        df = df.drop(columns=["near_earnings"])
    else:
        df["pass_earnings"] = True

    # ── Composite eligible flag ───────────────────────────────────────────────
    df["eligible"] = (
        df["in_universe"]
        & df["pass_price"]
        & df["pass_adv"]
        & df["pass_vol"]
        & df["pass_earnings"]
    )

    # ── Binding-constraint status (Section 3.5) ───────────────────────────────
    # First failing filter wins; in_universe=False → MCAP_FAIL (not in top-1000)
    conditions = [
        ~df["in_universe"],
        ~df["pass_price"],
        ~df["pass_adv"],
        ~df["pass_vol"],
        ~df["pass_earnings"],
    ]
    choices = ["MCAP_FAIL", "PRICE_FAIL", "ADV_FAIL", "VOL_FAIL", "EARN_WINDOW"]
    df["eligibility_status"] = np.select(conditions, choices, default="OK")

    return df.drop(columns=["dollar_vol"])


def capacity_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-year breakdown of filter attrition as % of in-universe stock-days.
    Answers Section 2 Q5.
    """
    df = panel[panel["in_universe"]].copy()
    df["year"] = df["date"].dt.year

    summary = (
        df.groupby("year")
        .agg(
            stock_days=("in_universe", "count"),
            pct_pass_price=("pass_price", "mean"),
            pct_pass_adv=("pass_adv", "mean"),
            pct_pass_vol=("pass_vol", "mean"),
            pct_pass_earnings=("pass_earnings", "mean"),
            pct_eligible=("eligible", "mean"),
        )
    )
    # convert fractions to percentages
    pct_cols = [c for c in summary.columns if c.startswith("pct_")]
    summary[pct_cols] = (summary[pct_cols] * 100).round(1)
    return summary


# ── G8: Fail-count table (Brief §3 Q4) ───────────────────────────────────────

def fail_count_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-year count of in-universe stock-days excluded by each filter.

    Brief §3 Q4: "Provide a table by year of how many stock-days are
    excluded by each reason."  (capacity_summary shows pass-rates; this
    shows absolute fail-counts so the two complement each other.)
    """
    df = panel[panel["in_universe"]].copy()
    df["year"] = df["date"].dt.year
    return (
        df.groupby("year")
        .agg(
            n_total=("in_universe", "count"),
            n_PRICE_FAIL=("pass_price", lambda x: int((~x).sum())),
            n_ADV_FAIL=("pass_adv",    lambda x: int((~x).sum())),
            n_VOL_FAIL=("pass_vol",    lambda x: int((~x).sum())),
            n_EARN_WINDOW=("pass_earnings", lambda x: int((~x).sum())),
            n_OK=("eligible", "sum"),
        )
    )


# ── G5: Roll spread (Brief §3.2) ─────────────────────────────────────────────

def add_roll_spread(panel: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Compute the Roll (1984) bid-ask spread proxy and add as 'roll_spread'.

    Formula:  s = 2 * sqrt(max(0, -Cov(Δ log P_t, Δ log P_{t-1})))

    Uses unadjusted close (same-day ratio unaffected by splits).
    Result is shifted 1 day so it is point-in-time at 15:50 ET on day t
    (the rolling window through t-1 drives the value seen on day t).

    Brief §3.2 — "A rolling effective-spread estimator built from daily
    prices. The Roll spread is one well-known choice."
    """
    df = panel.sort_values(["ticker", "date"]).copy()

    df["_lr"] = df.groupby("ticker")["close"].transform(
        lambda s: np.log(s).diff()
    )
    df["_lr_lag1"] = df.groupby("ticker")["_lr"].transform(lambda s: s.shift(1))

    # Rolling covariance Cov(Δ_t, Δ_{t-1}) per ticker, then shift 1 for PIT
    raw_cov = df.groupby("ticker", group_keys=False).apply(
        lambda g: g["_lr"].rolling(window, min_periods=10).cov(g["_lr_lag1"]).shift(1)
    )
    # Flatten MultiIndex if present (pandas version differences)
    if isinstance(raw_cov.index, pd.MultiIndex):
        raw_cov = raw_cov.droplevel(0)

    df["roll_spread"] = 2.0 * np.sqrt((-raw_cov).clip(lower=0))
    return df.drop(columns=["_lr", "_lr_lag1"])


# ── G6: Implied slippage table (Brief §3 Q2) ─────────────────────────────────

def impact_slippage_report(k: float = 0.7, f: float = 0.05) -> pd.DataFrame:
    """
    Compute implied auction slippage at the participation cap for representative names.

    Formula (Brief §3.3):  Impact ≈ k × σ_daily × √f × 10^4  (basis points)

    Compares the model-implied impact against the flat 1.5 bps auction-slippage
    assumption in Table 6.1 (Brief §7.3 flags this reconciliation as required).

    Parameters
    ----------
    k : float   closing-auction venue constant (0.7, per brief §3.3)
    f : float   participation cap fraction (PARTICIPATION_CAP from config)
    """
    profiles = [
        ("Small-cap",  0.40, 5e6),
        ("Mid-cap",    0.25, 25e6),
        ("Large-cap",  0.18, 200e6),
    ]
    rows = []
    for name, ann_vol, adv in profiles:
        sigma_daily = ann_vol / np.sqrt(252)
        impact_bps = k * sigma_daily * np.sqrt(f) * 1e4
        rows.append({
            "profile":        name,
            "ann_vol":        f"{ann_vol*100:.0f}%",
            "adv_M":          f"${adv/1e6:.0f}M",
            "part_cap_f":     f"{f*100:.0f}%",
            "impact_bps":     round(impact_bps, 1),
            "table61_bps":    1.5,
            "ratio_vs_flat":  round(impact_bps / 1.5, 1),
        })
    return pd.DataFrame(rows)


# ── G7: AUM-dependent eligibility (Brief §3 Q3) ──────────────────────────────

def aum_eligible_summary(
    panel: pd.DataFrame,
    aum_levels: dict,
    n_names: int = 200,
    participation_cap: float = 0.05,
) -> pd.DataFrame:
    """
    Per-year fraction of eligible stock-days that are NOT participation-cap-binding
    at each portfolio-AUM level.

    At AUM level A with n_names total positions (long + short), the equal-weight
    notional per name per side is  A / n_names / 2.
    A name is 'not cap-binding' if:  adv20 × participation_cap ≥ equal-weight.

    Brief §3 Q3: "How does the eligible set evolve at 50M, 250M, and 1B?"

    Parameters
    ----------
    aum_levels        : dict   e.g. {"50M": 50e6, "250M": 250e6, "1B": 1e9}
    n_names           : int    total long+short basket size (200 → 100L + 100S)
    participation_cap : float  fraction of ADV cap per name
    """
    elig = panel[panel["eligible"]].copy()
    if "adv20" not in elig.columns:
        raise ValueError("adv20 column required; run apply_capacity_filters first.")
    elig["year"] = elig["date"].dt.year

    rows = []
    for year, grp in elig.groupby("year"):
        row: dict = {"year": year, "n_eligible_stockdays": len(grp)}
        for label, aum in aum_levels.items():
            eq_weight = aum / n_names / 2          # per-name notional (one side)
            cap_limit = grp["adv20"] * participation_cap
            pct_ok = float((cap_limit >= eq_weight).mean()) * 100
            row[f"pct_not_capbound_{label}"] = round(pct_ok, 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")
