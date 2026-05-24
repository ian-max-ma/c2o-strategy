"""
run_sanity.py

Generate Section 2 Q1-Q6 outputs for the report.
All files written to step1_panel/sanity_output/.

Usage:
    python run_sanity.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import gc

from config import AUM_LEVELS, PARTICIPATION_CAP, IMPACT_K
from step1_panel.loader import (
    load_prices, load_earnings_calendar, load_short_interest, load_sp500_constituents,
)
from step1_panel.returns import (
    compute_returns, reconciliation_summary, return_sanity_summary,
    corporate_action_examples,
)
from step1_panel.sanity_check import (
    compute_ew_streams, plot_stylised_fact, yearly_sharpe, verify_amc_bmo,
    stylised_fact_diagnostics,
)
from step2_universe.universe import (
    build_yearly_universe, mark_eligible, universe_summary, mid_year_exits,
    survivorship_bias_diagnostic,
)
from step2_universe.capacity import apply_capacity_filters, aum_eligible_summary, fail_count_summary
from step3_borrow.borrow_filter import (
    build_daily_si, assign_htb_tier, plot_si_representative,
)

OUT = Path(__file__).parent / "step1_panel" / "sanity_output"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  C2O Section 2 Sanity Outputs (Q1-Q6)")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n[Loading] prices (2008-12-01 → TRAIN_END)...")
prices_full = load_prices(start="2008-12-01")
print(f"  {len(prices_full):,} rows | {prices_full['ticker'].nunique():,} tickers "
      f"| {prices_full['date'].min().date()} → {prices_full['date'].max().date()}")

print("\n[Loading] prices (TRAIN_START → TRAIN_END) for panel...")
prices = load_prices()

print("\n[Computing] returns...")
panel = compute_returns(prices)

print("\n[Building] yearly universe...")
yearly_univ = build_yearly_universe(prices_full)
panel = mark_eligible(panel, yearly_univ)

print("\n[Loading] earnings calendar...")
earnings = None
try:
    earnings = load_earnings_calendar()
    print(f"  {len(earnings):,} events loaded")
except FileNotFoundError:
    print("  [not found — earnings window filter skipped]")

print("\n[Applying] capacity filters...")
panel = apply_capacity_filters(panel, earnings=earnings)

print("\n[Loading] short interest...")
try:
    si_raw = load_short_interest()
    panel = build_daily_si(panel, si_raw)
    panel = assign_htb_tier(panel)
    si_ok = True
    print(f"  SI rows with data: {panel['dsi'].notna().sum():,}")
except FileNotFoundError:
    si_ok = False
    print("  [not found — SI step skipped]")


# ══════════════════════════════════════════════════════════════════════════════
# Q1 — Data source, window, survivorship
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q1 — Data source, window, survivorship")

q1 = {
    "total_rows":        len(prices),
    "n_tickers":         prices["ticker"].nunique(),
    "n_instruments":     prices["instrument_id"].nunique() if "instrument_id" in prices.columns else np.nan,
    "date_start":        str(prices["date"].min().date()),
    "date_end":          str(prices["date"].max().date()),
    "status_col_used":   "status" in prices.columns,
    "n_after_dedup":     len(prices),
    "tickers_last_before_2024_dec": int(
        (prices.groupby("ticker")["date"].max() < pd.Timestamp("2024-12-01")).sum()
    ),
}
pd.Series(q1).to_csv(OUT / "q1_data_summary.csv", header=["value"])
print(pd.Series(q1).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Q2 — Reconciliation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q2 — Return reconciliation and corporate-action sanity")

recon = reconciliation_summary(panel)
recon.to_csv(OUT / "q2_reconciliation.csv")
print(recon.to_string())

# also report max residual
compounded = (1 + panel["r_ON"]) * (1 + panel["r_ID"]) - 1
residual   = (compounded - panel["r_CC"]).abs()
max_res    = residual.max()
n_fail     = (residual > 1e-6).sum()
print(f"\n  Max residual: {max_res:.2e}")
print(f"  Failures > 1e-6: {n_fail:,}")
pd.Series({"max_residual": max_res, "n_fail_1e6": n_fail}).to_csv(
    OUT / "q2_reconciliation_summary.csv", header=["value"]
)

ret_sanity = return_sanity_summary(panel)
ret_sanity.to_csv(OUT / "q2_return_sanity.csv", header=["value"])
corporate_action_examples(panel).to_csv(OUT / "q2_corporate_action_examples.csv", index=False)

print("\n  Non-tautological return sanity:")
print(ret_sanity.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Q3 — AMC / BMO timing flag
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q3 — AMC / BMO timing flag")

if earnings is not None:
    needed = {"report_date", "strat_trading_date", "before_after_market"}
    if needed <= set(earnings.columns):
        amc_rows = earnings[earnings["before_after_market"] == "after"].dropna(
            subset=["report_date", "strat_trading_date"]
        ).head(5).copy()
        amc_rows["shift_days"] = (
            amc_rows["strat_trading_date"] - amc_rows["report_date"]
        ).dt.days
        # AMC: strat = report_date (same day, shift=0); entry at MOC that night is the
        # contaminated trade. Exclusion window is centred on report_date.
        amc_rows["ok"] = amc_rows["shift_days"] == 0

        bmo_rows = earnings[earnings["before_after_market"] == "before"].dropna(
            subset=["report_date", "strat_trading_date"]
        ).head(5).copy()
        bmo_rows["shift_days"] = (
            bmo_rows["strat_trading_date"] - bmo_rows["report_date"]
        ).dt.days
        # BMO: strat = report_date - 1 BDay (shift=-1); the overnight D-1→D open
        # already reflects the pre-market release. Exclusion window centred on D-1.
        bmo_rows["ok"] = bmo_rows["shift_days"] == -1

        examples = pd.concat([amc_rows, bmo_rows])
        examples.to_csv(OUT / "q3_amc_bmo_examples.csv", index=False)

        print("\nAMC examples:")
        print(amc_rows[["stock_id", "report_date", "strat_trading_date", "shift_days", "ok"]].to_string(index=False))
        print("\nBMO examples:")
        print(bmo_rows[["stock_id", "report_date", "strat_trading_date", "shift_days", "ok"]].to_string(index=False))

        total_amc = (earnings["before_after_market"] == "after").sum()
        total_bmo = (earnings["before_after_market"] == "before").sum()
        print(f"\n  Total AMC: {total_amc:,}  |  Total BMO: {total_bmo:,}")
    else:
        print(f"  Missing columns: {needed - set(earnings.columns)}")
else:
    print("  [earnings not loaded]")


# ══════════════════════════════════════════════════════════════════════════════
# Q4 — Short-interest construction + representative plot
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q4 — Short-interest construction and representative plot")

if si_ok and "instrument_id" in panel.columns:
    peak = panel.groupby("instrument_id")["dsi"].max()
    rep_id = int(peak.idxmax())
    rep_ticker = (
        panel.loc[panel["instrument_id"] == rep_id, "ticker"]
        .dropna().iloc[0]
        if (panel["instrument_id"] == rep_id).any()
        else "Unknown"
    )
    print(f"  Representative name: {rep_ticker} (instrument_id={rep_id}, peak dsi={peak.max():.3f})")
    q4_path = OUT / "q4_si_representative.png"
    if q4_path.exists():
        print(f"  Existing SI plot reused: {q4_path}")
    else:
        plot_si_representative(
            panel,
            instrument_id=rep_id,
            ticker_label=rep_ticker,
            save_path=str(q4_path),
        )
elif si_ok:
    print("  [instrument_id not in panel — skipping SI plot]")
else:
    print("  [SI data not loaded]")

# Q5/Q6 do not need daily SI columns; drop them to keep this script below the
# memory ceiling on laptops and restricted runners.
drop_si_cols = [c for c in ["dsi", "dtcn", "ddtcn", "htb_tier", "htb_flag"] if c in panel.columns]
if drop_si_cols:
    panel = panel.drop(columns=drop_si_cols)
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# Q5 — Eligible universe year by year
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q5 — Eligible universe evolution")

univ_sum = universe_summary(yearly_univ)
exits    = mid_year_exits(prices_full, yearly_univ)

q5 = univ_sum.set_index("year").join(exits[["n_exits", "pct_exits"]])
q5.to_csv(OUT / "q5_universe_evolution.csv")
print(q5.to_string())

try:
    sp500 = load_sp500_constituents()
    surv = survivorship_bias_diagnostic(
        sp500,
        set(prices_full["ticker"].unique()),
        start_year=int(q5.index.min()),
        end_year=int(q5.index.max()),
    )
    surv.to_csv(OUT / "q5_sp500_removal_proxy.csv")
    print("\n  S&P 500 removal proxy (missing from prices):")
    print(surv.to_string())
except FileNotFoundError:
    print("  [sp500 constituents not found — survivorship proxy skipped]")

# Release large raw price tables before plotting Q6.
del prices_full, prices
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# Q6 — Stylised fact (EW portfolio)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q6 — Stylised fact (EW year-start universe portfolio)")

# Brief Section 2.3 asks for the names eligible in the Section 2.2 universe:
# the frozen year-start top-1,000 list. Section 3 capacity filters are reported
# separately, so use in_universe here rather than the post-capacity eligible flag.
ew = compute_ew_streams(panel, filter_col="in_universe")
plot_stylised_fact(ew, save_path=str(OUT / "q6_stylised_fact.png"))

sharpe_df = yearly_sharpe(ew).round(3)
sharpe_df.to_csv(OUT / "q6_yearly_sharpe.csv")
print(sharpe_df.to_string())

stylised_diag = stylised_fact_diagnostics(ew)
stylised_diag.to_csv(OUT / "q6_stylised_fact_diagnostics.csv", header=["value"])
print("\n  Stylised fact diagnostics:")
print(stylised_diag.to_string())

# overall stats
for col, label in [("ew_ON", "ON"), ("ew_ID", "ID"), ("ew_CC", "CC")]:
    mu  = ew[col].mean() * 252
    vol = ew[col].std()  * np.sqrt(252)
    sr  = mu / vol if vol > 0 else np.nan
    print(f"  {label}: ann_ret={mu:.3f}  ann_vol={vol:.3f}  Sharpe={sr:.2f}")

print(f"\nAll outputs written to: {OUT}/")
print("  q1_data_summary.csv")
print("  q2_reconciliation.csv")
print("  q2_reconciliation_summary.csv")
print("  q3_amc_bmo_examples.csv")
print("  q4_si_representative.png")
print("  q5_universe_evolution.csv")
print("  q6_stylised_fact.png")
print("  q6_yearly_sharpe.csv")
