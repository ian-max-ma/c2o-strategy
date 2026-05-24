"""
run_step2_sanity.py

Generate Section 3 Q1-Q4 outputs for the report.
All files written to step2_universe/sanity_output/.

Usage:
    python run_step2_sanity.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from config import (
    AUM_LEVELS, PARTICIPATION_CAP, IMPACT_K,
    MIN_PRICE, MIN_ADV_USD, MAX_ANNUAL_VOL, EARNINGS_WINDOW, UNIVERSE_SIZE,
)
from step1_panel.loader import load_prices, load_earnings_calendar
from step1_panel.returns import compute_returns
from step2_universe.universe import build_yearly_universe, mark_eligible
from step2_universe.capacity import (
    apply_capacity_filters,
    fail_count_summary,
    capacity_summary,
    aum_eligible_summary,
    impact_slippage_report,
)

OUT = Path(__file__).parent / "step2_universe" / "sanity_output"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  C2O Section 3 Sanity Outputs (Q1-Q4)")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n[Loading] prices (2008-12-01 → TRAIN_END)...")
prices_full = load_prices(start="2008-12-01")
print(f"  {len(prices_full):,} rows | {prices_full['ticker'].nunique():,} tickers")

print("\n[Loading] prices (TRAIN_START → TRAIN_END)...")
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
n_univ = panel["in_universe"].sum()
n_elig = panel["eligible"].sum()
print(f"  in_universe: {n_univ:,}  |  eligible: {n_elig:,}  ({100*n_elig/n_univ:.1f}% of in-universe)")


# ══════════════════════════════════════════════════════════════════════════════
# Q1 — Filter thresholds
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q1 — Filter thresholds")

q1 = pd.Series({
    "UNIVERSE_SIZE":     UNIVERSE_SIZE,
    "MIN_PRICE":         MIN_PRICE,
    "MIN_ADV_USD":       MIN_ADV_USD,
    "MAX_ANNUAL_VOL":    MAX_ANNUAL_VOL,
    "EARNINGS_WINDOW":   EARNINGS_WINDOW,
    "PARTICIPATION_CAP": PARTICIPATION_CAP,
    "IMPACT_K":          IMPACT_K,
})
q1.to_csv(OUT / "q1_thresholds.csv", header=["value"])
print(q1.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Q2 — Participation cap and implied slippage
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q2 — Participation cap and implied slippage")

slippage = impact_slippage_report(k=IMPACT_K, f=PARTICIPATION_CAP)
slippage.to_csv(OUT / "q2_slippage.csv", index=False)
print(slippage.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Q3 — AUM-dependent eligible set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q3 — Eligible set at 50M / 250M / 1B AUM")

aum_df = aum_eligible_summary(
    panel, AUM_LEVELS, n_names=200, participation_cap=PARTICIPATION_CAP
)
aum_df.to_csv(OUT / "q3_aum_eligible.csv")
print(aum_df.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Q4 — Binding-constraint distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Q4 — Binding-constraint distribution by year")

fc = fail_count_summary(panel)
fc["ok_pct"] = (100 * fc["n_OK"] / fc["n_total"]).round(1)
fc.to_csv(OUT / "q4_fail_counts.csv")
print(fc.to_string())

print("\n  Capacity pass-rate summary (complement):")
cap = capacity_summary(panel)
cap.to_csv(OUT / "q4_capacity_summary.csv")
print(cap.to_string())


print(f"\nAll outputs written to: {OUT}/")
print("  q1_thresholds.csv")
print("  q2_slippage.csv")
print("  q3_aum_eligible.csv")
print("  q4_fail_counts.csv")
print("  q4_capacity_summary.csv")
