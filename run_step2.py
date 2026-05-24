"""
run_step2.py

Full pipeline: Steps 1–3 → saves outputs/panel_step3.parquet for teammates.

Steps run:
  1. Load prices, compute ON/ID/CC returns
  2. Build yearly universe (top-1,000 by market cap, 12-month history)
  3. Mark in_universe on daily panel
  4. Apply capacity filters → eligible flag
  5. Forward-fill short interest → assign HTB tiers (A/B/C)
  6. Save panel_step3.parquet

Usage:
    python run_step2.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step1_panel.loader import load_prices, load_earnings_calendar, load_short_interest
from step1_panel.returns import compute_returns
from step2_universe.universe import build_yearly_universe, mark_eligible
from step2_universe.capacity import apply_capacity_filters
from step3_borrow.borrow_filter import build_daily_si, assign_htb_tier

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading prices (2008-12-01 → TRAIN_END) for universe construction...")
prices_full = load_prices(start="2008-12-01")

print("Building yearly universe (top-1,000 by market cap)...")
yearly_univ = build_yearly_universe(prices_full)

print("Loading prices (TRAIN_START → TRAIN_END) for panel...")
prices = load_prices()
print(f"  {len(prices):,} rows, {prices['ticker'].nunique():,} tickers")

print("Computing returns...")
panel = compute_returns(prices)

print("Marking in_universe...")
panel = mark_eligible(panel, yearly_univ)
print(f"  in_universe stock-days: {panel['in_universe'].sum():,}")

print("Loading earnings calendar...")
earnings = None
try:
    earnings = load_earnings_calendar()
    print(f"  {len(earnings):,} events loaded")
except FileNotFoundError:
    print("  [earnings_calendar.parquet not found — skipping earnings window filter]")

print("Applying capacity filters...")
panel = apply_capacity_filters(panel, earnings=earnings)
print(f"  eligible stock-days: {panel['eligible'].sum():,}")

print("Building daily short-interest panel...")
try:
    si_raw = load_short_interest()
    print(f"  {len(si_raw):,} bi-weekly snapshots, {si_raw['stock_id'].nunique():,} instruments")
    panel = build_daily_si(panel, si_raw)
    panel = assign_htb_tier(panel)
    n_si = panel["dsi"].notna().sum()
    print(f"  SI coverage: {n_si:,} / {len(panel):,} ({100*n_si/len(panel):.1f}%)")
except FileNotFoundError:
    print("  [short_interest_transfo.parquet not found — skipping SI step]")

panel_path = OUTPUT_DIR / "panel_step3.parquet"
panel.to_parquet(panel_path, index=False)
print(f"\nDone. Panel saved → {panel_path}")
print("  Columns for Steps 3–5:")
print("    r_ON, r_CC_lag1, r_ID_lag1, market_cap_lag1  — returns / features")
print("    in_universe, eligible                          — universe / eligibility masks")
print("    dsi, dtcn, htb_tier, htb_flag                 — short interest / borrow")
