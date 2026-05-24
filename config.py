"""
config.py — Global parameters for the C2O strategy.

All pipeline stages import from here. To change a parameter,
change it here and re-run the pipeline — never hardcode values
in individual modules.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent
DATA_DIR  = ROOT_DIR / "data"

# ── Date window ──────────────────────────────────────────────────────────────
# CRITICAL: nothing in the development pipeline may use data after this date.
# The marker will re-run the code on 2025-2026; leaking that window is invalid.
TRAIN_START = "2010-01-01"
TRAIN_END   = "2024-12-31"   # ← single configurable cutoff, never change for dev

# ── Universe ─────────────────────────────────────────────────────────────────
UNIVERSE_SIZE       = 1000        # top N names by market cap at each year-start
MIN_PRICE           = 5.0         # USD — penny-stock floor
MIN_ADV_USD         = 1_000_000   # minimum 20-day average dollar volume
MAX_ANNUAL_VOL      = 1.50        # 150% annualised vol ceiling (removes distressed names)
EARNINGS_WINDOW     = 1           # trading days to exclude around earnings (each side)

# ── Capacity & execution ──────────────────────────────────────────────────────
PARTICIPATION_CAP   = 0.05        # max fraction of ADV per position per day
IMPACT_K            = 0.7         # square-root impact constant for closing auction
TRADING_DAYS        = 252

# ── Portfolio AUM levels to report ───────────────────────────────────────────
AUM_LEVELS = {
    "50M":  50_000_000,
    "250M": 250_000_000,
    "1B":   1_000_000_000,
}
DEFAULT_AUM = AUM_LEVELS["250M"]  # used for the QuantStats tear-sheet

# ── Cost schedule (Section 6.3 — fixed, do not change) ───────────────────────
COMMISSION_BPS      = 0.5         # per leg
SLIPPAGE_BPS        = 1.5         # per leg (auction)
ROUNDTRIP_BPS       = 4.0         # commission + slippage, both legs

BORROW_TIER_A_BPS   = 40          # General Collateral (annual)
BORROW_TIER_B_BPS   = 200         # Mid-tier specials (annual)
BORROW_TIER_C_BPS   = 800         # Deep specials (annual)

# ── Short-interest lag (Section 2.1.3) ───────────────────────────────────────
# FINRA publishes ~8 calendar days after snapshot; add 2 days for vendor delivery
SI_PUBLICATION_LAG  = 8           # calendar days
SI_DELIVERY_LAG     = 2           # additional calendar days
SI_TOTAL_LAG        = SI_PUBLICATION_LAG + SI_DELIVERY_LAG  # = 10

# ── Hard-to-borrow thresholds (your proxy — adjust after analysis) ────────────
HTB_MODERATE_SI     = 0.10        # short interest / float > 10% → Tier B
HTB_HIGH_SI         = 0.20        # short interest / float > 20% → Tier C
HTB_DTC_THRESHOLD   = 10          # days-to-cover > 10 → flag as HTB

# ── Basket construction ───────────────────────────────────────────────────────
BASKET_QUANTILE     = 0.10        # top & bottom 10% of scores → long/short
WEIGHTING_SCHEME    = "equal"     # "equal" | "score" | "vol"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED         = 42