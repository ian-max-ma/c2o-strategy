"""
config.py -- Global parameters for the C2O strategy.

All pipeline stages import from here. To change a parameter, change it here
and re-run the pipeline. Do not hard-code research choices in individual
modules.
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# Development cutoff
TRAIN_START = "2010-01-01"
TRAIN_END = "2024-12-31"

# Universe
UNIVERSE_SIZE = 1000
MIN_PRICE = 5.0
MIN_ADV_USD = 1_000_000
MAX_ANNUAL_VOL = 1.50
EARNINGS_WINDOW = 1

# Capacity and execution
PARTICIPATION_CAP = 0.05
IMPACT_K = 0.7
TRADING_DAYS = 252

# AUM levels
AUM_LEVELS = {
    "50M": 50_000_000,
    "250M": 250_000_000,
    "1B": 1_000_000_000,
}
DEFAULT_AUM = AUM_LEVELS["250M"]

# Section 6.3 fixed cost schedule
COMMISSION_BPS = 0.5
SLIPPAGE_BPS = 1.5
ROUNDTRIP_BPS = 4.0

BORROW_TIER_A_BPS = 40
BORROW_TIER_B_BPS = 200
BORROW_TIER_C_BPS = 800

# Short-interest availability lag
SI_PUBLICATION_LAG = 8
SI_DELIVERY_LAG = 2
SI_TOTAL_LAG = SI_PUBLICATION_LAG + SI_DELIVERY_LAG

# Deterministic hard-to-borrow proxy
HTB_MODERATE_SI = 0.10
HTB_HIGH_SI = 0.20
HTB_DTC_THRESHOLD = 10

# Portfolio construction
BASKET_QUANTILE = 0.10
BASELINE_BASKET_QUANTILE = 0.10
FINAL_BASKET_QUANTILE = 0.005
BASKET_QUANTILES_SENSITIVITY = (0.005, 0.01, 0.025, 0.05, 0.10, 0.20)
WEIGHTING_SCHEME = "equal"

FINAL_SCORE_COL = "score_random_forest"
COMPARISON_SCORE_COLS = [
    "score_baseline",
    "score_elastic_net",
    "score_random_forest",
]

# Cost-aware signal-strength scaling used by the reframed Step 5 version.
USE_SIGNAL_SCALING = True
SIGNAL_SCALING_LOOKBACK = 252
SIGNAL_SCALING_MIN_PERIODS = 60
SIGNAL_SCALING_QUANTILE = 0.60
SIGNAL_SCALING_MIN_MULT = 0.0
SIGNAL_SCALING_MID_MULT = 0.5
SIGNAL_SCALING_MAX_MULT = 1.0

# Borrow-aware robustness setting
USE_BORROW_ADJUSTED_SHORT_SCORE = False
BORROW_SCORE_PENALTY = {
    "A": 0.0,
    "B": 0.25,
    "C": 0.75,
}

# QuantStats
QUANTSTATS_AUM_LABEL = "250M"
QUANTSTATS_START_DATE = "2010-01-01"
QUANTSTATS_END_DATE = TRAIN_END

# Reproducibility
RANDOM_SEED = 42
