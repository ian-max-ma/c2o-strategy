"""
Step 5 portfolio construction and costed backtest.

This module turns Step 4 alpha scores into a daily overnight long-short
portfolio. It is deliberately explicit about execution costs because the
strategy trades a full close-to-open round trip every day.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AUM_LEVELS,
    BASKET_QUANTILE,
    BASKET_QUANTILES_SENSITIVITY,
    BORROW_SCORE_PENALTY,
    BORROW_TIER_A_BPS,
    BORROW_TIER_B_BPS,
    BORROW_TIER_C_BPS,
    COMMISSION_BPS,
    DEFAULT_AUM,
    FINAL_SCORE_COL,
    PARTICIPATION_CAP,
    ROUNDTRIP_BPS,
    SIGNAL_SCALING_LOOKBACK,
    SIGNAL_SCALING_MAX_MULT,
    SIGNAL_SCALING_MID_MULT,
    SIGNAL_SCALING_MIN_MULT,
    SIGNAL_SCALING_MIN_PERIODS,
    SIGNAL_SCALING_QUANTILE,
    SLIPPAGE_BPS,
    TRADING_DAYS,
    TRAIN_END,
    USE_BORROW_ADJUSTED_SHORT_SCORE,
    WEIGHTING_SCHEME,
)


BORROW_BPS_BY_TIER = {
    "A": BORROW_TIER_A_BPS,
    "B": BORROW_TIER_B_BPS,
    "C": BORROW_TIER_C_BPS,
}

STRESS_WINDOWS = {
    "2018_Q4": ("2018-10-01", "2018-12-31"),
    "2020_Q1": ("2020-01-01", "2020-03-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2022_H2": ("2022-07-01", "2022-12-31"),
}


@dataclass(frozen=True)
class PortfolioConfig:
    """Parameters for one Step 5 backtest run."""

    aum: float = DEFAULT_AUM
    score_col: str = "score_random_forest"
    basket_quantile: float = BASKET_QUANTILE
    weighting_scheme: str = WEIGHTING_SCHEME
    participation_cap: float = PARTICIPATION_CAP
    commission_bps: float = COMMISSION_BPS
    slippage_bps: float = SLIPPAGE_BPS
    roundtrip_bps: float = ROUNDTRIP_BPS
    use_signal_scaling: bool = False
    gross_multiplier_col: str = "gross_multiplier"
    use_borrow_adjusted_short_score: bool = USE_BORROW_ADJUSTED_SHORT_SCORE

    @property
    def per_leg_cost_bps(self) -> float:
        """Commission plus auction slippage for one trade leg."""
        return self.commission_bps + self.slippage_bps


def validate_cost_config(config: PortfolioConfig) -> None:
    """
    Check that the fixed Brief Section 6 cost schedule is internally consistent.

    For this coursework, per-leg cost is 0.5 bps commission + 1.5 bps slippage
    = 2.0 bps. An overnight round trip has entry and exit legs, so the daily
    full-gross cost is 4.0 bps of gross exposure, or 10.08% annualised at
    100% gross exposure.
    """
    implied_roundtrip = 2.0 * config.per_leg_cost_bps
    if not np.isclose(config.roundtrip_bps, implied_roundtrip):
        raise ValueError(
            "Cost schedule mismatch: ROUNDTRIP_BPS should equal "
            f"2 * (COMMISSION_BPS + SLIPPAGE_BPS) = {implied_roundtrip:.2f}, "
            f"but got {config.roundtrip_bps:.2f}."
        )


def cost_schedule_summary(config: PortfolioConfig | None = None) -> pd.DataFrame:
    """
    Return the fixed cost schedule used in Step 5.

    This is a small audit table for the report. It makes clear that 4 bps is
    applied per full overnight round trip, while 2 bps is applied per leg.
    """
    if config is None:
        config = PortfolioConfig()
    validate_cost_config(config)

    return pd.DataFrame(
        [
            {
                "item": "commission_per_leg",
                "bps": config.commission_bps,
                "frequency": "entry and exit",
                "applies_to": "traded notional",
            },
            {
                "item": "auction_slippage_per_leg",
                "bps": config.slippage_bps,
                "frequency": "entry and exit",
                "applies_to": "traded notional",
            },
            {
                "item": "total_per_leg",
                "bps": config.per_leg_cost_bps,
                "frequency": "entry or exit",
                "applies_to": "traded notional",
            },
            {
                "item": "roundtrip_commission_slippage",
                "bps": config.roundtrip_bps,
                "frequency": "daily overnight round trip",
                "applies_to": "gross exposure",
            },
            {
                "item": "tier_A_borrow",
                "bps": BORROW_TIER_A_BPS,
                "frequency": "annual, charged daily / 252",
                "applies_to": "short notional",
            },
            {
                "item": "tier_B_borrow",
                "bps": BORROW_TIER_B_BPS,
                "frequency": "annual, charged daily / 252",
                "applies_to": "short notional",
            },
            {
                "item": "tier_C_borrow",
                "bps": BORROW_TIER_C_BPS,
                "frequency": "annual, charged daily / 252",
                "applies_to": "short notional",
            },
        ]
    )


def annualised_return(returns: pd.Series) -> float:
    """Compound annualised return from a daily return series."""
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1.0)


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised daily-return volatility."""
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualised Sharpe ratio with zero risk-free rate."""
    vol = annualised_volatility(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return annualised_return(returns) / vol


def cumulative_return(returns: pd.Series) -> float:
    """Cumulative return from a daily return series."""
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from a daily return series."""
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def choose_score_column(df: pd.DataFrame) -> str:
    """
    Pick the pre-specified final Step 4 score column.

    The headline Step 5 run uses the model selected in Step 4 before portfolio
    construction. Other score columns are retained only for diagnostic
    comparison, not for ex-post portfolio selection.
    """
    preferred = [
        FINAL_SCORE_COL,
        "score_random_forest",
        "score_elastic_net",
        "score_baseline",
    ]
    for col in dict.fromkeys(preferred):
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError("No usable Step 4 score column found.")


def available_score_columns(df: pd.DataFrame) -> list[str]:
    """Return usable Step 4 score columns in preferred model order."""
    preferred = [
        FINAL_SCORE_COL,
        "score_random_forest",
        "score_elastic_net",
        "score_baseline",
    ]
    return [
        col
        for col in dict.fromkeys(preferred)
        if col in df.columns and df[col].notna().any()
    ]


def add_daily_signal_strength(
    df: pd.DataFrame,
    score_col: str,
    basket_quantile: float = BASKET_QUANTILE,
    lookback: int = SIGNAL_SCALING_LOOKBACK,
    min_periods: int = SIGNAL_SCALING_MIN_PERIODS,
    threshold_quantile: float = SIGNAL_SCALING_QUANTILE,
    min_mult: float = SIGNAL_SCALING_MIN_MULT,
    mid_mult: float = SIGNAL_SCALING_MID_MULT,
    max_mult: float = SIGNAL_SCALING_MAX_MULT,
) -> pd.DataFrame:
    """
    Estimate daily signal strength from cross-sectional score dispersion.

    The threshold is computed from prior dates only via shift(1), so the
    exposure decision for date t does not use realised returns or future score
    dispersion. This is a cost-aware trading filter: when the model has little
    cross-sectional separation, the strategy reduces or skips gross exposure.
    """
    rows = []
    data = df.dropna(subset=[score_col]).copy()
    for date, day in data.groupby("date"):
        if len(day) < 2:
            continue
        n_side = max(1, int(np.floor(len(day) * basket_quantile)))
        ranked = day.sort_values(score_col)
        bottom = ranked.head(n_side)
        top = ranked.tail(n_side)
        rows.append(
            {
                "date": date,
                "score_spread": top[score_col].mean() - bottom[score_col].mean(),
                "n_candidates": len(day),
                "n_each_side": n_side,
            }
        )

    sig = pd.DataFrame(rows).sort_values("date")
    if sig.empty:
        return sig

    sig["spread_threshold"] = (
        sig["score_spread"]
        .shift(1)
        .rolling(lookback, min_periods=min_periods)
        .quantile(threshold_quantile)
    )
    sig["spread_threshold_high"] = 1.25 * sig["spread_threshold"]
    sig["gross_multiplier"] = np.select(
        [
            sig["score_spread"] < sig["spread_threshold"],
            sig["score_spread"] < sig["spread_threshold_high"],
        ],
        [min_mult, mid_mult],
        default=max_mult,
    )
    sig["gross_multiplier"] = sig["gross_multiplier"].fillna(max_mult)
    return sig


def add_borrow_adjusted_score(
    df: pd.DataFrame,
    score_col: str,
    tier_col: str = "htb_tier",
) -> pd.DataFrame:
    """
    Add a robustness-only short-selection score penalising expensive borrow.

    Higher scores are less attractive shorts. Adding a tier penalty makes Tier
    B/C names less likely to enter the short book, while all borrow costs remain
    charged normally if those names are still selected.
    """
    out = df.copy()
    if tier_col not in out.columns:
        return out
    score_std = out.groupby("date")[score_col].transform("std").replace(0, np.nan)
    penalty = out[tier_col].map(BORROW_SCORE_PENALTY).fillna(0.0)
    out[f"{score_col}_short_adjusted"] = out[score_col] + penalty * score_std.fillna(0.0)
    return out


def prepare_step5_input(alpha_scores: pd.DataFrame, panel_step3: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Step 4 scores with Step 3 execution and borrow fields.

    Step 4 supplies alpha scores and the realised next-overnight target used
    for backtesting. Step 3 supplies ADV20, volatility, eligibility and borrow
    tier fields needed for portfolio construction.
    """
    key = ["date", "instrument_id"]
    alpha = alpha_scores.copy()
    panel = panel_step3.copy()

    alpha["date"] = pd.to_datetime(alpha["date"])
    panel["date"] = pd.to_datetime(panel["date"])

    keep_cols = [
        "date",
        "instrument_id",
        "ticker",
        "eligible",
        "adv20",
        "vol20",
        "dsi",
        "dtcn",
        "ddtcn",
        "htb_tier",
        "htb_flag",
        "r_ON",
    ]
    keep_cols = [col for col in keep_cols if col in panel.columns]

    merged = alpha.merge(
        panel[keep_cols].drop_duplicates(key),
        on=key,
        how="left",
        suffixes=("", "_panel"),
    )

    if "ticker_panel" in merged.columns:
        merged["ticker"] = merged["ticker"].fillna(merged["ticker_panel"])
        merged = merged.drop(columns=["ticker_panel"])

    if "eligible" in merged.columns:
        merged = merged[merged["eligible"].fillna(False)].copy()

    if "target_raw" not in merged.columns:
        raise ValueError(
            "Step 5 requires target_raw from Step 4. Do not reconstruct the "
            "backtest target inside Step 5 after the eligibility merge."
        )

    merged["htb_tier"] = merged.get("htb_tier", "A")
    merged["htb_tier"] = merged["htb_tier"].fillna("A")
    merged["adv20"] = merged["adv20"].replace([np.inf, -np.inf], np.nan)
    return merged


def _raw_side_weights(side_df: pd.DataFrame, score_col: str, scheme: str) -> pd.Series:
    """Return positive weights for one side of the book."""
    if side_df.empty:
        return pd.Series(dtype=float)

    if scheme == "equal":
        weights = pd.Series(1.0, index=side_df.index)
    elif scheme == "score":
        centered = side_df[score_col] - side_df[score_col].mean()
        weights = centered.abs().replace(0.0, np.nan)
        weights = weights.fillna(weights.mean()).fillna(1.0)
    elif scheme == "vol":
        if "vol20" not in side_df.columns:
            raise ValueError("weighting_scheme='vol' requires vol20.")
        weights = 1.0 / side_df["vol20"].replace(0.0, np.nan)
        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(weights.median())
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    weights = weights.astype(float).clip(lower=0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=side_df.index)
    return weights / weights.sum()


def allocate_with_caps(
    side_df: pd.DataFrame,
    side_budget: float,
    score_col: str,
    scheme: str,
    participation_cap: float,
) -> pd.Series:
    """
    Allocate one side of the portfolio subject to ADV participation caps.

    Capped capital is redistributed across still-uncapped names. If the side
    cannot absorb the full budget, the returned allocation is smaller than the
    requested side budget rather than violating the cap.
    """
    if side_df.empty or side_budget <= 0:
        return pd.Series(dtype=float)

    caps = (side_df["adv20"] * participation_cap).fillna(0.0).clip(lower=0.0)
    weights = _raw_side_weights(side_df, score_col, scheme)
    allocation = pd.Series(0.0, index=side_df.index)
    remaining_budget = float(side_budget)
    active = weights.index[weights > 0].tolist()

    for _ in range(len(active) + 1):
        if remaining_budget <= 1e-8 or not active:
            break

        active_weights = weights.loc[active]
        active_weights = active_weights / active_weights.sum()
        desired = active_weights * remaining_budget
        remaining_cap = (caps.loc[active] - allocation.loc[active]).clip(lower=0.0)
        fill = np.minimum(desired, remaining_cap)

        allocation.loc[active] += fill
        remaining_budget -= float(fill.sum())
        active = [idx for idx in active if allocation.loc[idx] < caps.loc[idx] - 1e-8]

    return allocation


def build_daily_positions(day: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Build one day's dollar-neutral long-short positions."""
    day = day.dropna(subset=[config.score_col, "target_raw", "adv20"]).copy()
    if day.empty:
        return day.assign(position=0.0, side="none")

    n_side = max(1, int(np.floor(len(day) * config.basket_quantile)))
    long_ranked = day.sort_values(config.score_col, ascending=False).copy()
    long_ranked["score_rank"] = np.arange(1, len(long_ranked) + 1)
    long_df = long_ranked.head(n_side).copy()

    short_score_col = config.score_col
    adjusted_col = f"{config.score_col}_short_adjusted"
    if config.use_borrow_adjusted_short_score and adjusted_col in day.columns:
        short_score_col = adjusted_col
    short_ranked = day.sort_values(short_score_col, ascending=True).copy()
    short_ranked["score_rank"] = np.arange(1, len(short_ranked) + 1)
    short_df = short_ranked.head(n_side).copy()

    gross_multiplier = 1.0
    if config.use_signal_scaling and config.gross_multiplier_col in day.columns:
        gross_multiplier = float(day[config.gross_multiplier_col].iloc[0])
    gross_multiplier = float(np.clip(gross_multiplier, 0.0, 1.0))
    side_budget = config.aum * 0.5 * gross_multiplier
    if side_budget <= 0:
        return day.iloc[0:0].assign(position=0.0, side="none")
    long_alloc = allocate_with_caps(
        long_df,
        side_budget,
        config.score_col,
        config.weighting_scheme,
        config.participation_cap,
    )
    short_alloc = allocate_with_caps(
        short_df,
        side_budget,
        short_score_col,
        config.weighting_scheme,
        config.participation_cap,
    )

    long_total = float(long_alloc.sum())
    short_total = float(short_alloc.sum())
    matched_side = min(long_total, short_total, side_budget)
    if matched_side <= 0:
        return day.iloc[0:0].assign(position=0.0, side="none")

    if long_total > 0:
        long_alloc = long_alloc * (matched_side / long_total)
    if short_total > 0:
        short_alloc = short_alloc * (matched_side / short_total)

    return pd.concat(
        [
            long_df.assign(position=long_alloc, side="long"),
            short_df.assign(position=-short_alloc, side="short"),
        ],
        axis=0,
    )


def build_positions(df: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Build daily positions for the full backtest window."""
    required = {"date", "instrument_id", config.score_col, "target_raw", "adv20"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Step 5: {sorted(missing)}")

    daily_positions = []
    for date, day in df.sort_values(["date", "instrument_id"]).groupby("date"):
        positions = build_daily_positions(day, config)
        if positions.empty:
            continue
        positions["date"] = date
        daily_positions.append(positions)

    if not daily_positions:
        return pd.DataFrame()
    return pd.concat(daily_positions, ignore_index=True)


def backtest_positions(positions: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """
    Compute daily gross and net returns from position-level records.

    Cost convention:
    - gross_exposure is long notional + short notional divided by AUM.
    - roundtrip_turnover is entry plus exit notional, equal to 2 * gross_exposure.
    - the full overnight round trip costs 4 bps of gross exposure.
    """
    validate_cost_config(config)
    if positions.empty:
        raise ValueError("No positions were generated.")

    pos = positions.copy()
    pos["gross_pnl"] = pos["position"] * pos["target_raw"]
    pos["abs_position"] = pos["position"].abs()
    pos["short_notional"] = np.where(pos["position"] < 0, -pos["position"], 0.0)
    pos["participation"] = pos["abs_position"] / pos["adv20"]
    pos["participation_cap"] = config.participation_cap
    pos["cap_binding"] = pos["participation"] >= config.participation_cap - 1e-10
    pos["borrow_bps"] = pos["htb_tier"].map(BORROW_BPS_BY_TIER).fillna(BORROW_TIER_A_BPS)
    pos["borrow_cost"] = pos["short_notional"] * (pos["borrow_bps"] / 10_000) / TRADING_DAYS

    daily = (
        pos.groupby("date")
        .agg(
            gross_pnl=("gross_pnl", "sum"),
            gross_notional=("abs_position", "sum"),
            long_notional=("position", lambda x: x[x > 0].sum()),
            short_notional=("short_notional", "sum"),
            borrow_cost=("borrow_cost", "sum"),
            n_long=("position", lambda x: int((x > 0).sum())),
            n_short=("position", lambda x: int((x < 0).sum())),
            n_positions=("position", "size"),
            n_cap_binding=("cap_binding", "sum"),
            pct_cap_binding=("cap_binding", "mean"),
            max_participation=("participation", "max"),
        )
        .reset_index()
    )

    daily["gross_exposure"] = daily["gross_notional"] / config.aum
    daily["entry_turnover"] = daily["gross_exposure"]
    daily["exit_turnover"] = daily["gross_exposure"]
    daily["roundtrip_turnover"] = daily["entry_turnover"] + daily["exit_turnover"]
    daily["gross_return"] = daily["gross_pnl"] / config.aum
    daily["trading_cost_return"] = daily["gross_exposure"] * (
        config.roundtrip_bps / 10_000
    )
    daily["commission_return"] = daily["roundtrip_turnover"] * (
        config.commission_bps / 10_000
    )
    daily["slippage_return"] = daily["roundtrip_turnover"] * (
        config.slippage_bps / 10_000
    )
    daily["trading_cost"] = daily["trading_cost_return"] * config.aum
    daily["borrow_cost_return"] = daily["borrow_cost"] / config.aum
    daily["return_after_commission"] = daily["gross_return"] - daily["commission_return"]
    daily["return_after_slippage"] = daily["return_after_commission"] - daily["slippage_return"]
    daily["net_return_before_borrow"] = daily["return_after_slippage"]
    daily["net_return"] = daily["return_after_slippage"] - daily["borrow_cost_return"]
    daily["gross_return_bps"] = 10_000 * daily["gross_return"]
    daily["commission_bps"] = 10_000 * daily["commission_return"]
    daily["slippage_bps"] = 10_000 * daily["slippage_return"]
    daily["borrow_bps"] = 10_000 * daily["borrow_cost_return"]
    daily["net_return_bps"] = 10_000 * daily["net_return"]
    daily["hit_full_gross"] = daily["gross_exposure"] >= 0.999
    daily["capacity_constrained"] = ~daily["hit_full_gross"]
    return daily


def performance_summary(
    daily: pd.DataFrame,
    label: str,
    aum: float,
    score_col: str,
) -> dict[str, float | str]:
    """Summarise the required Step 5 performance and cost metrics."""
    ret = daily["net_return"].dropna()
    gross_ret = daily["gross_return"].dropna()
    after_commission = daily["return_after_commission"].dropna()
    after_slippage = daily["return_after_slippage"].dropna()
    active = daily[daily["n_positions"] > 0].copy()

    gross_sharpe = sharpe_ratio(gross_ret)
    after_commission_sharpe = sharpe_ratio(after_commission)
    after_slippage_sharpe = sharpe_ratio(after_slippage)
    net_sharpe = sharpe_ratio(ret)

    return {
        "score_col": score_col,
        "aum_label": label,
        "aum": aum,
        "basket_quantile": float(daily.attrs.get("basket_quantile", np.nan)),
        "use_signal_scaling": bool(daily.attrs.get("use_signal_scaling", False)),
        "gross_ann_return": annualised_return(gross_ret),
        "commission_ann_drag": float(daily["commission_return"].mean() * TRADING_DAYS),
        "slippage_ann_drag": float(daily["slippage_return"].mean() * TRADING_DAYS),
        "trading_cost_ann_drag": float(daily["trading_cost_return"].mean() * TRADING_DAYS),
        "borrow_cost_ann_drag": float(daily["borrow_cost_return"].mean() * TRADING_DAYS),
        "net_ann_return": annualised_return(ret),
        "net_ann_vol": annualised_volatility(ret),
        "gross_sharpe": gross_sharpe,
        "after_commission_sharpe": after_commission_sharpe,
        "after_slippage_sharpe": after_slippage_sharpe,
        "post_trading_cost_sharpe": after_slippage_sharpe,
        "net_sharpe": net_sharpe,
        "commission_sharpe_drag": gross_sharpe - after_commission_sharpe,
        "slippage_sharpe_drag": after_commission_sharpe - after_slippage_sharpe,
        "trading_cost_sharpe_drag": gross_sharpe - after_slippage_sharpe,
        "borrow_sharpe_drag": after_slippage_sharpe - net_sharpe,
        "max_drawdown": max_drawdown(ret),
        "avg_entry_turnover": float(daily["entry_turnover"].mean()),
        "avg_exit_turnover": float(daily["exit_turnover"].mean()),
        "avg_roundtrip_turnover": float(daily["roundtrip_turnover"].mean()),
        "avg_daily_turnover": float(daily["roundtrip_turnover"].mean()),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()),
        "pct_days_with_positions": float((daily["n_positions"] > 0).mean()),
        "avg_n_long_active": (
            float(active["n_long"].mean()) if not active.empty else 0.0
        ),
        "avg_n_short_active": (
            float(active["n_short"].mean()) if not active.empty else 0.0
        ),
        "avg_gross_exposure_active": (
            float(active["gross_exposure"].mean()) if not active.empty else 0.0
        ),
        "avg_daily_turnover_active": (
            float(active["roundtrip_turnover"].mean()) if not active.empty else 0.0
        ),
        "pct_days_full_target_gross": float(daily["hit_full_gross"].mean()),
        "pct_days_capacity_constrained": float(daily["capacity_constrained"].mean()),
        "avg_n_long": float(daily["n_long"].mean()),
        "avg_n_short": float(daily["n_short"].mean()),
        "avg_n_cap_binding": float(daily["n_cap_binding"].mean()),
        "avg_pct_cap_binding": float(daily["pct_cap_binding"].mean()),
        "max_participation": float(daily["max_participation"].max()),
        "avg_gross_return_bps": float(daily["gross_return_bps"].mean()),
        "avg_commission_bps": float(daily["commission_bps"].mean()),
        "avg_slippage_bps": float(daily["slippage_bps"].mean()),
        "avg_borrow_bps": float(daily["borrow_bps"].mean()),
        "avg_net_return_bps": float(daily["net_return_bps"].mean()),
        "n_days": int(len(ret)),
    }


def run_aum_backtests(
    step5_df: pd.DataFrame,
    score_col: str | None = None,
    aum_levels: dict[str, float] | None = None,
    participation_cap: float = PARTICIPATION_CAP,
    basket_quantile: float = BASKET_QUANTILE,
    use_signal_scaling: bool = False,
    use_borrow_adjusted_short_score: bool = False,
    include_all_target_dates: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Run Step 5 for all required AUM levels.

    include_all_target_dates keeps dates with no model score in the daily
    return series as zero-exposure days. This is used for the headline ML
    strategy so the 2010-2024 Section 6 window includes the pre-ML warm-up
    period without inventing in-sample model predictions.
    """
    if score_col is None:
        score_col = choose_score_column(step5_df)
    if aum_levels is None:
        aum_levels = AUM_LEVELS

    summaries = []
    daily_by_aum: dict[str, pd.DataFrame] = {}
    positions_by_aum: dict[str, pd.DataFrame] = {}

    for label, aum in aum_levels.items():
        config = PortfolioConfig(
            aum=aum,
            score_col=score_col,
            participation_cap=participation_cap,
            basket_quantile=basket_quantile,
            use_signal_scaling=use_signal_scaling,
            use_borrow_adjusted_short_score=use_borrow_adjusted_short_score,
        )
        positions = build_positions(step5_df, config)
        daily = backtest_positions(positions, config)
        if include_all_target_dates:
            expected_mask = step5_df["target_raw"].notna()
        else:
            expected_mask = step5_df[score_col].notna() & step5_df["target_raw"].notna()
        expected_dates = pd.Index(
            pd.to_datetime(step5_df.loc[expected_mask, "date"].unique())
        ).sort_values()
        daily = fill_missing_daily_returns(daily, expected_dates, config)
        daily.attrs["basket_quantile"] = basket_quantile
        daily.attrs["use_signal_scaling"] = use_signal_scaling

        summaries.append(performance_summary(daily, label, aum, score_col))
        daily_by_aum[label] = daily
        positions_by_aum[label] = positions

    return pd.DataFrame(summaries), daily_by_aum, positions_by_aum


def fill_missing_daily_returns(
    daily: pd.DataFrame,
    expected_dates: pd.Index,
    config: PortfolioConfig,
) -> pd.DataFrame:
    """Keep zero-exposure signal-scaling dates in the daily return series."""
    if expected_dates.empty:
        return daily
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    missing = expected_dates.difference(pd.Index(daily["date"]))
    if len(missing) == 0:
        return daily.sort_values("date").reset_index(drop=True)

    zero = pd.DataFrame({"date": missing})
    zero_cols = [
        "gross_pnl",
        "gross_notional",
        "long_notional",
        "short_notional",
        "borrow_cost",
        "gross_exposure",
        "entry_turnover",
        "exit_turnover",
        "roundtrip_turnover",
        "gross_return",
        "trading_cost_return",
        "commission_return",
        "slippage_return",
        "trading_cost",
        "borrow_cost_return",
        "return_after_commission",
        "return_after_slippage",
        "net_return_before_borrow",
        "net_return",
        "gross_return_bps",
        "commission_bps",
        "slippage_bps",
        "borrow_bps",
        "net_return_bps",
        "n_long",
        "n_short",
        "n_positions",
        "n_cap_binding",
        "pct_cap_binding",
        "max_participation",
    ]
    for col in zero_cols:
        zero[col] = 0.0
    zero["hit_full_gross"] = False
    zero["capacity_constrained"] = False
    out = pd.concat([daily, zero], ignore_index=True, sort=False)
    return out.sort_values("date").reset_index(drop=True)


def cap_sensitivity_analysis(
    step5_df: pd.DataFrame,
    score_col: str,
    aum_levels: dict[str, float] | None = None,
    headline_cap: float = PARTICIPATION_CAP,
    loose_cap: float = 1.0,
    basket_quantile: float = BASKET_QUANTILE,
    use_signal_scaling: bool = False,
) -> pd.DataFrame:
    """
    Compare the required 5% ADV cap with a loose-cap reference run.

    The loose-cap run is not a tradable recommendation; it is a diagnostic to
    demonstrate that the participation cap genuinely changes deployable
    exposure and cost at larger AUM.
    """
    if aum_levels is None:
        aum_levels = {"250M": AUM_LEVELS["250M"], "1B": AUM_LEVELS["1B"]}

    rows = []
    for scenario, cap in [
        ("headline_5pct_adv_cap", headline_cap),
        ("loose_100pct_adv_reference", loose_cap),
    ]:
        summary, _, positions_by_aum = run_aum_backtests(
            step5_df,
            score_col=score_col,
            aum_levels=aum_levels,
            participation_cap=cap,
            basket_quantile=basket_quantile,
            use_signal_scaling=use_signal_scaling,
        )
        audit = position_capacity_audit(
            positions_by_aum,
            aum_levels=aum_levels,
            participation_cap=cap,
        )
        merged = summary.merge(
            audit[
                [
                    "aum_label",
                    "max_single_name_participation",
                    "mean_daily_max_participation",
                    "n_days_with_cap_breach",
                ]
            ],
            on="aum_label",
            how="left",
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "scenario": scenario,
                    "participation_cap": cap,
                    "aum_label": row["aum_label"],
                    "gross_ann_return": row["gross_ann_return"],
                    "net_ann_return": row["net_ann_return"],
                    "net_sharpe": row["net_sharpe"],
                    "avg_gross_exposure": row["avg_gross_exposure"],
                    "pct_days_capacity_constrained": row["pct_days_capacity_constrained"],
                    "avg_n_cap_binding": row["avg_n_cap_binding"],
                    "max_single_name_participation": row["max_single_name_participation"],
                    "mean_daily_max_participation": row["mean_daily_max_participation"],
                    "n_days_with_cap_breach": row["n_days_with_cap_breach"],
                    "n_days": row["n_days"],
                    "interpretation": (
                        "Required tradable cap from the brief."
                        if np.isclose(cap, headline_cap)
                        else "Loose-cap diagnostic only; used to show whether cap bites."
                    ),
                }
            )
    return pd.DataFrame(rows)


def basket_size_sensitivity(
    step5_df: pd.DataFrame,
    score_col: str,
    aum: float = DEFAULT_AUM,
    basket_quantiles: tuple[float, ...] = BASKET_QUANTILES_SENSITIVITY,
    use_signal_scaling: bool = True,
) -> pd.DataFrame:
    """Run 250M sensitivity over alternative top/bottom basket sizes."""
    rows = []
    for q in basket_quantiles:
        working = step5_df.copy()
        if use_signal_scaling:
            signal = add_daily_signal_strength(working, score_col, basket_quantile=q)
            working = working.merge(
                signal[
                    [
                        "date",
                        "score_spread",
                        "spread_threshold",
                        "spread_threshold_high",
                        "gross_multiplier",
                    ]
                ],
                on="date",
                how="left",
            )
            working["gross_multiplier"] = working["gross_multiplier"].fillna(1.0)
        summary, _, _ = run_aum_backtests(
            working,
            score_col=score_col,
            aum_levels={"250M": aum},
            basket_quantile=q,
            use_signal_scaling=use_signal_scaling,
        )
        row = summary.iloc[0].to_dict()
        row["basket_quantile"] = q
        rows.append(row)
    return pd.DataFrame(rows)


def stress_window_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Report gross, cost and net performance in required stress windows."""
    rows = []
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])

    for name, (start, end) in STRESS_WINDOWS.items():
        sub = data[(data["date"] >= start) & (data["date"] <= end)]
        if sub.empty:
            continue
        rows.append(
            {
                "window": name,
                "start": start,
                "end": end,
                "n_days": int(len(sub)),
                "gross_cumulative_return": cumulative_return(sub["gross_return"]),
                "net_cumulative_return": cumulative_return(sub["net_return"]),
                "gross_sharpe": sharpe_ratio(sub["gross_return"]),
                "net_sharpe": sharpe_ratio(sub["net_return"]),
                "net_ann_vol": annualised_volatility(sub["net_return"]),
                "max_drawdown": max_drawdown(sub["net_return"]),
                "avg_daily_turnover": float(sub["roundtrip_turnover"].mean()),
                "avg_gross_exposure": float(sub["gross_exposure"].mean()),
                "avg_gross_bps": float(sub["gross_return_bps"].mean()),
                "avg_commission_bps": float(sub["commission_bps"].mean()),
                "avg_slippage_bps": float(sub["slippage_bps"].mean()),
                "avg_borrow_bps": float(sub["borrow_bps"].mean()),
                "avg_net_bps": float(sub["net_return_bps"].mean()),
                "best_day_net_return": float(sub["net_return"].max()),
                "worst_day_net_return": float(sub["net_return"].min()),
            }
        )
    return pd.DataFrame(rows)


def position_capacity_audit(
    positions_by_aum: dict[str, pd.DataFrame],
    aum_levels: dict[str, float] | None = None,
    participation_cap: float = PARTICIPATION_CAP,
) -> pd.DataFrame:
    """Audit position-level capacity and dollar-neutrality constraints."""
    if aum_levels is None:
        aum_levels = AUM_LEVELS

    rows = []
    for label, positions in positions_by_aum.items():
        pos = positions.copy()
        pos["abs_position"] = pos["position"].abs()
        pos["participation"] = pos["abs_position"] / pos["adv20"]
        pos["cap_breach"] = pos["participation"] > participation_cap + 1e-8
        pos["at_cap"] = np.isclose(pos["participation"], participation_cap, atol=1e-5)

        daily = pos.groupby("date").agg(
            long_notional=("position", lambda x: x[x > 0].sum()),
            short_notional=("position", lambda x: -x[x < 0].sum()),
            gross_notional=("abs_position", "sum"),
            max_participation=("participation", "max"),
            mean_participation=("participation", "mean"),
            pct_names_at_cap=("at_cap", "mean"),
            n_cap_breaches=("cap_breach", "sum"),
            n_long=("position", lambda x: int((x > 0).sum())),
            n_short=("position", lambda x: int((x < 0).sum())),
        )
        daily["long_short_imbalance"] = (
            daily["long_notional"] - daily["short_notional"]
        ).abs()
        daily["gross_exposure"] = daily["gross_notional"] / aum_levels[label]

        rows.append(
            {
                "aum_label": label,
                "max_single_name_participation": float(daily["max_participation"].max()),
                "mean_daily_max_participation": float(daily["max_participation"].mean()),
                "mean_participation_all_positions": float(pos["participation"].mean()),
                "pct_positions_at_cap": float(pos["at_cap"].mean()),
                "n_days_with_cap_breach": int((daily["n_cap_breaches"] > 0).sum()),
                "max_long_short_imbalance_dollar": float(
                    daily["long_short_imbalance"].max()
                ),
                "avg_long_short_imbalance_dollar": float(
                    daily["long_short_imbalance"].mean()
                ),
                "avg_n_long": float(daily["n_long"].mean()),
                "avg_n_short": float(daily["n_short"].mean()),
                "avg_gross_exposure": float(daily["gross_exposure"].mean()),
            }
        )

    return pd.DataFrame(rows)


def borrow_tier_audit(positions_by_aum: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise short exposure, borrow cost and concentration by tier."""
    rows = []
    for label, positions in positions_by_aum.items():
        pos = positions.copy()
        pos["short_notional"] = np.where(pos["position"] < 0, -pos["position"], 0.0)
        pos["borrow_bps"] = pos["htb_tier"].map(BORROW_BPS_BY_TIER).fillna(BORROW_TIER_A_BPS)
        pos["borrow_cost"] = pos["short_notional"] * (pos["borrow_bps"] / 10_000) / TRADING_DAYS
        tier = (
            pos.groupby("htb_tier")
            .agg(
                total_short_notional=("short_notional", "sum"),
                total_borrow_cost=("borrow_cost", "sum"),
                n_rows=("htb_tier", "size"),
            )
            .reset_index()
        )
        short_total = tier["total_short_notional"].sum()
        borrow_total = tier["total_borrow_cost"].sum()
        tier["short_notional_share"] = (
            tier["total_short_notional"] / short_total if short_total else np.nan
        )
        tier["borrow_cost_share"] = (
            tier["total_borrow_cost"] / borrow_total if borrow_total else np.nan
        )
        tier.insert(0, "aum_label", label)
        rows.append(tier)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def impact_cap_summary(
    positions_by_aum: dict[str, pd.DataFrame],
    k: float = 0.7,
    participation_cap: float = PARTICIPATION_CAP,
    sensitivity_caps: tuple[float, ...] = (0.01, 0.025, 0.05),
) -> pd.DataFrame:
    """
    Summarise square-root impact at several participation caps.

    This is diagnostic only. Headline results still use the fixed Brief Section
    6 cost schedule. The 5% cap should be read as an aggressive upper bound
    under this diagnostic, not as a guarantee of low impact.
    """
    rows = []
    for label, positions in positions_by_aum.items():
        if "vol20" not in positions.columns:
            continue
        pos = positions.dropna(subset=["vol20"]).copy()
        pos["daily_vol"] = pos["vol20"] / np.sqrt(TRADING_DAYS)

        buckets = pd.qcut(
            pos["adv20"],
            3,
            labels=["low_adv", "mid_adv", "high_adv"],
            duplicates="drop",
        )
        for cap in sensitivity_caps:
            pos["sqrt_impact_bps"] = k * pos["daily_vol"] * np.sqrt(cap) * 10_000
            for bucket, group in pos.groupby(buckets, observed=True):
                rows.append(
                    {
                        "aum_label": label,
                        "adv_bucket": str(bucket),
                        "median_adv20": float(group["adv20"].median()),
                        "median_annual_vol20": float(group["vol20"].median()),
                        "participation_cap": cap,
                        "is_headline_cap": bool(np.isclose(cap, participation_cap)),
                        "median_sqrt_impact_bps": float(group["sqrt_impact_bps"].median()),
                        "diagnostic_interpretation": (
                            "5% is an aggressive upper bound under the square-root "
                            "diagnostic; headline backtest costs follow the fixed "
                            "Brief Section 6.3 schedule."
                            if np.isclose(cap, participation_cap)
                            else "Lower participation sensitivity diagnostic."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def lo_adjusted_sharpe(returns: pd.Series, max_lag: int = 5) -> float:
    """
    Lo-style autocorrelation-adjusted annual Sharpe.

    The adjustment inflates daily volatility by the sample autocorrelation up to
    max_lag. This is a robustness diagnostic; headline Sharpe remains the
    standard annualised daily Sharpe used elsewhere in Step 5.
    """
    ret = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) <= max_lag + 2:
        return np.nan

    demeaned = ret - ret.mean()
    mean_daily = ret.mean()
    gamma0 = float(np.mean(demeaned * demeaned))
    if gamma0 <= 0 or np.isnan(gamma0):
        return np.nan

    adjustment = 1.0
    for lag in range(1, max_lag + 1):
        rho = ret.autocorr(lag=lag)
        if np.isnan(rho):
            continue
        adjustment += 2.0 * (1.0 - lag / (max_lag + 1.0)) * rho
    adjustment = max(adjustment, 1e-8)
    adjusted_daily_vol = np.sqrt(gamma0 * adjustment)
    return float((mean_daily / adjusted_daily_vol) * np.sqrt(TRADING_DAYS))


def rolling_window_diagnostics(
    daily: pd.DataFrame,
    return_col: str = "net_return",
    windows: tuple[int, ...] = (63, 126, 252),
) -> pd.DataFrame:
    """Worst rolling 3, 6 and 12 month compounded returns."""
    rows = []
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    series = data.set_index("date")[return_col].dropna().sort_index()

    for window in windows:
        if len(series) < window:
            continue
        rolling = (1.0 + series).rolling(window).apply(np.prod, raw=True) - 1.0
        rolling = rolling.dropna()
        if rolling.empty:
            continue
        worst_end = rolling.idxmin()
        rows.append(
            {
                "return_col": return_col,
                "window_days": window,
                "approx_months": round(window / 21),
                "worst_cumulative_return": float(rolling.min()),
                "worst_window_start": worst_end - pd.tseries.offsets.BDay(window - 1),
                "worst_window_end": worst_end,
            }
        )
    return pd.DataFrame(rows)


def non_overlapping_subperiod_summary(
    daily: pd.DataFrame,
    return_col: str = "net_return",
    windows: tuple[tuple[str, str, str], ...] = (
        ("2010_2012", "2010-01-01", "2012-12-31"),
        ("2013_2016", "2013-01-01", "2016-12-31"),
        ("2017_2019", "2017-01-01", "2019-12-31"),
        ("2020_2022", "2020-01-01", "2022-12-31"),
        ("2023_2024", "2023-01-01", "2024-12-31"),
    ),
) -> pd.DataFrame:
    """
    Summarise Sharpe stability across non-overlapping calendar subperiods.

    This directly supports the Section 7 self-check question on whether Sharpe
    is stable across non-overlapping windows.
    """
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    rows = []
    for label, start, end in windows:
        window = data[
            (data["date"] >= pd.Timestamp(start))
            & (data["date"] <= pd.Timestamp(end))
        ]
        ret = window[return_col].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "subperiod": label,
                "start": start,
                "end": end,
                "n_days": int(len(ret)),
                "cumulative_return": cumulative_return(ret),
                "ann_return": annualised_return(ret),
                "ann_vol": annualised_volatility(ret),
                "sharpe": sharpe_ratio(ret),
                "max_drawdown": max_drawdown(ret),
                "avg_gross_exposure": (
                    float(window["gross_exposure"].mean())
                    if "gross_exposure" in window.columns and not window.empty
                    else np.nan
                ),
                "avg_daily_turnover": (
                    float(window["roundtrip_turnover"].mean())
                    if "roundtrip_turnover" in window.columns and not window.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def robustness_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Additional Step 6 statistical robustness checks for the 250M portfolio.

    These rows are intended to answer common marker questions directly from a
    reproducible output table.
    """
    ret = daily["net_return"].replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return pd.DataFrame()

    top_n = max(1, int(np.ceil(len(ret) * 0.05)))
    top_5pct_sum = ret.nlargest(top_n).sum()
    total_sum = ret.sum()

    rows = [
        {
            "diagnostic": "standard_net_sharpe",
            "value": sharpe_ratio(ret),
            "detail": "Annualised daily Sharpe with zero risk-free rate.",
        },
        {
            "diagnostic": "lo_adjusted_net_sharpe_lag5",
            "value": lo_adjusted_sharpe(ret, max_lag=5),
            "detail": "Autocorrelation-adjusted Sharpe using lags 1-5.",
        },
        {
            "diagnostic": "lag1_autocorrelation",
            "value": float(ret.autocorr(lag=1)),
            "detail": "Daily net-return autocorrelation.",
        },
        {
            "diagnostic": "top_5pct_days_net_return_sum",
            "value": float(top_5pct_sum),
            "detail": f"Sum of the best {top_n} daily net returns.",
        },
        {
            "diagnostic": "total_net_return_sum",
            "value": float(total_sum),
            "detail": "Arithmetic sum of all daily net returns.",
        },
        {
            "diagnostic": "top_5pct_days_share_of_total_sum",
            "value": float(top_5pct_sum / total_sum) if total_sum else np.nan,
            "detail": "Negative values can occur when total net return is negative.",
        },
    ]

    rolling = rolling_window_diagnostics(daily)
    for _, row in rolling.iterrows():
        rows.append(
            {
                "diagnostic": f"worst_rolling_{int(row['approx_months'])}m_return",
                "value": row["worst_cumulative_return"],
                "detail": (
                    f"{int(row['window_days'])} trading-day window ending "
                    f"{pd.Timestamp(row['worst_window_end']).date()}."
                ),
            }
        )

    dated_ret = daily[["date", "net_return"]].copy()
    dated_ret["date"] = pd.to_datetime(dated_ret["date"])
    dated_ret = dated_ret.dropna(subset=["net_return"]).set_index("date")["net_return"]
    for year, group in dated_ret.groupby(dated_ret.index.year):
        rows.append(
            {
                "diagnostic": f"annual_net_sharpe_{year}",
                "value": sharpe_ratio(group),
                "detail": f"Net Sharpe for calendar year {year}.",
            }
        )
    return pd.DataFrame(rows)


def borrow_sensitivity_analysis(
    step5_df: pd.DataFrame,
    base_summary: pd.DataFrame,
    base_positions_250m: pd.DataFrame,
    score_col: str,
    aum: float = DEFAULT_AUM,
    basket_quantile: float = BASKET_QUANTILE,
    use_signal_scaling: bool = False,
) -> pd.DataFrame:
    """
    Compare tiered borrow costs with hard-borrow exclusion scenarios.

    This directly answers two common audit questions:
    - how much gross PnL came from short-interest > 10% names?
    - does removing expensive borrow names change the conclusion?

    The caller should pass the same analysis window for step5_df, base_summary
    and base_positions_250m so the hard-exclusion rows are comparable to the
    baseline row. basket_quantile and use_signal_scaling must also match the
    baseline run, otherwise hard-exclusion comparisons are not like-for-like.
    """
    base_row = base_summary.loc[base_summary["aum"] == aum].iloc[0].to_dict()
    rows = [
        {
            "scenario": "tiered_borrow_all_names",
            "excluded_rule": "None; apply A/B/C borrow tiers.",
            "gross_ann_return": base_row["gross_ann_return"],
            "net_ann_return": base_row["net_ann_return"],
            "gross_sharpe": base_row["gross_sharpe"],
            "net_sharpe": base_row["net_sharpe"],
            "borrow_cost_ann_drag": base_row["borrow_cost_ann_drag"],
            "avg_gross_exposure": base_row["avg_gross_exposure"],
            "n_days": base_row["n_days"],
        }
    ]

    scenarios = [
        (
            "hard_exclude_tier_C",
            "Remove Tier C names before portfolio construction.",
            step5_df["htb_tier"].fillna("A") != "C",
        ),
        (
            "hard_exclude_tier_BC",
            "Remove Tier B and Tier C names before portfolio construction.",
            ~step5_df["htb_tier"].fillna("A").isin(["B", "C"]),
        ),
    ]
    for scenario, excluded_rule, mask in scenarios:
        scenario_summary, _, _ = run_aum_backtests(
            step5_df[mask].copy(),
            score_col=score_col,
            aum_levels={"250M": aum},
            basket_quantile=basket_quantile,
            use_signal_scaling=use_signal_scaling,
        )
        row = scenario_summary.iloc[0].to_dict()
        rows.append(
            {
                "scenario": scenario,
                "excluded_rule": excluded_rule,
                "gross_ann_return": row["gross_ann_return"],
                "net_ann_return": row["net_ann_return"],
                "gross_sharpe": row["gross_sharpe"],
                "net_sharpe": row["net_sharpe"],
                "borrow_cost_ann_drag": row["borrow_cost_ann_drag"],
                "avg_gross_exposure": row["avg_gross_exposure"],
                "n_days": row["n_days"],
            }
        )

    pos = base_positions_250m.copy()
    if "dsi" in pos.columns:
        pos["high_short_interest"] = pos["dsi"] > 0.10
        high_si_short = pos[(pos["position"] < 0) & pos["high_short_interest"]].copy()
        total_short = pos[pos["position"] < 0].copy()
        high_gross_pnl = float((high_si_short["position"] * high_si_short["target_raw"]).sum())
        total_short_pnl = float((total_short["position"] * total_short["target_raw"]).sum())
        n_days = max(1, pos["date"].nunique())
        rows.append(
            {
                "scenario": "short_interest_gt_10pct_short_book_contribution",
                "excluded_rule": "Diagnostic only; no names excluded.",
                "gross_ann_return": np.nan,
                "net_ann_return": np.nan,
                "gross_sharpe": np.nan,
                "net_sharpe": np.nan,
                "borrow_cost_ann_drag": np.nan,
                "avg_gross_exposure": np.nan,
                "n_days": n_days,
                "gross_bps_per_day": high_gross_pnl / aum / n_days * 10_000,
                "share_of_short_side_gross_pnl": (
                    high_gross_pnl / total_short_pnl if total_short_pnl else np.nan
                ),
                "avg_short_notional_share": (
                    high_si_short["position"].abs().sum() / total_short["position"].abs().sum()
                    if not total_short.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_calendar_year_net_returns(
    daily: pd.DataFrame,
    benchmark: pd.Series,
    output_path: Path,
    strategy_label: str = "250M Random Forest strategy net",
) -> None:
    """Plot calendar-year compounded net returns against SP500_TR."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    strategy = daily.set_index("date")["net_return"].sort_index().rename(strategy_label)
    strategy.index = pd.to_datetime(strategy.index)
    bench = benchmark.sort_index().rename("SP500_TR")
    merged = pd.concat([strategy, bench], axis=1).dropna()
    annual = (
        merged.assign(year=merged.index.year)
        .groupby("year")
        .apply(lambda x: (1.0 + x[[strategy_label, "SP500_TR"]]).prod() - 1.0)
    )

    x = np.arange(len(annual.index))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 5.4), dpi=180)
    ax.bar(x - width / 2, annual["SP500_TR"], width=width, label="SP500_TR", color="#f2c94c")
    ax.bar(x + width / 2, annual[strategy_label], width=width, label=strategy_label, color="#2f91bd")
    ax.axhline(0, color="#333333", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(annual.index.astype(str), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Annual return")
    ax.set_title("Calendar-Year Net Returns vs SP500_TR")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_return_quantiles(
    daily: pd.DataFrame,
    output_path: Path,
    title: str = "250M Elastic Net Strategy Net Return Quantiles",
) -> None:
    """Plot daily, weekly, monthly, quarterly and yearly net return distributions."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    series = daily.set_index("date")["net_return"].dropna().sort_index()
    series.index = pd.to_datetime(series.index)
    periods = {
        "Daily": series,
        "Weekly": (1.0 + series).resample("W-FRI").prod() - 1.0,
        "Monthly": (1.0 + series).resample("ME").prod() - 1.0,
        "Quarterly": (1.0 + series).resample("QE").prod() - 1.0,
        "Yearly": (1.0 + series).resample("YE").prod() - 1.0,
    }

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    box = ax.boxplot(
        [ret.dropna().values for ret in periods.values()],
        tick_labels=list(periods.keys()),
        patch_artist=True,
        showfliers=True,
    )
    colors = ["#5b8a9a", "#34a3b8", "#f2c94c", "#f2994a", "#9b7cc0"]
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=0.85)
    for median in box["medians"]:
        median.set(color="#222222", linewidth=1.4)
    ax.axhline(0, color="#333333", linewidth=1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Period return")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def figure_captions(
    strategy_name: str = "Random Forest",
    quantile_figure: str = "assets/250M_random_forest_strategy_return_quantiles.png",
    basket_quantile: float = BASKET_QUANTILE,
    use_signal_scaling: bool = False,
) -> pd.DataFrame:
    """Captions with AUM, basket size and cap assumptions for all Step 5 figures."""
    scaling_text = (
        "with trailing score-dispersion signal-strength gross scaling"
        if use_signal_scaling
        else "with full target gross exposure"
    )
    basket_pct = f"{basket_quantile:.1%}"
    return pd.DataFrame(
        [
            {
                "figure": "gross_to_net_decomposition.png",
                "caption": (
                    "Gross-to-net annualised return decomposition for 50M, 250M "
                    f"and 1B AUM. Portfolio uses top/bottom {basket_pct} baskets, equal "
                    "weights within each side, 5% ADV single-name cap, daily "
                    f"dollar-neutral matching after caps, {scaling_text}, and "
                    "the fixed Table 6.1 cost schedule."
                ),
            },
            {
                "figure": "assets/calendar_year_net_returns_vs_SP500_TR.png",
                "caption": (
                    f"Calendar-year compounded net returns for the 250M {strategy_name} "
                    "Step 4 ML strategy versus SP500_TR. Portfolio uses "
                    f"top/bottom {basket_pct} baskets, equal side weights, "
                    f"{scaling_text}, and a 5% ADV cap."
                ),
            },
            {
                "figure": quantile_figure,
                "caption": (
                    f"Distribution of 250M {strategy_name} strategy net returns after "
                    "commission, slippage and borrow. Daily returns are compounded "
                    "to weekly, monthly, quarterly and yearly periods. Assumptions: "
                    f"top/bottom {basket_pct} baskets, equal weights, "
                    f"{scaling_text}, 5% ADV cap."
                ),
            },
        ]
    )


def plot_gross_to_net_decomposition(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot annual return decomposition from gross alpha to net return."""
    import matplotlib.pyplot as plt

    plot_df = summary.set_index("aum_label")[
        [
            "gross_ann_return",
            "commission_ann_drag",
            "slippage_ann_drag",
            "borrow_cost_ann_drag",
            "net_ann_return",
        ]
    ]
    x = np.arange(len(plot_df.index))
    width = 0.16

    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.bar(x - 2 * width, plot_df["gross_ann_return"], width, label="Gross alpha")
    ax.bar(x - width, -plot_df["commission_ann_drag"], width, label="Commission")
    ax.bar(x, -plot_df["slippage_ann_drag"], width, label="Slippage")
    ax.bar(x + width, -plot_df["borrow_cost_ann_drag"], width, label="Borrow")
    ax.bar(x + 2 * width, plot_df["net_ann_return"], width, label="Net")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index)
    ax.set_ylabel("Annualised return contribution")
    ax.set_title("Diagnostic Gross-to-Net Return Decomposition")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_quantstats_tearsheet(
    daily: pd.DataFrame,
    output_path: Path,
    benchmark: pd.Series | None = None,
    title: str = "C2O Step 5 250M Strategy Net Returns vs SP500_TR",
    end_date: str | None = TRAIN_END,
) -> None:
    """Create the required QuantStats HTML report for the 250M portfolio."""
    import quantstats as qs

    returns = daily.set_index("date")["net_return"].sort_index()
    returns.index = pd.to_datetime(returns.index)
    if end_date is not None:
        returns = returns[returns.index <= pd.Timestamp(end_date)]
    if benchmark is not None:
        benchmark = benchmark.sort_index()
        benchmark.index = pd.to_datetime(benchmark.index)
        if end_date is not None:
            benchmark = benchmark[benchmark.index <= pd.Timestamp(end_date)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=str(output_path),
        title=title,
    )
