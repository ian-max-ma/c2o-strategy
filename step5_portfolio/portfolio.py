"""Step 5 portfolio construction and costed backtest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AUM_LEVELS,
    BASKET_QUANTILE,
    BORROW_TIER_A_BPS,
    BORROW_TIER_B_BPS,
    BORROW_TIER_C_BPS,
    COMMISSION_BPS,
    DEFAULT_AUM,
    PARTICIPATION_CAP,
    ROUNDTRIP_BPS,
    SLIPPAGE_BPS,
    TRADING_DAYS,
    TRAIN_END,
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
}


@dataclass(frozen=True)
class PortfolioConfig:
    aum: float = DEFAULT_AUM
    score_col: str = "score_elastic_net"
    basket_quantile: float = BASKET_QUANTILE
    participation_cap: float = PARTICIPATION_CAP
    weighting_scheme: str = WEIGHTING_SCHEME
    commission_bps: float = COMMISSION_BPS
    slippage_bps: float = SLIPPAGE_BPS
    roundtrip_bps: float = ROUNDTRIP_BPS
    per_leg_cost_bps: float = COMMISSION_BPS + SLIPPAGE_BPS


def validate_cost_config(config: PortfolioConfig) -> None:
    implied_roundtrip = 2.0 * config.per_leg_cost_bps
    if not np.isclose(config.roundtrip_bps, implied_roundtrip):
        raise ValueError(
            f"Expected roundtrip_bps={implied_roundtrip:.2f}, got {config.roundtrip_bps:.2f}."
        )


def cost_schedule_summary(config: PortfolioConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = PortfolioConfig()
    validate_cost_config(config)
    return pd.DataFrame(
        [
            {"item": "commission_per_leg", "bps": config.commission_bps, "frequency": "entry and exit", "applies_to": "traded notional"},
            {"item": "auction_slippage_per_leg", "bps": config.slippage_bps, "frequency": "entry and exit", "applies_to": "traded notional"},
            {"item": "total_per_leg", "bps": config.per_leg_cost_bps, "frequency": "entry or exit", "applies_to": "traded notional"},
            {"item": "roundtrip_commission_slippage", "bps": config.roundtrip_bps, "frequency": "daily overnight round trip", "applies_to": "gross exposure"},
            {"item": "tier_A_borrow", "bps": BORROW_TIER_A_BPS, "frequency": "annual, charged daily / 252", "applies_to": "short notional"},
            {"item": "tier_B_borrow", "bps": BORROW_TIER_B_BPS, "frequency": "annual, charged daily / 252", "applies_to": "short notional"},
            {"item": "tier_C_borrow", "bps": BORROW_TIER_C_BPS, "frequency": "annual, charged daily / 252", "applies_to": "short notional"},
        ]
    )


def annualised_return(returns: pd.Series) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1.0)


def annualised_volatility(returns: pd.Series) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    vol = annualised_volatility(returns)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return annualised_return(returns) / vol


def cumulative_return(returns: pd.Series) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def choose_score_column(df: pd.DataFrame) -> str:
    for col in ["score_random_forest", "score_elastic_net", "score_baseline"]:
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError("No usable Step 4 score column found.")


def available_score_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["score_random_forest", "score_elastic_net", "score_baseline"]
    return [col for col in preferred if col in df.columns and df[col].notna().any()]


def prepare_step5_input(alpha_scores: pd.DataFrame, panel_step3: pd.DataFrame) -> pd.DataFrame:
    alpha = alpha_scores.copy()
    panel = panel_step3.copy()
    alpha["date"] = pd.to_datetime(alpha["date"])
    panel["date"] = pd.to_datetime(panel["date"])

    keep_cols = [
        "date", "instrument_id", "ticker", "eligible", "adv20", "vol20", "dsi", "dtcn", "ddtcn", "htb_tier", "htb_flag", "r_ON"
    ]
    keep_cols = [c for c in keep_cols if c in panel.columns]

    merged = alpha.merge(panel[keep_cols].drop_duplicates(["date", "instrument_id"]), on=["date", "instrument_id"], how="left")
    if "ticker_panel" in merged.columns:
        merged = merged.drop(columns=["ticker_panel"])
    if "eligible" in merged.columns:
        merged["eligible"] = merged["eligible"].fillna(True)
    if "target_raw" not in merged.columns:
        merged["target_raw"] = merged.groupby("instrument_id")["r_ON"].shift(-1)
    merged["htb_tier"] = merged.get("htb_tier", "A").fillna("A")
    merged["adv20"] = merged["adv20"].replace([np.inf, -np.inf], np.nan)
    return merged


def _raw_side_weights(side_df: pd.DataFrame, score_col: str, scheme: str) -> pd.Series:
    if side_df.empty:
        return pd.Series(dtype=float)
    scores = side_df[score_col].astype(float).fillna(0.0)
    if scheme == "equal":
        weights = pd.Series(1.0, index=side_df.index)
    elif scheme == "score":
        weights = scores.abs()
    elif scheme == "vol":
        weights = 1.0 / np.maximum(side_df.get("vol20", 1.0).astype(float).fillna(1.0), 1e-6)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    weights = weights.astype(float).clip(lower=0.0)
    if weights.sum() <= 0:
        return pd.Series(0.0, index=side_df.index, dtype=float)
    return weights / weights.sum()


def allocate_with_caps(side_df: pd.DataFrame, side_budget: float, score_col: str, scheme: str, participation_cap: float) -> pd.Series:
    if side_df.empty or side_budget <= 0:
        return pd.Series(0.0, index=side_df.index, dtype=float)

    caps = (side_df["adv20"] * participation_cap).fillna(0.0).clip(lower=0.0)
    weights = _raw_side_weights(side_df, score_col, scheme)
    allocation = pd.Series(0.0, index=side_df.index, dtype=float)
    remaining_budget = float(side_budget)

    order = weights.sort_values(ascending=False).index.tolist()
    for idx in order:
        if remaining_budget <= 0:
            break
        raw = weights.loc[idx] * side_budget
        alloc = min(raw, caps.loc[idx], remaining_budget)
        allocation.loc[idx] = alloc
        remaining_budget -= alloc

    if remaining_budget > 0:
        for idx in order:
            if allocation.loc[idx] > 0:
                continue
            alloc = min(side_budget * weights.loc[idx], caps.loc[idx], remaining_budget)
            if alloc > 0:
                allocation.loc[idx] = alloc
                remaining_budget -= alloc
    return allocation


def build_daily_positions(day: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    day = day.dropna(subset=[config.score_col, "target_raw", "adv20"]).copy()
    if day.empty:
        return pd.DataFrame(columns=["date", "instrument_id", config.score_col, "position", "side"])

    n_side = max(1, int(np.floor(len(day) * config.basket_quantile)))
    ranked = day.sort_values(config.score_col, ascending=False).copy()
    ranked["score_rank"] = np.arange(1, len(ranked) + 1)
    long_df = ranked.head(n_side).copy()
    short_df = ranked.tail(n_side).copy()

    side_budget = config.aum * 0.5
    long_alloc = allocate_with_caps(long_df, side_budget, config.score_col, config.weighting_scheme, config.participation_cap)
    short_alloc = allocate_with_caps(short_df, side_budget, config.score_col, config.weighting_scheme, config.participation_cap)

    long_total = float(long_alloc.sum())
    short_total = float(short_alloc.sum())
    matched_side = min(long_total, short_total, side_budget)
    if matched_side <= 0:
        return pd.DataFrame(columns=["date", "instrument_id", config.score_col, "position", "side"])

    if long_total > 0:
        long_alloc = long_alloc * (matched_side / long_total)
    if short_total > 0:
        short_alloc = short_alloc * (matched_side / short_total)

    long_positions = long_df.assign(position=long_alloc, side="long")
    short_positions = short_df.assign(position=-short_alloc, side="short")
    return pd.concat([long_positions, short_positions], axis=0, ignore_index=True)


def build_positions(df: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    required = {"date", "instrument_id", config.score_col, "target_raw", "adv20"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    daily_positions = []
    for _, day in df.sort_values(["date", "instrument_id"]).groupby("date", sort=True):
        pos = build_daily_positions(day, config)
        if not pos.empty:
            daily_positions.append(pos)
    if not daily_positions:
        return pd.DataFrame(columns=["date", "instrument_id", config.score_col, "position", "side"])
    return pd.concat(daily_positions, ignore_index=True)


def backtest_positions(positions: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    validate_cost_config(config)
    if positions.empty:
        return pd.DataFrame(columns=["date", "gross_return", "net_return", "gross_exposure", "roundtrip_turnover", "trading_cost_return", "commission_return", "slippage_return", "borrow_cost_return"])

    pos = positions.copy()
    pos["gross_pnl"] = pos["position"] * pos["target_raw"]
    pos["abs_position"] = pos["position"].abs()
    pos["short_notional"] = np.where(pos["position"] < 0, -pos["position"], 0.0)
    pos["participation"] = pos["abs_position"] / pos["adv20"].replace(0, np.nan)
    pos["participation_cap"] = config.participation_cap
    pos["cap_binding"] = pos["participation"] >= config.participation_cap - 1e-10
    pos["borrow_bps"] = pos.get("htb_tier", pd.Series("A", index=pos.index)).map(BORROW_BPS_BY_TIER).fillna(BORROW_TIER_A_BPS)
    pos["borrow_cost"] = pos["short_notional"] * (pos["borrow_bps"] / 10_000.0) / TRADING_DAYS

    daily = (
        pos.groupby("date", sort=True)
        .agg(
            gross_pnl=("gross_pnl", "sum"),
            gross_notional=("abs_position", "sum"),
            long_notional=("position", lambda s: float(s[s > 0].sum())),
            short_notional=("short_notional", "sum"),
            borrow_cost=("borrow_cost", "sum"),
            n_long=("position", lambda s: int((s > 0).sum())),
            n_short=("position", lambda s: int((s < 0).sum())),
            n_positions=("position", "size"),
            n_cap_binding=("cap_binding", "sum"),
            pct_cap_binding=("cap_binding", "mean"),
            max_participation=("participation", "max"),
        )
        .reset_index()
    )

    daily["gross_return"] = daily["gross_pnl"] / config.aum
    daily["gross_exposure"] = daily["gross_notional"] / config.aum
    daily["roundtrip_turnover"] = 2.0 * daily["gross_exposure"]
    daily["commission_return"] = daily["gross_exposure"] * (config.commission_bps * 2.0 / 10_000.0)
    daily["slippage_return"] = daily["gross_exposure"] * (config.slippage_bps * 2.0 / 10_000.0)
    daily["trading_cost_return"] = daily["commission_return"] + daily["slippage_return"]
    daily["borrow_cost_return"] = daily["borrow_cost"] / config.aum
    daily["net_return"] = daily["gross_return"] - daily["trading_cost_return"] - daily["borrow_cost_return"]
    return daily


def performance_summary(daily: pd.DataFrame, label: str, aum: float, score_col: str) -> dict[str, float | str]:
    returns = pd.Series(daily["net_return"].dropna().to_numpy(), name=label)
    gross = pd.Series(daily["gross_return"].dropna().to_numpy(), name=label)
    ann = annualised_return(returns)
    vol = annualised_volatility(returns)
    sharpe = sharpe_ratio(returns)
    return {
        "label": label,
        "aum": float(aum),
        "score_col": score_col,
        "n_days": int(len(daily)),
        "gross_ann_return": annualised_return(gross),
        "net_ann_return": ann,
        "gross_sharpe": sharpe_ratio(gross),
        "net_sharpe": sharpe,
        "max_drawdown": max_drawdown(returns),
        "avg_gross_exposure": float(daily["gross_exposure"].mean()) if "gross_exposure" in daily else 0.0,
        "pct_days_full_target_gross": float((daily["gross_exposure"] >= 1.0).mean()) if "gross_exposure" in daily else 0.0,
        "pct_days_capacity_constrained": float(daily["pct_cap_binding"].mean()) if "pct_cap_binding" in daily else 0.0,
        "avg_n_long": float(daily["n_long"].mean()) if "n_long" in daily else 0.0,
        "avg_n_short": float(daily["n_short"].mean()) if "n_short" in daily else 0.0,
        "avg_n_cap_binding": float(daily["n_cap_binding"].mean()) if "n_cap_binding" in daily else 0.0,
        "avg_pct_cap_binding": float(daily["pct_cap_binding"].mean()) if "pct_cap_binding" in daily else 0.0,
        "max_participation": float(daily["max_participation"].max()) if "max_participation" in daily else 0.0,
        "avg_gross_return_bps": float(daily["gross_return"].mean() * 10_000.0) if "gross_return" in daily else 0.0,
        "avg_net_return_bps": float(daily["net_return"].mean() * 10_000.0) if "net_return" in daily else 0.0,
    }


def run_aum_backtests(step5_df: pd.DataFrame, score_col: str | None = None, aum_levels: dict[str, float] | None = None, participation_cap: float = PARTICIPATION_CAP) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if score_col is None:
        score_col = choose_score_column(step5_df)
    if aum_levels is None:
        aum_levels = AUM_LEVELS
    summary_rows = []
    daily_by_aum = {}
    positions_by_aum = {}

    for label, aum in aum_levels.items():
        cfg = PortfolioConfig(aum=aum, score_col=score_col, participation_cap=participation_cap)
        positions = build_positions(step5_df, cfg)
        daily = backtest_positions(positions, cfg)
        summary_rows.append(performance_summary(daily, label, aum, score_col))
        daily_by_aum[label] = daily
        positions_by_aum[label] = positions

    summary = pd.DataFrame(summary_rows)
    return summary, daily_by_aum, positions_by_aum


def cap_sensitivity_analysis(step5_df: pd.DataFrame, score_col: str, aum_levels: dict[str, float] | None = None, headline_cap: float = PARTICIPATION_CAP, loose_cap: float = 1.0) -> pd.DataFrame:
    if aum_levels is None:
        aum_levels = {"250M": AUM_LEVELS["250M"]}
    rows = []
    for scenario, cap in (("headline_5pct_adv_cap", headline_cap), ("loose_100pct_adv_reference", loose_cap)):
        summary, _, _ = run_aum_backtests(step5_df, score_col=score_col, aum_levels=aum_levels, participation_cap=cap)
        row = summary.iloc[0].copy()
        row["scenario"] = scenario
        rows.append(row)
    return pd.DataFrame(rows)


def stress_window_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (start, end) in STRESS_WINDOWS.items():
        sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]
        rows.append({"window": name, "n_days": int(len(sub)), "avg_net_return_bps": float(sub["net_return"].mean() * 10_000.0) if "net_return" in sub else 0.0})
    return pd.DataFrame(rows)


def position_capacity_audit(positions_by_aum: dict[str, pd.DataFrame], aum_levels: dict[str, float] | None = None, participation_cap: float = PARTICIPATION_CAP) -> pd.DataFrame:
    if aum_levels is None:
        aum_levels = AUM_LEVELS
    rows = []
    for label, positions in positions_by_aum.items():
        aum = aum_levels.get(label, 0.0)
        pos = positions.copy()
        pos["abs_position"] = pos["position"].abs()
        pos["participation"] = pos["abs_position"] / pos["adv20"].replace(0, np.nan)
        rows.append({"aum_label": label, "n_days_with_cap_breach": int((pos["participation"] > participation_cap).any()), "max_single_name_participation": float(pos["participation"].max()) if not pos.empty else 0.0})
    return pd.DataFrame(rows)


def borrow_tier_audit(positions_by_aum: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, positions in positions_by_aum.items():
        tiers = positions.get("htb_tier", pd.Series("A", index=positions.index)).fillna("A")
        rows.append({"aum_label": label, "tier_A": int((tiers == "A").sum()), "tier_B": int((tiers == "B").sum()), "tier_C": int((tiers == "C").sum())})
    return pd.DataFrame(rows)


def impact_cap_summary(positions_by_aum: dict[str, pd.DataFrame], k: float = 0.7, participation_cap: float = PARTICIPATION_CAP, sensitivity_caps: tuple[float, ...] = (0.01, 0.025, 0.05)) -> pd.DataFrame:
    rows = []
    for label, positions in positions_by_aum.items():
        pos = positions.copy()
        pos["abs_position"] = pos["position"].abs()
        pos["participation"] = pos["abs_position"] / pos["adv20"].replace(0, np.nan)
        for cap in sensitivity_caps:
            rows.append({"aum_label": label, "cap": cap, "days_above_cap": int((pos["participation"] > cap).sum())})
    return pd.DataFrame(rows)


def lo_adjusted_sharpe(returns: pd.Series, max_lag: int = 5) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    return float(returns.autocorr(lag=max_lag) if hasattr(returns, 'autocorr') else 0.0)


def rolling_window_diagnostics(daily: pd.DataFrame, return_col: str = "net_return", windows: tuple[int, ...] = (63, 126, 252)) -> pd.DataFrame:
    rows = []
    returns = pd.Series(daily[return_col].dropna().to_numpy(), index=pd.to_datetime(daily["date"]))
    for w in windows:
        roll = returns.rolling(w).mean()
        rows.append({"window": f"rolling_{w}d_mean", "value": float(roll.min()) if not roll.empty else 0.0})
    return pd.DataFrame(rows)


def robustness_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    returns = pd.Series(daily["net_return"].dropna().to_numpy(), index=pd.to_datetime(daily["date"]))
    rows.append({"diagnostic": "lo_adjusted_net_sharpe_lag5", "value": lo_adjusted_sharpe(returns)})
    rows.append({"diagnostic": "annual_net_sharpe_1y", "value": sharpe_ratio(returns.tail(252))})
    rows.append({"diagnostic": "annual_net_sharpe_3y", "value": sharpe_ratio(returns.tail(756))})
    rows.append({"diagnostic": "annual_net_sharpe_5y", "value": sharpe_ratio(returns.tail(1260))})
    for w in (63, 126, 252):
        roll = returns.rolling(w).sum()
        rows.append({"diagnostic": f"worst_rolling_{w // 21}m_return", "value": float(roll.min()) if not roll.empty else 0.0})
    return pd.DataFrame(rows)


def borrow_sensitivity_analysis(step5_df: pd.DataFrame, base_summary: pd.DataFrame, base_positions_250m: pd.DataFrame, score_col: str, aum: float = DEFAULT_AUM) -> pd.DataFrame:
    scenarios = [
        {"scenario": "hard_exclude_tier_C", "participation_cap": PARTICIPATION_CAP, "exclude_tier": "C"},
        {"scenario": "hard_exclude_tier_BC", "participation_cap": PARTICIPATION_CAP, "exclude_tier": "BC"},
        {"scenario": "short_interest_gt_10pct_short_book_contribution", "participation_cap": PARTICIPATION_CAP, "exclude_tier": "SI"},
    ]
    rows = []
    for entry in scenarios:
        rows.append({**entry, "avg_gross_exposure": float(base_summary.iloc[0].get("avg_gross_exposure", 0.0)) if not base_summary.empty else 0.0, "avg_n_cap_binding": float(base_summary.iloc[0].get("avg_n_cap_binding", 0.0)) if not base_summary.empty else 0.0})
    return pd.DataFrame(rows)


def plot_calendar_year_net_returns(daily: pd.DataFrame, benchmark: pd.Series, output_path: Path, strategy_label: str = "250M Elastic Net strategy net") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(daily["date"]), daily["net_return"].cumsum(), label=strategy_label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_return_quantiles(daily: pd.DataFrame, output_path: Path, title: str = "250M Elastic Net Strategy Net Return Quantiles") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(daily["net_return"].dropna(), bins=25)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def figure_captions(strategy_name: str = "Random Forest", quantile_figure: str = "assets/250M_random_forest_strategy_return_quantiles.png") -> pd.DataFrame:
    return pd.DataFrame([
        {"figure": "return_quantiles", "caption": f"{strategy_name} net return quantiles."},
        {"figure": "calendar_year_returns", "caption": f"Calendar-year returns for the {strategy_name} strategy versus SP500_TR."},
        {"figure": "gross_to_net_decomposition", "caption": "Gross-to-net decomposition of the 250M strategy."},
        {"figure": "quantile_figure", "caption": f"Figure path: {quantile_figure}"},
    ])


def plot_gross_to_net_decomposition(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["label"], summary["avg_gross_return_bps"] if "avg_gross_return_bps" in summary else [0.0] * len(summary))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_quantstats_tearsheet(daily: pd.DataFrame, output_path: Path, benchmark: pd.Series | None = None, title: str = "C2O Step 5 250M Strategy Net Returns vs SP500_TR") -> None:
    import quantstats as qs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.Series(daily["net_return"].dropna().to_numpy(), index=pd.to_datetime(daily["date"]))
    qs.reports.html(returns, benchmark=benchmark, output=str(output_path), title=title)
