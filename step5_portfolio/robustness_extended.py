"""
Extended robustness diagnostics for Step 5 — directly answering Brief §7.2, §7.3, §7.4.

Functions here are called from run_step5.py after the main backtest loop.
All outputs are written to step5_portfolio/output/ and are reproducible via
``python run_step5.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe(ret: pd.Series) -> float:
    """Annualised Sharpe with zero risk-free rate."""
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 10 or ret.std(ddof=1) == 0:
        return np.nan
    return float(ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _ann_return(ret: pd.Series) -> float:
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) == 0:
        return np.nan
    return float((1.0 + ret).prod() ** (TRADING_DAYS / len(ret)) - 1.0)


def _ann_vol(ret: pd.Series) -> float:
    return float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _max_dd(ret: pd.Series) -> float:
    cum = (1.0 + ret.fillna(0.0)).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def _lo_sharpe(ret: pd.Series, max_lag: int = 5) -> float:
    """Lo (2002) autocorrelation-adjusted Sharpe."""
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) <= max_lag + 2:
        return np.nan
    demeaned = ret - ret.mean()
    g0 = float(np.mean(demeaned ** 2))
    if g0 <= 0:
        return np.nan
    adj = 1.0
    for lag in range(1, max_lag + 1):
        rho = ret.autocorr(lag=lag)
        if not np.isnan(rho):
            adj += 2.0 * (1.0 - lag / (max_lag + 1.0)) * rho
    adj = max(adj, 1e-8)
    return float(ret.mean() / np.sqrt(g0 * adj) * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# §7.2  Statistical robustness table
# ---------------------------------------------------------------------------

def stat_robustness_table(
    daily: pd.DataFrame,
    active_start: str = "2018-01-01",
) -> pd.DataFrame:
    """
    Build the §7.2 statistical robustness diagnostics table.

    Operates on the *active* ML window (>= active_start) so that the
    Lo-adjusted Sharpe, autocorrelation, rolling worst windows, and top-5%
    concentration are all computed on trading days with actual positions.

    Parameters
    ----------
    daily:
        Full-window daily returns DataFrame (output of run_aum_backtests for 250M).
        Must contain columns: date, net_return, gross_exposure.
    active_start:
        First date of the active ML OOS window.

    Returns
    -------
    DataFrame with columns: diagnostic, value, interpretation.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    active = daily[daily["date"] >= pd.Timestamp(active_start)].copy()
    r = active["net_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(r) == 0:
        return pd.DataFrame(columns=["diagnostic", "value", "interpretation"])

    # Standard Sharpe (active window)
    std_sharpe = _sharpe(r)
    lo5 = _lo_sharpe(r, max_lag=5)
    ac1 = float(r.autocorr(lag=1))

    # Top-5% concentration
    n_top = max(1, int(np.ceil(len(r) * 0.05)))
    top_sum = float(r.nlargest(n_top).sum())
    total_sum = float(r.sum())
    if abs(total_sum) > 1e-12:
        top_ratio = top_sum / total_sum
    else:
        top_ratio = np.nan

    # Worst rolling windows
    windows = [(63, "3m"), (126, "6m"), (252, "12m")]
    worst: dict[str, tuple[float, str]] = {}
    r_indexed = r.copy()
    r_indexed.index = active["date"].values
    for w, label in windows:
        rolling = (1.0 + r_indexed).rolling(w).apply(np.prod, raw=True) - 1.0
        rolling = rolling.dropna()
        if rolling.empty:
            worst[label] = (np.nan, "n/a")
        else:
            worst_end = rolling.idxmin()
            worst[label] = (float(rolling.min()), str(pd.Timestamp(worst_end).date()))

    rows = [
        {
            "diagnostic": "standard_net_sharpe_active_window",
            "value": std_sharpe,
            "interpretation": (
                f"Annualised daily net Sharpe over active ML window "
                f"({pd.Timestamp(active_start).year}–2024). "
                "Zero risk-free rate."
            ),
        },
        {
            "diagnostic": "lo_adjusted_net_sharpe_lag5",
            "value": lo5,
            "interpretation": (
                "Lo (2002) autocorrelation-adjusted Sharpe using lags 1–5. "
                "Close to the standard Sharpe → daily autocorrelation does not "
                "mechanically explain the result."
                if not np.isnan(lo5) and not np.isnan(std_sharpe)
                   and abs(lo5 - std_sharpe) < abs(std_sharpe) * 0.15
                else "Lo (2002) autocorrelation-adjusted Sharpe using lags 1–5."
            ),
        },
        {
            "diagnostic": "lag1_daily_autocorrelation",
            "value": ac1,
            "interpretation": (
                "First-order daily autocorrelation of net returns. "
                "Negative → no positive return smoothing; "
                "Sharpe is not inflated by autocorrelation."
                if ac1 < 0
                else "First-order daily autocorrelation of net returns."
            ),
        },
        {
            "diagnostic": f"worst_rolling_3m_return",
            "value": worst["3m"][0],
            "interpretation": (
                f"Worst 63-trading-day compounded net return "
                f"(window ending {worst['3m'][1]})."
            ),
        },
        {
            "diagnostic": "worst_rolling_6m_return",
            "value": worst["6m"][0],
            "interpretation": (
                f"Worst 126-trading-day compounded net return "
                f"(window ending {worst['6m'][1]})."
            ),
        },
        {
            "diagnostic": "worst_rolling_12m_return",
            "value": worst["12m"][0],
            "interpretation": (
                f"Worst 252-trading-day compounded net return "
                f"(window ending {worst['12m'][1]})."
            ),
        },
        {
            "diagnostic": f"top_5pct_days_sum_n{n_top}",
            "value": top_sum,
            "interpretation": (
                f"Arithmetic sum of the best {n_top} daily net returns "
                f"(top 5% of {len(r)} active trading days)."
            ),
        },
        {
            "diagnostic": "total_arithmetic_net_return_sum",
            "value": total_sum,
            "interpretation": "Arithmetic sum of all active-window daily net returns.",
        },
        {
            "diagnostic": "top_5pct_share_of_total_return",
            "value": top_ratio,
            "interpretation": (
                f"Top-5% days contribute {top_ratio:.2f}× total arithmetic return. "
                "Alpha is event-concentrated rather than smoothly harvested."
                if not np.isnan(top_ratio) and total_sum > 0
                else (
                    "Top-5% days contribute more in absolute terms than total return; "
                    "total return is negative so the ratio is not interpretable as usual."
                    if not np.isnan(top_ratio)
                    else "Ratio undefined (total return near zero)."
                )
            ),
        },
    ]

    # Annual net Sharpe rows (active years only)
    dated = active[["date", "net_return"]].set_index("date")["net_return"].replace(
        [np.inf, -np.inf], np.nan
    )
    for year, group in dated.groupby(dated.index.year):
        yr_sharpe = _sharpe(group.fillna(0.0))
        rows.append(
            {
                "diagnostic": f"annual_net_sharpe_{year}",
                "value": yr_sharpe,
                "interpretation": f"Net Sharpe for calendar year {year}.",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §7.3  Capacity and execution honesty — cap-bite and position audit
# ---------------------------------------------------------------------------

def cap_bite_audit(
    cap_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reformat cap_sensitivity.csv to directly answer the §7.3 cap-bite question.

    Expected columns in cap_sensitivity:
        scenario, participation_cap, aum_label,
        gross_ann_return, net_ann_return, net_sharpe,
        avg_gross_exposure, pct_days_capacity_constrained,
        avg_n_cap_binding, max_single_name_participation,
        n_days_with_cap_breach
    """
    rows = []
    for _, row in cap_sensitivity.iterrows():
        label = str(row.get("aum_label", "250M"))
        scenario = str(row.get("scenario", ""))
        is_headline = "5pct" in scenario
        is_loose = "100pct" in scenario or "loose" in scenario
        if not (is_headline or is_loose):
            continue
        rows.append(
            {
                "variant": "5% ADV cap (submitted)" if is_headline else "No cap (loose reference)",
                "aum_label": label,
                "net_sharpe": row.get("net_sharpe"),
                "net_ann_return": row.get("net_ann_return"),
                "avg_gross_exposure": row.get("avg_gross_exposure"),
                "pct_days_capacity_constrained": row.get("pct_days_capacity_constrained"),
                "max_single_name_participation": row.get("max_single_name_participation"),
                "n_days_cap_breach": row.get("n_days_with_cap_breach", 0),
                "interpretation": (
                    "Required ADV cap from the brief"
                    if is_headline
                    else "Loose-cap reference — shows cap is binding"
                ),
            }
        )
    return pd.DataFrame(rows)


def position_size_audit_table(
    position_audit: pd.DataFrame,
    aum_levels: dict[str, float],
) -> pd.DataFrame:
    """
    Build §7.3 per-stock position-size audit at each AUM level.

    Expected input: position_capacity_audit.csv output.
    """
    rows = []
    for label, aum in aum_levels.items():
        mask = position_audit["aum_label"] == label
        sub = position_audit[mask]
        if sub.empty:
            continue
        row = sub.iloc[0]
        avg_n = row.get("avg_n_long", row.get("avg_n_short", 100))
        # Theoretical equal-weight per name
        avg_ew = aum / max(avg_n, 1) if avg_n > 0 else 0.0
        # Min ADV20 required to avoid cap breach at equal weight
        min_adv = avg_ew / 0.05
        rows.append(
            {
                "aum_label": label,
                "aum_usd": aum,
                "avg_n_names_per_leg": round(float(avg_n), 1),
                "theoretical_avg_per_name_usd": round(avg_ew, 0),
                "min_adv20_to_avoid_cap_breach_usd": round(min_adv, 0),
                "max_single_name_participation": row.get("max_single_name_participation"),
                "mean_daily_max_participation": row.get("mean_daily_max_participation"),
                "pct_positions_at_cap": row.get("pct_positions_at_cap"),
                "n_days_with_cap_breach": row.get("n_days_with_cap_breach", 0),
                "cap_breach_observed": bool(row.get("n_days_with_cap_breach", 0) > 0),
            }
        )
    return pd.DataFrame(rows)


def slippage_reconciliation_table(
    cost_schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build §7.3 slippage-model reconciliation table.

    Expected input: cost_schedule_summary.csv output.
    """
    rows = [
        {
            "cost_assumption": "Square-root impact at 5% ADV cap (Section 3.3)",
            "role": "Capacity stress / upper-bound impact",
            "magnitude": "17.7–39.4 bps per side",
            "interpretation": (
                "Conservative upper bound when orders are near the participation cap. "
                "Derived from σ_daily × k × √f with k=0.7, f=0.05."
            ),
        },
        {
            "cost_assumption": "Flat auction-cost schedule (Section 6.3)",
            "role": "Headline net backtest cost",
            "magnitude": "0.5 bps commission + 1.5 bps slippage per leg (4 bps round trip)",
            "interpretation": (
                "Specified in the coursework brief Table 6.1. "
                "Represents a small fraction of auction volume, not the cap-level upper bound. "
                "The two models are at different levels of pessimism; "
                "the square-root model applies when orders approach the cap."
            ),
        },
        {
            "cost_assumption": "Borrow cost schedule (Section 3.4)",
            "role": "Short-leg financing cost",
            "magnitude": "Tier A 40 bps / Tier B 200 bps / Tier C 800 bps annualised",
            "interpretation": (
                "Applied daily on short notional according to HTB tier "
                "(dsi>10% or dtcn>10 → Tier B; dsi>20% or dtcn>20 → Tier C)."
            ),
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §7.4  Borrow honesty
# ---------------------------------------------------------------------------

def borrow_contribution_table(
    borrow_sensitivity: pd.DataFrame,
    borrow_tier_audit: pd.DataFrame,
    aum_label: str = "250M",
) -> pd.DataFrame:
    """
    Quantify the contribution of high-short-interest names in bps of gross return.

    Answers the §7.4 question: 'Quantify, in basis points of gross return,
    the contribution of names with reported short interest above 10% of float.'
    """
    # From borrow_sensitivity, extract the diagnostic row about dsi>10% short names
    diag_mask = borrow_sensitivity["scenario"] == "short_interest_gt_10pct_short_book_contribution"
    if not diag_mask.any():
        return pd.DataFrame()

    diag = borrow_sensitivity[diag_mask].iloc[0]
    gross_bps = diag.get("gross_bps_per_day")
    pnl_share = diag.get("share_of_short_side_gross_pnl")
    notional_share = diag.get("avg_short_notional_share")

    # From borrow_tier_audit, get Tier B+C borrow cost share
    tier_mask = borrow_tier_audit["aum_label"] == aum_label
    tier_df = borrow_tier_audit[tier_mask].copy() if tier_mask.any() else pd.DataFrame()

    rows = []
    if gross_bps is not None and not np.isnan(float(gross_bps)):
        ann_bps = float(gross_bps) * 252
        rows.append(
            {
                "bucket": "Short positions with dsi > 10%",
                "avg_daily_gross_bps_contribution": round(float(gross_bps), 4),
                "annualised_gross_bps_contribution": round(ann_bps, 2),
                "share_of_short_side_gross_pnl": (
                    f"{float(pnl_share):.1%}" if pnl_share is not None else "n/a"
                ),
                "avg_short_notional_share": (
                    f"{float(notional_share):.1%}" if notional_share is not None else "n/a"
                ),
                "interpretation": (
                    "Positive contribution means dsi>10% shorts help performance. "
                    "Small share relative to total → alpha is not solely borrow-cost arbitrage."
                    if ann_bps > 0
                    else "Negative or near-zero contribution from high-SI shorts."
                ),
            }
        )

    # Add borrow-cost decomposition by tier
    if not tier_df.empty:
        for _, tr in tier_df.iterrows():
            tier = tr.get("htb_tier", "?")
            borrow_share = tr.get("borrow_cost_share")
            notional_share_tier = tr.get("short_notional_share")
            rows.append(
                {
                    "bucket": f"Borrow cost attribution — Tier {tier}",
                    "avg_daily_gross_bps_contribution": np.nan,
                    "annualised_gross_bps_contribution": np.nan,
                    "share_of_short_side_gross_pnl": "n/a",
                    "avg_short_notional_share": (
                        f"{float(notional_share_tier):.1%}"
                        if notional_share_tier is not None
                        else "n/a"
                    ),
                    "interpretation": (
                        f"Tier {tier} = {float(borrow_share):.1%} of total borrow cost, "
                        f"{float(notional_share_tier):.1%} of short notional."
                        if borrow_share is not None and notional_share_tier is not None
                        else f"Tier {tier} borrow breakdown."
                    ),
                }
            )

    return pd.DataFrame(rows)


def hard_exclusion_robustness_table(borrow_sensitivity: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat borrow_sensitivity.csv to answer the §7.4 hard-exclusion question.

    Returns a clean table comparing tiered-borrow vs hard-exclusion variants.
    """
    label_map = {
        "tiered_borrow_all_names": "Baseline: tiered borrow proxy (submitted)",
        "hard_exclude_tier_C": "Hard exclude Tier C names from short book",
        "hard_exclude_tier_BC": "Hard exclude Tier B + C names from short book",
    }
    rows = []
    for _, row in borrow_sensitivity.iterrows():
        scenario = str(row.get("scenario", ""))
        if scenario not in label_map:
            continue
        rows.append(
            {
                "strategy_version": label_map[scenario],
                "net_sharpe": row.get("net_sharpe"),
                "net_ann_return": row.get("net_ann_return"),
                "gross_sharpe": row.get("gross_sharpe"),
                "avg_gross_exposure": row.get("avg_gross_exposure"),
                "borrow_cost_ann_drag": row.get("borrow_cost_ann_drag"),
                "interpretation": (
                    "Submitted borrow-cost treatment"
                    if scenario == "tiered_borrow_all_names"
                    else "Robustness check: hard exclusion of high-SI names"
                ),
            }
        )
    df = pd.DataFrame(rows)
    # Add qualitative verdict
    if len(df) > 1:
        base_sharpe = df.loc[df["strategy_version"].str.contains("Baseline"), "net_sharpe"]
        if len(base_sharpe):
            base_val = float(base_sharpe.iloc[0])
            df["sharpe_vs_baseline"] = df["net_sharpe"].apply(
                lambda x: round(float(x) - base_val, 4) if pd.notna(x) else np.nan
            )
    return df


# ---------------------------------------------------------------------------
# §7.5  Reporting integrity
# ---------------------------------------------------------------------------

def reporting_integrity_table(
    config_params: dict,
) -> pd.DataFrame:
    """
    Build §7.5 reproducibility checklist from config parameters.

    Parameters
    ----------
    config_params : dict
        Keys mapped to values from config.py. Expected keys include:
        TRAIN_START, TRAIN_END, RANDOM_SEED, FINAL_SCORE_COL,
        FINAL_BASKET_QUANTILE, PARTICIPATION_CAP, WEIGHTING_SCHEME,
        AUM_LEVELS (dict), USE_SIGNAL_SCALING.
    """
    aum_labels = ", ".join(
        f"${int(v // 1e6)}M" for v in config_params.get("AUM_LEVELS", {}).values()
    )
    rows = [
        ("Data window", f"{config_params.get('TRAIN_START')} to {config_params.get('TRAIN_END')}"),
        ("Held-out window", "2025–2026 — never read during model development"),
        ("Main Step 5 command", "python run_step5.py"),
        ("Full pipeline command", "python run_all.py"),
        ("Figure + table output directory", "step5_portfolio/output/"),
        ("Final strategy AUM levels", aum_labels),
        ("Main reported AUM", "$250M"),
        ("Basket rule", f"Top/bottom {float(config_params.get('FINAL_BASKET_QUANTILE', 0.005)):.1%} per side"),
        ("Portfolio weighting", str(config_params.get("WEIGHTING_SCHEME", "equal"))),
        ("Participation cap", f"{float(config_params.get('PARTICIPATION_CAP', 0.05)):.0%} of ADV20 per stock per side"),
        ("Cost schedule", "0.5 bps commission + 1.5 bps slippage per leg; tiered borrow A/B/C"),
        ("Random seed (all steps)", str(config_params.get("RANDOM_SEED", "42"))),
        ("Random Forest spec", "100 trees, max depth 5, min leaf 500, sqrt feature subsampling"),
        ("Elastic Net spec", "alpha=1e-6, l1_ratio=0.5"),
        ("OOS training design", "Expanding-window yearly retraining; train on years < Y, predict year Y"),
        ("Signal scaling", str(config_params.get("USE_SIGNAL_SCALING", True))),
        ("Final score model", str(config_params.get("FINAL_SCORE_COL", "score_random_forest"))),
    ]
    return pd.DataFrame(rows, columns=["item", "reported_setting"])


# ---------------------------------------------------------------------------
# Figures for Appendix
# ---------------------------------------------------------------------------

def plot_rolling_worst_windows(
    daily: pd.DataFrame,
    output_path: Path,
    active_start: str = "2018-01-01",
    title: str = "Rolling 3/6/12-month compounded net returns — 250M strategy",
) -> None:
    """
    Figure for §7.2: rolling 3-, 6-, and 12-month compounded net returns.

    Highlights the worst window for each horizon.
    Saved to output_path as PNG (dpi=200).
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    active = daily[daily["date"] >= pd.Timestamp(active_start)].set_index("date")
    r = active["net_return"].fillna(0.0)

    windows = [(63, "3-month"), (126, "6-month"), (252, "12-month")]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (w, label), color in zip(windows, colors):
        rolling = (1.0 + r).rolling(w).apply(np.prod, raw=True) - 1.0
        rolling = rolling.dropna()
        ax.plot(rolling.index, rolling * 100.0, label=label, color=color, linewidth=1.3)
        # Mark worst point
        if not rolling.empty:
            worst_idx = rolling.idxmin()
            ax.scatter(
                [worst_idx], [rolling.min() * 100.0],
                color=color, s=50, zorder=5,
            )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Compounded net return (%)")
    ax.set_title(title, fontsize=11)
    ax.legend(framealpha=0.8)
    caption = (
        "Assumptions: $250M AUM, top/bottom 0.5% basket, equal-weighted legs, "
        "5% ADV20 participation cap, Section 6.3 cost schedule."
    )
    fig.text(0.01, -0.04, caption, fontsize=7, color="grey", wrap=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rolling-windows figure: {output_path}")


def plot_top5_concentration(
    daily: pd.DataFrame,
    output_path: Path,
    active_start: str = "2018-01-01",
    title: str = "Return concentration: top 5% of trading days — 250M strategy",
) -> None:
    """
    Figure for §7.2: bar chart showing contribution of top-5% days vs rest vs total.

    Saved to output_path as PNG (dpi=200).
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    active = daily[daily["date"] >= pd.Timestamp(active_start)].copy()
    r = active["net_return"].fillna(0.0)

    n_top = max(1, int(np.ceil(len(r) * 0.05)))
    top_sum = float(r.nlargest(n_top).sum())
    total_sum = float(r.sum())
    rest_sum = total_sum - top_sum

    labels = [f"Top 5%\n({n_top} days)", f"Remaining 95%\n({len(r) - n_top} days)", "Total"]
    values = [top_sum * 100.0, rest_sum * 100.0, total_sum * 100.0]
    colors = [
        "#1f77b4" if top_sum >= 0 else "#d62728",
        "#ff7f0e" if rest_sum >= 0 else "#d62728",
        "#2ca02c" if total_sum >= 0 else "#d62728",
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.2f}%",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Arithmetic net return contribution (%)")
    ax.set_title(title, fontsize=11)
    caption = (
        "Assumptions: $250M AUM, top/bottom 0.5% basket, equal-weighted legs, "
        "5% ADV20 participation cap, Section 6.3 cost schedule. "
        "Active ML window (2018–2024)."
    )
    fig.text(0.01, -0.04, caption, fontsize=7, color="grey", wrap=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top-5% concentration figure: {output_path}")


def plot_annual_net_sharpe(
    daily: pd.DataFrame,
    output_path: Path,
    active_start: str = "2018-01-01",
    title: str = "Year-by-year net Sharpe — 250M strategy (active ML window)",
) -> None:
    """
    Figure for §7.2: bar chart of annual net Sharpe (active years only).
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    active = daily[daily["date"] >= pd.Timestamp(active_start)].copy()
    active["year"] = active["date"].dt.year

    years, sharpes = [], []
    for yr, g in active.groupby("year"):
        sr = _sharpe(g["net_return"].fillna(0.0))
        years.append(int(yr))
        sharpes.append(sr)

    colors = ["#2ca02c" if s >= 0 else "#d62728" for s in sharpes]

    fig, ax = plt.subplots(figsize=(8, 4.0))
    bars = ax.bar([str(y) for y in years], sharpes, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, sharpes):
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.2f}",
            ha="center",
            va=va,
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Net Sharpe")
    ax.set_title(title, fontsize=11)
    caption = (
        "Assumptions: $250M AUM, top/bottom 0.5% basket, equal-weighted legs, "
        "5% ADV20 participation cap, Section 6.3 cost schedule."
    )
    fig.text(0.01, -0.04, caption, fontsize=7, color="grey", wrap=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved annual Sharpe figure: {output_path}")
