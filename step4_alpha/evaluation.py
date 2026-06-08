import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    BORROW_TIER_A_BPS,
    BORROW_TIER_B_BPS,
    BORROW_TIER_C_BPS,
    HTB_MODERATE_SI,
    ROUNDTRIP_BPS,
    TRADING_DAYS,
)


BORROW_BPS_BY_TIER = {
    "A": BORROW_TIER_A_BPS,
    "B": BORROW_TIER_B_BPS,
    "C": BORROW_TIER_C_BPS,
}


def daily_spearman_ic(
    df: pd.DataFrame,
    score_col: str,
    target_col: str = "target_winsorized_demeaned",
) -> pd.Series:
    """
    Compute daily cross-sectional Spearman IC.

    IC measures whether stocks with higher scores on date t have higher
    realized next overnight returns.
    """
    def valid_daily_ic(group: pd.DataFrame) -> float:
        pair = group[[score_col, target_col]].dropna()
        if (
            len(pair) < 3
            or pair[score_col].nunique() < 2
            or pair[target_col].nunique() < 2
        ):
            return np.nan
        return pair[score_col].corr(pair[target_col], method="spearman")

    return df.groupby("date", sort=False).apply(valid_daily_ic).dropna()


def ic_summary(
    df: pd.DataFrame,
    score_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    ) -> pd.DataFrame:
    """
    Summarise mean IC, IC volatility and t-stat for multiple score columns.
    """
    rows = []

    for col in score_cols:
        daily_ic = daily_spearman_ic(df, col, target_col)

        mean_ic = daily_ic.mean()
        std_ic = daily_ic.std()
        n_days = len(daily_ic)

        t_stat = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic > 0 else np.nan

        rows.append({
            "score_col": col,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "t_stat": t_stat,
            "n_days": n_days,
        })

    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False)


def year_by_year_ic_summary(
    df: pd.DataFrame,
    score_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
) -> pd.DataFrame:
    """Summarise daily Spearman IC separately for each calendar year."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    rows = []

    for year, year_df in data.groupby(data["date"].dt.year, sort=True):
        for score_col in score_cols:
            if score_col not in year_df.columns:
                continue
            summary = ic_summary(
                year_df,
                score_cols=[score_col],
                target_col=target_col,
            ).iloc[0]
            rows.append(
                {
                    "year": int(year),
                    "score_col": score_col,
                    "mean_ic": summary["mean_ic"],
                    "std_ic": summary["std_ic"],
                    "t_stat": summary["t_stat"],
                    "n_days": int(summary["n_days"]),
                    "n_observations": int(
                        year_df[[score_col, target_col]].dropna().shape[0]
                    ),
                }
            )

    return pd.DataFrame(rows)


def regime_ic_summary(
    df: pd.DataFrame,
    score_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    vol_col: str = "vol20",
    short_interest_col: str = "dsi",
    short_interest_threshold: float = HTB_MODERATE_SI,
    supplied_regime_col: str = "regime",
) -> pd.DataFrame:
    """
    Summarise IC across volatility, short-interest and stress regimes.

    High/low volatility is defined from the median stock-level vol20 on each
    date, split at the median daily value over the evaluation sample. Regime
    subsets may overlap by design.
    """
    required = {"date", target_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for regime IC: {sorted(missing)}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    regimes: list[tuple[str, str, str, pd.Series]] = []

    if vol_col in data.columns:
        daily_vol = data.groupby("date")[vol_col].median()
        vol_threshold = float(daily_vol.median())
        mapped_vol = data["date"].map(daily_vol)
        regimes.extend(
            [
                (
                    "volatility",
                    "high_volatility_days",
                    f"daily median {vol_col} > {vol_threshold:.8g}",
                    mapped_vol > vol_threshold,
                ),
                (
                    "volatility",
                    "low_volatility_days",
                    f"daily median {vol_col} <= {vol_threshold:.8g}",
                    mapped_vol <= vol_threshold,
                ),
            ]
        )

    if short_interest_col in data.columns:
        short_interest = pd.to_numeric(data[short_interest_col], errors="coerce")
        regimes.extend(
            [
                (
                    "short_interest",
                    "high_short_interest_stocks",
                    f"{short_interest_col} > {short_interest_threshold:.2%}",
                    short_interest > short_interest_threshold,
                ),
                (
                    "short_interest",
                    "normal_short_interest_stocks",
                    f"{short_interest_col} <= {short_interest_threshold:.2%}",
                    short_interest <= short_interest_threshold,
                ),
            ]
        )

    regimes.extend(
        [
            (
                "stress_period",
                "2020_Q1",
                "2020-01-01 through 2020-03-31",
                data["date"].between("2020-01-01", "2020-03-31"),
            ),
            (
                "stress_period",
                "2022_rate_hike_bear_market",
                "2022-01-01 through 2022-12-31",
                data["date"].between("2022-01-01", "2022-12-31"),
            ),
        ]
    )

    if supplied_regime_col in data.columns:
        for label in sorted(data[supplied_regime_col].dropna().unique()):
            regimes.append(
                (
                    "supplied_regime",
                    str(label),
                    f"{supplied_regime_col} == {label}",
                    data[supplied_regime_col].eq(label),
                )
            )

    rows = []
    for regime_type, regime, definition, mask in regimes:
        regime_df = data[mask.fillna(False)]
        if regime_df.empty:
            continue
        for score_col in score_cols:
            if score_col not in regime_df.columns:
                continue
            summary = ic_summary(
                regime_df,
                score_cols=[score_col],
                target_col=target_col,
            ).iloc[0]
            rows.append(
                {
                    "regime_type": regime_type,
                    "regime": regime,
                    "definition": definition,
                    "score_col": score_col,
                    "mean_ic": summary["mean_ic"],
                    "std_ic": summary["std_ic"],
                    "t_stat": summary["t_stat"],
                    "n_days": int(summary["n_days"]),
                    "n_observations": int(
                        regime_df[[score_col, target_col]].dropna().shape[0]
                    ),
                }
            )

    return pd.DataFrame(rows)


def decile_spread_summary(
    df: pd.DataFrame,
    score_col: str = "score_baseline",
    target_col: str = "target_winsorized_demeaned",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute average realised target return by score decile.

    Each day, stocks are sorted by score and split into deciles.
    Decile 1 contains the lowest-score stocks.
    Decile 10 contains the highest-score stocks.
    """
    df = df[["date", score_col, target_col]].dropna().copy()

    df["decile"] = (
        df.groupby("date")[score_col]
        .transform(lambda x: pd.qcut(x.rank(method="first"), n_bins, labels=False) + 1)
    )

    decile_table = (
        df.groupby("decile")[target_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    bottom = decile_table.loc[decile_table["decile"] == 1, "mean"].iloc[0]
    top = decile_table.loc[decile_table["decile"] == n_bins, "mean"].iloc[0]

    spread_row = pd.DataFrame({
        "decile": ["top_minus_bottom"],
        "mean": [top - bottom],
        "std": [pd.NA],
        "count": [pd.NA],
    })

    return pd.concat([decile_table, spread_row], ignore_index=True)


def daily_score_spread(
    df: pd.DataFrame,
    score_col: str,
    target_col: str = "target_raw",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return the daily equal-weight top-minus-bottom score-decile spread."""
    required = {"date", score_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for daily spread: {sorted(missing)}")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    rows = []
    data = df[["date", score_col, target_col]].dropna()
    for date, day in data.groupby("date", sort=True):
        if len(day) < n_bins or day[score_col].nunique() < 2:
            continue

        ranked = day.sort_values(score_col, kind="mergesort")
        n_side = max(1, len(ranked) // n_bins)
        bottom = ranked.head(n_side)[target_col]
        top = ranked.tail(n_side)[target_col]
        bottom_mean = float(bottom.mean())
        top_mean = float(top.mean())
        rows.append(
            {
                "date": date,
                "bottom_mean": bottom_mean,
                "top_mean": top_mean,
                "top_minus_bottom": top_mean - bottom_mean,
                "n_bottom": int(len(bottom)),
                "n_top": int(len(top)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "bottom_mean",
            "top_mean",
            "top_minus_bottom",
            "n_bottom",
            "n_top",
        ],
    )


def cost_aware_decile_summary(
    df: pd.DataFrame,
    score_col: str,
    target_col: str = "target_raw",
    tier_col: str = "htb_tier",
    n_bins: int = 10,
    roundtrip_bps: float = ROUNDTRIP_BPS,
) -> pd.DataFrame:
    """
    Summarise a fully invested dollar-neutral top/bottom decile portfolio.

    Each side receives 50% notional. Trading cost is charged at the fixed
    full-gross overnight round-trip rate, while borrow is charged only against
    the short half using the average tier rate of bottom-decile names.
    """
    required = {"date", score_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for cost-aware spread: {sorted(missing)}")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    columns = ["date", score_col, target_col]
    if tier_col in df.columns:
        columns.append(tier_col)
    data = df[columns].dropna(subset=["date", score_col, target_col]).copy()
    if tier_col not in data.columns:
        data[tier_col] = "A"
    data[tier_col] = data[tier_col].fillna("A")

    daily_rows = []
    for date, day in data.groupby("date", sort=True):
        if len(day) < n_bins or day[score_col].nunique() < 2:
            continue

        ranked = day.sort_values(score_col, kind="mergesort")
        n_side = max(1, len(ranked) // n_bins)
        bottom = ranked.head(n_side)
        top = ranked.tail(n_side)

        top_mean = float(top[target_col].mean())
        bottom_mean = float(bottom[target_col].mean())
        gross_return = 0.5 * (top_mean - bottom_mean)
        trading_cost = roundtrip_bps / 10_000
        short_borrow_bps = (
            bottom[tier_col]
            .map(BORROW_BPS_BY_TIER)
            .fillna(BORROW_TIER_A_BPS)
            .mean()
        )
        borrow_cost = 0.5 * short_borrow_bps / 10_000 / TRADING_DAYS

        daily_rows.append(
            {
                "gross_return": gross_return,
                "trading_cost_return": trading_cost,
                "borrow_cost_return": borrow_cost,
                "net_return": gross_return - trading_cost - borrow_cost,
            }
        )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return pd.DataFrame()

    gross_std = daily["gross_return"].std()
    net_std = daily["net_return"].std()
    gross_sharpe = (
        daily["gross_return"].mean() / gross_std * np.sqrt(TRADING_DAYS)
        if gross_std > 0
        else np.nan
    )
    net_sharpe = (
        daily["net_return"].mean() / net_std * np.sqrt(TRADING_DAYS)
        if net_std > 0
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "score_col": score_col,
                "target_col": target_col,
                "n_days": len(daily),
                "mean_gross_return": daily["gross_return"].mean(),
                "mean_trading_cost_return": daily[
                    "trading_cost_return"
                ].mean(),
                "mean_borrow_cost_return": daily["borrow_cost_return"].mean(),
                "mean_net_return": daily["net_return"].mean(),
                "gross_ann_return": daily["gross_return"].mean() * TRADING_DAYS,
                "net_ann_return": daily["net_return"].mean() * TRADING_DAYS,
                "gross_sharpe": gross_sharpe,
                "net_sharpe": net_sharpe,
            }
        ]
    )


def plot_decile_spread(
    decile_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Baseline Score Decile Spread",
    ) -> None:
    """
    Plot average realised target return by decile.

    The input decile_df should be the output of decile_spread_summary().
    The special row 'top_minus_bottom' is excluded from the bar chart.
    """
    plot_df = decile_df[decile_df["decile"] != "top_minus_bottom"].copy()
    plot_df["decile"] = plot_df["decile"].astype(int)

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["decile"], plot_df["mean"])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Decile")
    plt.ylabel("Average next-overnight return")
    plt.title(title)
    plt.xticks(plot_df["decile"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_ic_summary(
    ic_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Mean Daily Spearman IC",
    n_top_features: int = 10,
) -> None:
    """
    Plot model ICs and the strongest individual feature ICs.

    All model scores are retained. Individual features are ranked by absolute
    mean IC so both strong positive and strong negative predictors are shown.
    The complete set remains available in the CSV summary.
    """
    model_names = {
        "score_baseline",
        "score_elastic_net",
        "score_random_forest",
    }
    model_rows = ic_df[ic_df["score_col"].isin(model_names)]
    feature_rows = (
        ic_df[~ic_df["score_col"].isin(model_names)]
        .assign(abs_mean_ic=lambda x: x["mean_ic"].abs())
        .nlargest(n_top_features, "abs_mean_ic")
        .drop(columns="abs_mean_ic")
    )
    plot_df = pd.concat([model_rows, feature_rows], ignore_index=True)
    plot_df = plot_df.sort_values("mean_ic")
    plot_df["label"] = (
        plot_df["score_col"]
        .str.removeprefix("z_feat_")
        .str.removesuffix("_lag1")
        .str.replace("_", " ")
    )

    plt.figure(figsize=(10, 7))
    colors = [
        "tab:blue" if name in model_names else "tab:gray"
        for name in plot_df["score_col"]
    ]
    plt.barh(plot_df["label"], plot_df["mean_ic"], color=colors)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Mean IC")
    plt.ylabel("Model / selected feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_year_by_year_ic(
    yearly_ic: pd.DataFrame,
    output_path: str | Path,
    title: str = "Year-by-Year Mean Daily Spearman IC",
) -> None:
    """Plot annual mean IC lines for the baseline and two ML models."""
    required = {"year", "score_col", "mean_ic"}
    missing = required - set(yearly_ic.columns)
    if missing:
        raise KeyError(f"Missing columns for yearly IC plot: {sorted(missing)}")

    model_styles = {
        "score_baseline": ("Baseline", "tab:gray"),
        "score_elastic_net": ("Elastic Net", "tab:blue"),
        "score_random_forest": ("Random Forest", "tab:orange"),
    }
    plot_df = yearly_ic[yearly_ic["score_col"].isin(model_styles)].copy()
    if plot_df.empty:
        raise ValueError("No recognised model rows found for yearly IC plot.")

    plt.figure(figsize=(10, 6))
    for score_col, (label, color) in model_styles.items():
        model_df = plot_df[plot_df["score_col"] == score_col].sort_values("year")
        if model_df.empty:
            continue
        plt.plot(
            model_df["year"],
            model_df["mean_ic"],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )

    years = sorted(plot_df["year"].unique())
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(years)
    plt.xlabel("Year")
    plt.ylabel("Mean daily Spearman IC")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
