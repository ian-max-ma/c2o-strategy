import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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
    return (
        df.groupby("date")
        .apply(lambda x: x[score_col].corr(x[target_col], method="spearman"))
        .dropna()
    )


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
) -> None:
    """
    Plot mean IC for each score / feature.
    """
    plot_df = ic_df.copy()

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["score_col"], plot_df["mean_ic"])
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Mean IC")
    plt.ylabel("Score / feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()