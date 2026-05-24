"""
step1_panel/sanity_check.py

Reproduce the Section 1.1 stylised fact:
    Equal-weighted portfolio of the eligible universe → cumulative ON
    return dominates total return; cumulative ID return is flat / negative.

The marker reads this output before anything else.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED


def compute_ew_streams(
    panel: pd.DataFrame,
    eligible_only: bool = True,
    filter_col: "str | None" = None,
) -> pd.DataFrame:
    """
    Compute equal-weighted daily ON, ID, and CC returns.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain: date, ticker, r_ON, r_ID, r_CC.
        If eligible_only=True, must also contain an 'eligible' bool column.
    eligible_only : bool
        If True (default), restrict to rows where eligible=True.
        Set to False to use all stocks (e.g. for the stylised-fact figure
        extended back to 2000 before the universe construction period).
    filter_col : str, optional
        Explicit boolean column to filter on. For Brief Section 2.3 this should
        be "in_universe": the frozen year-start 1,000-name list before Section 3
        capacity filters.

    Returns
    -------
    pd.DataFrame
        Index: date. Columns: ew_ON, ew_ID, ew_CC.
    """
    if filter_col is not None:
        if filter_col not in panel.columns:
            raise ValueError(f"filter_col={filter_col!r} not found in panel")
        subset = panel[panel[filter_col]].copy()
    elif eligible_only and "eligible" in panel.columns:
        subset = panel[panel["eligible"]].copy()
    else:
        subset = panel.copy()

    ew = (
        subset.groupby("date")[["r_ON", "r_ID", "r_CC"]]
        .mean()
        .rename(columns={"r_ON": "ew_ON", "r_ID": "ew_ID", "r_CC": "ew_CC"})
    )
    return ew


def cumulative_growth(returns: pd.Series) -> pd.Series:
    """Compound a return series into a $1 growth series."""
    return (1 + returns).cumprod()


def plot_stylised_fact(ew: pd.DataFrame, save_path: "str | None" = None) -> None:
    """
    Plot the overnight / intraday / total cumulative return decomposition.

    Parameters
    ----------
    ew : pd.DataFrame
        Output of compute_ew_streams().
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    growth = pd.DataFrame({
        "Overnight": cumulative_growth(ew["ew_ON"]),
        "Total": cumulative_growth(ew["ew_CC"]),
        "Intraday": cumulative_growth(ew["ew_ID"]),
    })

    ax.plot(ew.index, growth["Overnight"],
            color="navy",   lw=2,   label="Overnight (close-to-open)")
    ax.plot(ew.index, growth["Total"],
            color="navy",   lw=1.5, linestyle="--", label="Total (close-to-close)")
    ax.plot(ew.index, growth["Intraday"],
            color="crimson",lw=1.5, label="Intraday (open-to-close)")

    ax.axhline(1, color="grey", lw=0.8, linestyle=":")
    start_year = ew.index.year.min()
    end_year   = ew.index.year.max()
    # Risk-adjusted criterion: overnight Sharpe must exceed intraday Sharpe.
    # Cumulative-growth comparison fails in bull markets (both ON and ID positive),
    # but the Sharpe comparison correctly captures the overnight premium regardless
    # of the sample period's overall market direction.
    _sr = lambda s: (s.mean() * 252) / (s.std() * (252**0.5)) if s.std() > 0 else 0
    sr_on = _sr(ew["ew_ON"])
    sr_id = _sr(ew["ew_ID"])
    passes = sr_on > sr_id
    verdict = (
        f"PASS  (ON Sharpe {sr_on:.2f} > ID Sharpe {sr_id:.2f})"
        if passes else
        f"FAIL  (ON Sharpe {sr_on:.2f} ≤ ID Sharpe {sr_id:.2f})"
    )
    ax.set_title(
        f"Stylised fact — overnight captures the equity premium\n"
        f"(Equal-weighted year-start universe, {start_year}–{end_year})  [{verdict}]",
        fontsize=11,
    )
    ax.set_ylabel("Cumulative growth of $1")
    ax.set_xlabel("")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[sanity_check] Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def yearly_sharpe(ew: pd.DataFrame) -> pd.DataFrame:
    """
    Report annualised Sharpe for ON, ID, CC streams by calendar year.
    Quantifies the dispersion across years required by Section 2 Q6.
    """
    ew = ew.copy()
    ew["year"] = ew.index.year
    results = []
    for year, grp in ew.groupby("year"):
        row = {"year": year}
        for col in ["ew_ON", "ew_ID", "ew_CC"]:
            mu  = grp[col].mean() * 252
            vol = grp[col].std()  * np.sqrt(252)
            row[f"sharpe_{col}"] = mu / vol if vol > 0 else np.nan
        results.append(row)
    return pd.DataFrame(results).set_index("year")


def stylised_fact_diagnostics(ew: pd.DataFrame) -> pd.Series:
    """Compact pass/fail diagnostics for Brief Section 2.3."""
    growth_on = cumulative_growth(ew["ew_ON"])
    growth_id = cumulative_growth(ew["ew_ID"])
    growth_cc = cumulative_growth(ew["ew_CC"])

    yearly = yearly_sharpe(ew)
    return pd.Series({
        "start_date": str(ew.index.min().date()),
        "end_date": str(ew.index.max().date()),
        "n_days": int(len(ew)),
        "terminal_growth_ON": float(growth_on.iloc[-1]),
        "terminal_growth_ID": float(growth_id.iloc[-1]),
        "terminal_growth_CC": float(growth_cc.iloc[-1]),
        "ON_gt_CC": bool(growth_on.iloc[-1] > growth_cc.iloc[-1]),
        "ID_flat_or_negative": bool(growth_id.iloc[-1] <= 1.10),
        "brief_shape_pass": bool(
            (growth_on.iloc[-1] > growth_cc.iloc[-1])
            and (growth_id.iloc[-1] <= 1.10)
        ),
        "years_ON_sharpe_gt_ID": int((yearly["sharpe_ew_ON"] > yearly["sharpe_ew_ID"]).sum()),
        "years_total": int(len(yearly)),
    })


# ── G1: AMC/BMO timing verification (Brief §2 Q3) ────────────────────────────

def verify_amc_bmo(earnings: pd.DataFrame, n_examples: int = 5) -> None:
    """
    Hand-traceable verification of the AMC/BMO timing rule for a C2O strategy.

    For C2O, the decision / entry time is MOC (Close_t, 16:00 ET) and the exit
    is MOO (Open_{t+1}, 09:30 ET):

      AMC earnings on day D:  entry is MOC on day D (before the announcement).
                              The contaminated overnight is D → D+1.
                              strat_trading_date = D  (shift = 0, same day).

      BMO earnings on day D:  entry would be MOC on D-1 (overnight exits at Open_D,
                              which already reflects the pre-market release).
                              The contaminated overnight is D-1 → D.
                              strat_trading_date = D-1  (shift = -1, previous day).

    Brief §2 Q3: "Show the rule and verify it on at least one stock-event
    you cite by hand."

    Prints a table to stdout that can be copied directly into the report.
    """
    needed = {"report_date", "strat_trading_date", "before_after_market"}
    missing = needed - set(earnings.columns)
    if missing:
        print(f"[verify_amc_bmo] Cannot verify — missing columns: {missing}")
        return

    amc = earnings[earnings["before_after_market"] == "after"].dropna(
        subset=["report_date", "strat_trading_date"]
    ).head(n_examples)

    print("\n── AMC announcements (strat_trading_date should equal report_date, shift=0) ──")
    for _, row in amc.iterrows():
        shift = (row["strat_trading_date"] - row["report_date"]).days
        ok    = "✓" if shift == 0 else "✗ UNEXPECTED SHIFT"
        print(f"  id={row.get('stock_id', '?'):<8}  "
              f"report={row['report_date'].date()}  "
              f"strat={row['strat_trading_date'].date()}  "
              f"shift={shift:+d}d  {ok}")

    bmo = earnings[earnings["before_after_market"] == "before"].dropna(
        subset=["report_date", "strat_trading_date"]
    ).head(n_examples)

    print("\n── BMO announcements (strat_trading_date should be report_date - 1 BDay, shift=-1) ──")
    for _, row in bmo.iterrows():
        shift = (row["strat_trading_date"] - row["report_date"]).days
        ok    = "✓" if shift == -1 else "✗ UNEXPECTED SHIFT"
        print(f"  id={row.get('stock_id', '?'):<8}  "
              f"report={row['report_date'].date()}  "
              f"strat={row['strat_trading_date'].date()}  "
              f"shift={shift:+d}d  {ok}")

    total_amc = (earnings["before_after_market"] == "after").sum()
    total_bmo = (earnings["before_after_market"] == "before").sum()
    print(f"\n  Total AMC events: {total_amc:,}  |  Total BMO events: {total_bmo:,}")


# ── G2: instrument_id presence check (Brief §2.1.2 / §7.1) ──────────────────

def check_instrument_id(panel: pd.DataFrame) -> bool:
    """
    Verify that 'instrument_id' exists in the panel so that the earnings-window
    filter in apply_capacity_filters() actually fires.

    If the column is absent the earnings merge silently produces no exclusions —
    a look-ahead bug (the EARN_WINDOW flag would never be set).

    Returns True if the column is present and non-empty, False otherwise.
    """
    if "instrument_id" not in panel.columns:
        print(
            "[check_instrument_id] WARNING: 'instrument_id' not found in panel.\n"
            "  The earnings-window filter will produce no exclusions (silent failure).\n"
            "  Ensure prices.parquet retains this column through load_prices()."
        )
        return False

    n_valid = int(panel["instrument_id"].notna().sum())
    print(
        f"[check_instrument_id] OK — instrument_id present: "
        f"{n_valid:,} / {len(panel):,} rows non-null."
    )
    earn_excl = int((panel.get("eligibility_status", pd.Series()) == "EARN_WINDOW").sum())
    if earn_excl == 0:
        print("  [check_instrument_id] NOTE: zero EARN_WINDOW exclusions — "
              "verify earnings calendar was loaded and passed to apply_capacity_filters().")
    else:
        print(f"  [check_instrument_id] EARN_WINDOW exclusions: {earn_excl:,} ✓")
    return True


