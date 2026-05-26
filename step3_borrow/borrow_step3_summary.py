from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252

BORROW_ANNUAL_BPS = {
    "A": 40,
    "B": 200,
    "C": 800,
}


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "outputs" / "panel_step3.parquet"
OUT_DIR = ROOT / "step3_borrow" / "summary_output"


def annual_bps_to_daily_rate(annual_bps: float) -> float:
    return annual_bps / 10000.0 / TRADING_DAYS


def safe_sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    std = x.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return x.mean() / std * np.sqrt(TRADING_DAYS)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading panel from: {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)

    required_cols = {"date", "ticker", "instrument_id", "eligible", "htb_tier", "htb_flag"}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year

    panel["borrow_annual_bps"] = panel["htb_tier"].map(BORROW_ANNUAL_BPS)
    panel["borrow_daily_rate"] = panel["borrow_annual_bps"].apply(annual_bps_to_daily_rate)
    panel["borrow_daily_bps"] = panel["borrow_daily_rate"] * 10000

    print("\nPanel shape:")
    print(panel.shape)

    print("\nDate range:")
    print(panel["date"].min(), "to", panel["date"].max())

    # 1. Overall HTB tier distribution
    tier_all = (
        panel.groupby("htb_tier")
        .size()
        .rename("n_stock_days")
        .reset_index()
    )
    tier_all["share"] = tier_all["n_stock_days"] / tier_all["n_stock_days"].sum()
    tier_all["borrow_annual_bps"] = tier_all["htb_tier"].map(BORROW_ANNUAL_BPS)
    tier_all["borrow_daily_bps"] = tier_all["borrow_annual_bps"] / TRADING_DAYS
    tier_all.to_csv(OUT_DIR / "tier_distribution_all.csv", index=False)

    print("\nHTB tier distribution, all panel rows:")
    print(tier_all)

    # 2. Eligible universe HTB distribution
    eligible_panel = panel[panel["eligible"] == True].copy()

    tier_eligible = (
        eligible_panel.groupby("htb_tier")
        .size()
        .rename("n_stock_days")
        .reset_index()
    )
    tier_eligible["share"] = tier_eligible["n_stock_days"] / tier_eligible["n_stock_days"].sum()
    tier_eligible["borrow_annual_bps"] = tier_eligible["htb_tier"].map(BORROW_ANNUAL_BPS)
    tier_eligible["borrow_daily_bps"] = tier_eligible["borrow_annual_bps"] / TRADING_DAYS
    tier_eligible.to_csv(OUT_DIR / "tier_distribution_eligible.csv", index=False)

    print("\nHTB tier distribution, eligible rows only:")
    print(tier_eligible)

    # 3. Yearly HTB share
    yearly = (
        eligible_panel.groupby("year")
        .agg(
            n_eligible_stock_days=("eligible", "size"),
            n_htb_stock_days=("htb_flag", "sum"),
            avg_dsi=("dsi", "mean") if "dsi" in eligible_panel.columns else ("eligible", "size"),
            avg_dtcn=("dtcn", "mean") if "dtcn" in eligible_panel.columns else ("eligible", "size"),
        )
        .reset_index()
    )
    yearly["htb_share"] = yearly["n_htb_stock_days"] / yearly["n_eligible_stock_days"]
    yearly.to_csv(OUT_DIR / "yearly_htb_share.csv", index=False)

    print("\nYearly HTB share among eligible rows:")
    print(yearly.head())
    print("...")
    print(yearly.tail())

    # 4. Borrow cost schedule
    cost_schedule = pd.DataFrame(
        [
            {
                "htb_tier": tier,
                "borrow_annual_bps": annual_bps,
                "borrow_daily_rate": annual_bps_to_daily_rate(annual_bps),
                "borrow_daily_bps": annual_bps / TRADING_DAYS,
            }
            for tier, annual_bps in BORROW_ANNUAL_BPS.items()
        ]
    )
    cost_schedule.to_csv(OUT_DIR / "borrow_cost_schedule.csv", index=False)

    print("\nBorrow cost schedule:")
    print(cost_schedule)

    # 5. Signal affected fraction.
    # If a real short-signal column exists later, use it.
    # Otherwise report eligible-universe HTB share as the Step 3 proxy.
    possible_signal_cols = [
        "raw_short_signal",
        "short_signal",
        "is_short",
        "short_leg",
    ]
    signal_col = next((c for c in possible_signal_cols if c in panel.columns), None)

    if signal_col is not None:
        short_rows = panel[panel[signal_col] == True].copy()
        affected_fraction = short_rows["htb_flag"].mean()
        signal_summary = pd.DataFrame(
            [
                {
                    "basis": signal_col,
                    "n_short_signal_rows": len(short_rows),
                    "n_htb_affected_rows": int(short_rows["htb_flag"].sum()),
                    "affected_fraction": affected_fraction,
                }
            ]
        )
    else:
        affected_fraction = eligible_panel["htb_flag"].mean()
        signal_summary = pd.DataFrame(
            [
                {
                    "basis": "eligible_universe_proxy",
                    "n_short_signal_rows": len(eligible_panel),
                    "n_htb_affected_rows": int(eligible_panel["htb_flag"].sum()),
                    "affected_fraction": affected_fraction,
                }
            ]
        )

    signal_summary.to_csv(OUT_DIR / "short_signal_affected_fraction.csv", index=False)

    print("\nShort signal affected fraction:")
    print(signal_summary)

    # 6. Gross vs net borrow proxy.
    # This is a simple short overnight proxy using -r_ON.
    # It is not the final strategy backtest unless Step 4/5 adds actual signal weights.
    if "r_ON" in eligible_panel.columns:
        proxy = eligible_panel.dropna(subset=["r_ON", "borrow_daily_rate"]).copy()
        proxy["gross_short_on_return"] = -proxy["r_ON"]
        proxy["net_short_on_return"] = proxy["gross_short_on_return"] - proxy["borrow_daily_rate"]

        gross_daily = proxy.groupby("date")["gross_short_on_return"].mean()
        net_daily = proxy.groupby("date")["net_short_on_return"].mean()

        perf_summary = pd.DataFrame(
            [
                {
                    "portfolio": "equal_weight_short_ON_proxy",
                    "gross_ann_return": gross_daily.mean() * TRADING_DAYS,
                    "net_ann_return": net_daily.mean() * TRADING_DAYS,
                    "gross_ann_vol": gross_daily.std() * np.sqrt(TRADING_DAYS),
                    "net_ann_vol": net_daily.std() * np.sqrt(TRADING_DAYS),
                    "gross_sharpe": safe_sharpe(gross_daily),
                    "net_sharpe": safe_sharpe(net_daily),
                    "borrow_drag_ann": (gross_daily.mean() - net_daily.mean()) * TRADING_DAYS,
                }
            ]
        )
        perf_summary.to_csv(OUT_DIR / "gross_vs_net_borrow_proxy.csv", index=False)

        print("\nGross vs net borrow proxy:")
        print(perf_summary)
    else:
        print("\nColumn r_ON not found, skipping gross vs net borrow proxy.")

    print(f"\nDone. Summary files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()