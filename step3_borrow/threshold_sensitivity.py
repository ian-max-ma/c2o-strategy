"""
Sensitivity check for borrow-tier thresholds.

Purpose:
    Our initial hard-to-borrow proxy uses percentile thresholds.
    This script tests several threshold configurations and reports
    the resulting Tier A/B/C distribution.

Output:
    step3_borrow/threshold_sensitivity.csv
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "step3_borrow"

TRAIN_END = pd.Timestamp("2024-12-31")


CONFIGS = [
    {
        "name": "base_90_975",
        "moderate_q": 0.90,
        "high_q": 0.975,
    },
    {
        "name": "stricter_90_99",
        "moderate_q": 0.90,
        "high_q": 0.99,
    },
    {
        "name": "stricter_95_99",
        "moderate_q": 0.95,
        "high_q": 0.99,
    },
    {
        "name": "looser_85_975",
        "moderate_q": 0.85,
        "high_q": 0.975,
    },
]


def assign_tiers(si, moderate_q, high_q):
    """Assign A/B/C tiers under a given threshold configuration."""
    train = si[si["date"] <= TRAIN_END].copy()

    thresholds = {
        "dsi_moderate": train["dsi"].quantile(moderate_q),
        "dsi_high": train["dsi"].quantile(high_q),
        "dtcn_moderate": train["dtcn"].quantile(moderate_q),
        "dtcn_high": train["dtcn"].quantile(high_q),
        "ddtcn_moderate": train["ddtcn"].quantile(moderate_q),
        "ddtcn_high": train["ddtcn"].quantile(high_q),
    }

    out = si.copy()

    moderate_h2b = (
        (out["dsi"] >= thresholds["dsi_moderate"])
        | (out["dtcn"] >= thresholds["dtcn_moderate"])
        | (out["ddtcn"] >= thresholds["ddtcn_moderate"])
    )

    high_h2b = (
        (out["dsi"] >= thresholds["dsi_high"])
        | (out["dtcn"] >= thresholds["dtcn_high"])
        | (out["ddtcn"] >= thresholds["ddtcn_high"])
    )

    out["borrow_tier"] = "A"
    out.loc[moderate_h2b, "borrow_tier"] = "B"
    out.loc[high_h2b, "borrow_tier"] = "C"

    return out, thresholds


def main():
    si = pd.read_parquet(DATA_DIR / "short_interest_transfo.parquet")
    si["date"] = pd.to_datetime(si["date"])

    rows = []

    for config in CONFIGS:
        result, thresholds = assign_tiers(
            si,
            moderate_q=config["moderate_q"],
            high_q=config["high_q"],
        )

        dist = result["borrow_tier"].value_counts(normalize=True)

        rows.append(
            {
                "config": config["name"],
                "moderate_q": config["moderate_q"],
                "high_q": config["high_q"],
                "tier_A_pct": dist.get("A", 0.0),
                "tier_B_pct": dist.get("B", 0.0),
                "tier_C_pct": dist.get("C", 0.0),
                "dsi_moderate": thresholds["dsi_moderate"],
                "dsi_high": thresholds["dsi_high"],
                "dtcn_moderate": thresholds["dtcn_moderate"],
                "dtcn_high": thresholds["dtcn_high"],
                "ddtcn_moderate": thresholds["ddtcn_moderate"],
                "ddtcn_high": thresholds["ddtcn_high"],
            }
        )

    sensitivity = pd.DataFrame(rows)

    output_path = OUT_DIR / "threshold_sensitivity.csv"
    sensitivity.to_csv(output_path, index=False)

    print("Threshold sensitivity:")
    print(sensitivity)
    print()
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()