from pathlib import Path

import pandas as pd

from step4_alpha.evaluation import decile_spread_summary, ic_summary
from step4_alpha.model import expanding_window_elastic_net


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"
OUTPUT_DIR = PROJECT_ROOT / "step4_alpha" / "eval_output"
OUTPUT_PATH = OUTPUT_DIR / "elastic_net_tuning.csv"

VALIDATION_FIRST_YEAR = 2021
VALIDATION_LAST_YEAR = 2023
MAX_TRAIN_ROWS = 500_000

PARAM_GRID = [
    {"alpha": 1e-6, "l1_ratio": 0.5},
    {"alpha": 3e-6, "l1_ratio": 0.5},
    {"alpha": 1e-5, "l1_ratio": 0.1},
    {"alpha": 1e-5, "l1_ratio": 0.5},
    {"alpha": 1e-5, "l1_ratio": 0.9},
    {"alpha": 3e-5, "l1_ratio": 0.5},
]


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs/alpha_scores.parquet. "
            "Please run `python3 run_step4.py` first."
        )

    df = pd.read_parquet(INPUT_PATH)
    feature_cols = [c for c in df.columns if c.startswith("z_feat_")]
    rows = []

    for params in PARAM_GRID:
        alpha = params["alpha"]
        l1_ratio = params["l1_ratio"]
        variant = f"en_alpha{alpha:g}_l1_{l1_ratio:g}"
        print(f"\nRunning {variant}")

        df_pred = expanding_window_elastic_net(
            df=df,
            feature_cols=feature_cols,
            target_col="target_winsorized_demeaned",
            first_pred_year=VALIDATION_FIRST_YEAR,
            last_pred_year=VALIDATION_LAST_YEAR,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_train_rows=MAX_TRAIN_ROWS,
        )
        df_model = df_pred[df_pred["score_elastic_net"].notna()]
        summary = ic_summary(df_model, ["score_elastic_net"]).iloc[0].to_dict()
        deciles = decile_spread_summary(
            df_model,
            score_col="score_elastic_net",
            target_col="target_winsorized_demeaned",
        )
        summary.update({
            "variant": variant,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "n_features": len(feature_cols),
            "validation_years": (
                f"{VALIDATION_FIRST_YEAR}-{VALIDATION_LAST_YEAR}"
            ),
            "max_train_rows": MAX_TRAIN_ROWS,
            "top_bottom_spread": deciles.loc[
                deciles["decile"] == "top_minus_bottom", "mean"
            ].iloc[0],
        })
        rows.append(summary)

    result = pd.DataFrame(rows)[
        [
            "variant",
            "alpha",
            "l1_ratio",
            "n_features",
            "validation_years",
            "max_train_rows",
            "mean_ic",
            "std_ic",
            "t_stat",
            "n_days",
            "top_bottom_spread",
        ]
    ].sort_values(["mean_ic", "t_stat"], ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved Elastic Net tuning results to: {OUTPUT_PATH}")
    print(result.to_string(index=False))
