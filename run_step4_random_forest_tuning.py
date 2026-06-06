from pathlib import Path

import pandas as pd

from step4_alpha.evaluation import ic_summary
from step4_alpha.model import expanding_window_random_forest


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"
OUTPUT_DIR = PROJECT_ROOT / "step4_alpha" / "eval_output"
OUTPUT_PATH = OUTPUT_DIR / "random_forest_tuning.csv"


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs/alpha_scores.parquet. "
            "Please run `python3 run_step4.py` first."
        )

    df = pd.read_parquet(INPUT_PATH)

    feature_cols = [c for c in df.columns if c.startswith("z_feat_")]

    param_grid = [
        {
            "variant": "rf_depth5_leaf500",
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_leaf": 500,
            "max_features": "sqrt",
        },
        {
            "variant": "rf_depth6_leaf300",
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 300,
            "max_features": "sqrt",
        },
        {
            "variant": "rf_depth8_leaf500",
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 500,
            "max_features": "sqrt",
        },
        {
            "variant": "rf_depth6_leaf1000",
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 1000,
            "max_features": "sqrt",
        },
        {
            "variant": "rf_depth6_leaf100",
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
        },
    ]

    rows = []

    for params in param_grid:
        variant = params["variant"]

        print(f"\nRunning {variant}")
        print(params)

        df_pred = expanding_window_random_forest(
            df=df.copy(),
            feature_cols=feature_cols,
            target_col="target_winsorized_demeaned",
            first_pred_year=2018,
            last_pred_year=2024,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
        )

        score_col = "score_random_forest"
        df_model = df_pred[df_pred[score_col].notna()].copy()

        ic = ic_summary(
            df_model,
            score_cols=[score_col],
            target_col="target_winsorized_demeaned",
        )

        row = ic.iloc[0].to_dict()

        row["variant"] = variant
        row["n_estimators"] = params["n_estimators"]
        row["max_depth"] = params["max_depth"]
        row["min_samples_leaf"] = params["min_samples_leaf"]
        row["max_features"] = params["max_features"]

        rows.append(row)

    result = pd.DataFrame(rows)

    result = result[
        [
            "variant",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "mean_ic",
            "std_ic",
            "t_stat",
            "n_days",
        ]
    ].sort_values("mean_ic", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved Random Forest tuning results to: {OUTPUT_PATH}")
    print(result.to_string(index=False))
