import gc
from pathlib import Path

import pandas as pd

from step4_alpha.evaluation import decile_spread_summary, ic_summary
from step4_alpha.model import (
    expanding_window_elastic_net,
    expanding_window_random_forest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ML_INPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores_ml.parquet"
OUTPUT_DIR = PROJECT_ROOT / "step4_alpha" / "eval_output"
ELASTIC_TUNING_PATH = OUTPUT_DIR / "elastic_net_tuning.csv"
RF_TUNING_PATH = OUTPUT_DIR / "random_forest_tuning.csv"
OUTPUT_PATH = OUTPUT_DIR / "tuning_holdout_2024.csv"

HOLDOUT_YEAR = 2024
RF_FINAL_TRAIN_ROWS = 750_000


def score_summary(
    df: pd.DataFrame,
    score_col: str,
    variant: str,
) -> dict:
    sample = df[(df["date"].dt.year == HOLDOUT_YEAR) & df[score_col].notna()]
    ic = ic_summary(sample, [score_col]).iloc[0]
    deciles = decile_spread_summary(sample, score_col=score_col)
    spread = deciles.loc[
        deciles["decile"] == "top_minus_bottom", "mean"
    ].iloc[0]
    return {
        "variant": variant,
        "holdout_year": HOLDOUT_YEAR,
        "mean_ic": ic["mean_ic"],
        "std_ic": ic["std_ic"],
        "t_stat": ic["t_stat"],
        "n_days": ic["n_days"],
        "top_bottom_spread": spread,
    }


if __name__ == "__main__":
    required = [ML_INPUT_PATH, ELASTIC_TUNING_PATH, RF_TUNING_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required tuning inputs: {missing}")

    df = pd.read_parquet(ML_INPUT_PATH)
    feature_cols = [c for c in df.columns if c.startswith("z_feat_")]
    rows = [
        score_summary(df, "score_elastic_net", "current_elastic_net"),
        score_summary(df, "score_random_forest", "current_random_forest"),
    ]

    elastic_best = (
        pd.read_csv(ELASTIC_TUNING_PATH)
        .sort_values(["mean_ic", "t_stat"], ascending=False)
        .iloc[0]
    )
    elastic_pred = expanding_window_elastic_net(
        df=df,
        feature_cols=feature_cols,
        target_col="target_winsorized_demeaned",
        first_pred_year=HOLDOUT_YEAR,
        last_pred_year=HOLDOUT_YEAR,
        alpha=float(elastic_best["alpha"]),
        l1_ratio=float(elastic_best["l1_ratio"]),
        max_train_rows=None,
    )
    elastic_row = score_summary(
        elastic_pred,
        "score_elastic_net",
        "validation_winner_elastic_net",
    )
    elastic_row["selected_params"] = (
        f"alpha={elastic_best['alpha']}, "
        f"l1_ratio={elastic_best['l1_ratio']}"
    )
    rows.append(elastic_row)
    del elastic_pred
    gc.collect()

    rf_best = (
        pd.read_csv(RF_TUNING_PATH)
        .sort_values(["mean_ic", "t_stat"], ascending=False)
        .iloc[0]
    )
    max_features_raw = rf_best["max_features"]
    max_features = (
        max_features_raw
        if max_features_raw == "sqrt"
        else float(max_features_raw)
    )
    rf_pred = expanding_window_random_forest(
        df=df,
        feature_cols=feature_cols,
        target_col="target_winsorized_demeaned",
        first_pred_year=HOLDOUT_YEAR,
        last_pred_year=HOLDOUT_YEAR,
        n_estimators=int(rf_best["n_estimators"]),
        max_depth=int(rf_best["max_depth"]),
        min_samples_leaf=int(rf_best["min_samples_leaf"]),
        max_features=max_features,
        max_train_rows=RF_FINAL_TRAIN_ROWS,
    )
    rf_row = score_summary(
        rf_pred,
        "score_random_forest",
        "validation_winner_random_forest",
    )
    rf_row["selected_params"] = (
        f"n_estimators={int(rf_best['n_estimators'])}, "
        f"max_depth={int(rf_best['max_depth'])}, "
        f"min_samples_leaf={int(rf_best['min_samples_leaf'])}, "
        f"max_features={max_features}"
    )
    rows.append(rf_row)

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved tuning holdout results to: {OUTPUT_PATH}")
    print(result.to_string(index=False))
