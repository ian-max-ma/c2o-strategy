from pathlib import Path
from step4_alpha.alpha import build_alpha_dataset, feature_audit_table
from step4_alpha.model import (
    expanding_window_elastic_net,
    expanding_window_random_forest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"
ML_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores_ml.parquet"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "step4_alpha" / "eval_output"
FEATURE_AUDIT_PATH = EVAL_OUTPUT_DIR / "feature_audit.csv"

if __name__ == "__main__":
    df, z_cols = build_alpha_dataset()

    output_cols = [
        "date",
        "ticker",
        "instrument_id",
        "target_raw",
        "target_winsorized_demeaned",
        "target_rank",
        "score_baseline",
        "htb_tier",
        "htb_flag",
        "adv20",
        "dsi",
        "dtcn",
        "ddtcn",
    ] + z_cols

    output_cols = [col for col in output_cols if col in df.columns]

    BASELINE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[output_cols].to_parquet(BASELINE_OUTPUT_PATH, index=False)
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_audit_table([col.replace("z_", "", 1) for col in z_cols]).to_csv(
        FEATURE_AUDIT_PATH,
        index=False,
    )

    print(f"Saved baseline alpha scores to: {BASELINE_OUTPUT_PATH}")
    print("Baseline output shape:", df[output_cols].shape)
    print(f"Saved feature audit to: {FEATURE_AUDIT_PATH}")

    # Production and tuning use the same target and expanding-window scheme.
    df_ml = expanding_window_elastic_net(
        df=df[output_cols].copy(),
        feature_cols=z_cols,
        target_col="target_winsorized_demeaned",
        first_pred_year=2018,
        last_pred_year=2024,
        alpha=0.000001,
        l1_ratio=0.5,
    )

    df_ml = expanding_window_random_forest(
        df=df_ml,
        feature_cols=z_cols,
        target_col="target_winsorized_demeaned",
        first_pred_year=2018,
        last_pred_year=2024,
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=500,
        max_features="sqrt",
        max_train_rows=750_000,
    )

    df_ml.to_parquet(ML_OUTPUT_PATH, index=False)

    print(f"Saved ML alpha scores to: {ML_OUTPUT_PATH}")
    print("ML output shape:", df_ml.shape)
    print("Non-null Elastic Net scores:", df_ml["score_elastic_net"].notna().sum())
    print("Non-null Random Forest scores:", df_ml["score_random_forest"].notna().sum())
