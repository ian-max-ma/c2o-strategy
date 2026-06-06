from pathlib import Path
import pandas as pd
from step4_alpha.evaluation import (
    ic_summary,
    decile_spread_summary,
    plot_decile_spread,
    plot_ic_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores_ml.parquet"
BASELINE_INPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"

OUTPUT_DIR = PROJECT_ROOT / "step4_alpha" / "eval_output"
OUTPUT_PATH = OUTPUT_DIR / "ic_summary_ml.csv"
BASELINE_FULL_IC_PATH = OUTPUT_DIR / "ic_baseline_full_sample.csv"

IC_PLOT_PATH = OUTPUT_DIR / "ic_summary_ml.png"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Missing outputs/alpha_scores_ml.parquet. "
        "Please run `python3 run_step4.py` first."
    )

if __name__ == "__main__":

    df = pd.read_parquet(INPUT_PATH)

    score_cols = [c for c in df.columns if c.startswith("z_feat_")]

    for col in ["score_baseline", "score_elastic_net", "score_random_forest"]:
        if col in df.columns:
            score_cols.append(col)

    model_score_cols = [
        col for col in ["score_elastic_net", "score_random_forest"]
        if col in df.columns
    ]

    if model_score_cols:
        df_model = df.dropna(subset=model_score_cols).copy()
    else:
        df_model = df.copy()

    print(f"Evaluation sample rows: {len(df_model):,}")
    print(f"Evaluation date range: {df_model['date'].min()} → {df_model['date'].max()}")

    ic = ic_summary(df_model, score_cols)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ic.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved IC summary to: {OUTPUT_PATH}")
    print(ic.to_string(index=False))

    # Full-sample baseline diagnostic.
    # This uses the baseline-only alpha file, so it is not restricted to the
    # 2018-2024 OOS period where ML predictions are available.
    if BASELINE_INPUT_PATH.exists():
        df_baseline_full = pd.read_parquet(BASELINE_INPUT_PATH)

        ic_baseline_full = ic_summary(
            df_baseline_full,
            score_cols=["score_baseline"],
            target_col="target_winsorized_demeaned",
        )

        ic_baseline_full.to_csv(BASELINE_FULL_IC_PATH, index=False)

        print(f"\nSaved full-sample baseline IC to: {BASELINE_FULL_IC_PATH}")
        print(ic_baseline_full.to_string(index=False))
    else:
        print(
            "\nSkipping full-sample baseline IC: "
            "outputs/alpha_scores.parquet not found."
        )

    # decile spread check

    DECILE_BASELINE_OUTPUT_PATH = OUTPUT_DIR / "decile_spread_baseline.csv"
    DECILE_ELASTIC_NET_OUTPUT_PATH = OUTPUT_DIR / "decile_spread_elastic_net.csv"

    DECILE_BASELINE_PLOT_PATH = OUTPUT_DIR / "decile_spread_baseline.png"
    DECILE_ELASTIC_NET_PLOT_PATH = OUTPUT_DIR / "decile_spread_elastic_net.png"

    DECILE_RANDOM_FOREST_OUTPUT_PATH = OUTPUT_DIR / "decile_spread_random_forest.csv"
    DECILE_RANDOM_FOREST_PLOT_PATH = OUTPUT_DIR / "decile_spread_random_forest.png"


    decile_baseline = decile_spread_summary(df_model, score_col="score_baseline")
    decile_baseline.to_csv(DECILE_BASELINE_OUTPUT_PATH, index=False)

    plot_decile_spread(
        decile_baseline,
        DECILE_BASELINE_PLOT_PATH,
        title="Baseline Score Decile Spread",
    )

    print(f"\nSaved baseline decile spread to: {DECILE_BASELINE_OUTPUT_PATH}")
    print(decile_baseline.to_string(index=False))

    if "score_elastic_net" in df_model.columns:
        decile_elastic_net = decile_spread_summary(df_model, score_col="score_elastic_net")
        decile_elastic_net.to_csv(DECILE_ELASTIC_NET_OUTPUT_PATH, index=False)

        plot_decile_spread(
            decile_elastic_net,
            DECILE_ELASTIC_NET_PLOT_PATH,
            title="Elastic Net Score Decile Spread",
        )

        print(f"\nSaved Elastic Net decile spread to: {DECILE_ELASTIC_NET_OUTPUT_PATH}")
        print(decile_elastic_net.to_string(index=False))

    if "score_random_forest" in df_model.columns:
        decile_random_forest = decile_spread_summary(df_model, score_col="score_random_forest")
        decile_random_forest.to_csv(DECILE_RANDOM_FOREST_OUTPUT_PATH, index=False)

        plot_decile_spread(
            decile_random_forest,
            DECILE_RANDOM_FOREST_PLOT_PATH,
            title="Random Forest Score Decile Spread",
        )

        print(f"\nSaved Random Forest decile spread to: {DECILE_RANDOM_FOREST_OUTPUT_PATH}")
        print(decile_random_forest.to_string(index=False))

    # plot
    plot_ic_summary(ic, IC_PLOT_PATH)
    print(f"Saved IC plot to: {IC_PLOT_PATH}")
