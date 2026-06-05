from pathlib import Path

import pandas as pd

from step4_alpha.model import expanding_window_elastic_net
from step4_alpha.evaluation import ic_summary


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"
OUTPUT_DIR = PROJECT_ROOT / "step4_alpha" / "eval_output"
OUTPUT_PATH = OUTPUT_DIR / "elastic_net_feature_ablation.csv"


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs/alpha_scores.parquet. "
            "Please run `python3 run_step4.py` first."
        )

    df = pd.read_parquet(INPUT_PATH)

    all_features = [c for c in df.columns if c.startswith("z_feat_")]

    return_features = [
        "z_feat_r_cc_lag1",
        "z_feat_r_id_lag1",
    ]

    risk_features = [
        "z_feat_vol20_lag1",
    ]

    liquidity_size_features = [
        "z_feat_log_adv20_lag1",
        "z_feat_log_mcap_lag1",
    ]

    borrow_features = [
        "z_feat_dsi_lag1",
        "z_feat_dtcn_lag1",
        "z_feat_ddtcn_lag1",
        "z_feat_htb_flag_lag1",
    ]

    feature_sets = {
        "elastic_net_all": all_features,
        "elastic_net_no_return": [c for c in all_features if c not in return_features],
        "elastic_net_no_risk": [c for c in all_features if c not in risk_features],
        "elastic_net_no_liquidity_size": [
            c for c in all_features if c not in liquidity_size_features
        ],
        "elastic_net_no_borrow": [c for c in all_features if c not in borrow_features],
    }

    result_rows = []

    for variant_name, feature_cols in feature_sets.items():
        print(f"\nRunning {variant_name}")
        print("Features:", feature_cols)

        df_pred = expanding_window_elastic_net(
            df=df.copy(),
            feature_cols=feature_cols,
            target_col="target_winsorized_demeaned",
            first_pred_year=2018,
            last_pred_year=2024,
            alpha=0.00001,
            l1_ratio=0.5,
        )

        score_col = "score_elastic_net"
        df_model = df_pred[df_pred[score_col].notna()].copy()
        print(df_model["date"].dt.year.value_counts().sort_index())

        ic = ic_summary(
            df_model,
            score_cols=[score_col],
            target_col="target_winsorized_demeaned",
        )

        row = ic.iloc[0].to_dict()
        row["variant"] = variant_name
        row["n_features"] = len(feature_cols)
        row["features"] = ", ".join(feature_cols)

        result_rows.append(row)

        daily_score_std = df_model.groupby("date")[score_col].std()

        print("Non-constant days:", (daily_score_std > 0).sum())
        print("Constant days:", (daily_score_std == 0).sum())
        print("Min daily score std:", daily_score_std.min())
        print("Median daily score std:", daily_score_std.median())

    result = pd.DataFrame(result_rows)

    result = result[
        [
            "variant",
            "n_features",
            "mean_ic",
            "std_ic",
            "t_stat",
            "n_days",
            "features",
        ]
    ].sort_values("mean_ic", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved Elastic Net feature ablation to: {OUTPUT_PATH}")
    print(result.to_string(index=False))