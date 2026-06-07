from pathlib import Path

import pandas as pd

from step4_alpha.model import expanding_window_elastic_net
from step4_alpha.evaluation import decile_spread_summary, ic_summary


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"
OUTPUT_DIR = PROJECT_ROOT / "step4_alpha" / "eval_output"
OUTPUT_PATH = OUTPUT_DIR / "elastic_net_feature_ablation.csv"

VALIDATION_FIRST_YEAR = 2021
VALIDATION_LAST_YEAR = 2023
MAX_TRAIN_ROWS = 500_000


FEATURE_GROUPS = {
    "return": [
        "z_feat_r_cc_lag1",
        "z_feat_r_id_lag1",
        "z_feat_momentum_on_5",
        "z_feat_momentum_id_5",
    ],
    "risk": [
        "z_feat_vol20",
        "z_feat_vol20_on",
        "z_feat_vol20_id",
        "z_feat_range_lag1",
        "z_feat_jump_lag1",
    ],
    "liquidity_size": [
        "z_feat_log_adv20_lag1",
        "z_feat_log_mcap_lag1",
        "z_feat_turnover20",
        "z_feat_amihud20",
        "z_feat_price_lag1",
    ],
    "borrow_short_interest": [
        "z_feat_dsi_lag1",
        "z_feat_dtcn_lag1",
        "z_feat_ddtcn_lag1",
        "z_feat_htb_flag_lag1",
        "z_feat_short_interest_lag1",
    ],
    "fundamentals": [
        "z_feat_piot_norm_lag1",
        "z_feat_asset_turnover_ratio_lag1",
        "z_feat_current_liabilities_lag1",
        "z_feat_ev_to_ebit_lag1",
        "z_feat_gross_profit_margin_lag1",
        "z_feat_interest_expenses_net_lag1",
        "z_feat_long_term_debt_lag1",
        "z_feat_net_cash_flow_oper_lag1",
        "z_feat_net_debt_to_equity_lag1",
        "z_feat_net_income_before_extr_lag1",
        "z_feat_price_to_book_lag1",
        "z_feat_total_assets_lag1",
        "z_feat_total_curr_assets_lag1",
    ],
    "earnings_revisions": [
        "z_feat_epsp_lag1",
        "z_feat_epsf_lag1",
        "z_feat_reps1_lag1",
        "z_feat_repsf4_lag1",
        "z_feat_sue_lag1",
        "z_feat_inesp_lag1",
        "z_feat_inesn_lag1",
        "z_feat_reps41_lag1",
        "z_feat_repsfs_lag1",
        "z_feat_repsfl_lag1",
        "z_feat_nspc5_lag1",
        "z_feat_deps_lag1",
    ],
    "value_composite": [
        "z_feat_value_mean_eps_lag1",
        "z_feat_value_smart_eps_lag1",
        "z_feat_value_split_adj_mean_eps_lag1",
        "z_feat_value_split_adj_smart_eps_lag1",
        "z_feat_valuation_score_lag1",
        "z_feat_quality_score_lag1",
        "z_feat_health_score_lag1",
        "z_feat_momentum_score_lag1",
        "z_feat_final_score_clean_lag1",
        "z_feat_score_velocity_lag1",
        "z_feat_score_acceleration_lag1",
        "z_feat_regime_break_lag1",
        "z_feat_value_trap_lag1",
    ],
    "credit": [
        "z_feat_downgrade_prob_1m_lag1",
        "z_feat_downgrade_prob_2m_lag1",
        "z_feat_downgrade_prob_3m_lag1",
        "z_feat_downgrade_prob_6m_lag1",
        "z_feat_upgrade_prob_1m_lag1",
        "z_feat_upgrade_prob_2m_lag1",
        "z_feat_upgrade_prob_3m_lag1",
        "z_feat_upgrade_prob_6m_lag1",
    ],
}


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs/alpha_scores.parquet. "
            "Please run `python3 run_step4.py` first."
        )

    df = pd.read_parquet(INPUT_PATH)

    all_features = [c for c in df.columns if c.startswith("z_feat_")]

    grouped_features = {c for cols in FEATURE_GROUPS.values() for c in cols}
    if grouped_features != set(all_features):
        missing = sorted(set(all_features) - grouped_features)
        unknown = sorted(grouped_features - set(all_features))
        raise ValueError(
            f"Feature-group mismatch. Missing={missing}; unknown={unknown}"
        )

    feature_sets = {"elastic_net_all": all_features}
    feature_sets.update({
        f"elastic_net_no_{group_name}": [
            col for col in all_features if col not in group_cols
        ]
        for group_name, group_cols in FEATURE_GROUPS.items()
    })

    result_rows = []

    for variant_name, feature_cols in feature_sets.items():
        print(f"\nRunning {variant_name}")
        print("Features:", feature_cols)

        df_pred = expanding_window_elastic_net(
            df=df,
            feature_cols=feature_cols,
            target_col="target_winsorized_demeaned",
            first_pred_year=VALIDATION_FIRST_YEAR,
            last_pred_year=VALIDATION_LAST_YEAR,
            alpha=0.000001,
            l1_ratio=0.5,
            max_train_rows=MAX_TRAIN_ROWS,
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
        row["validation_years"] = (
            f"{VALIDATION_FIRST_YEAR}-{VALIDATION_LAST_YEAR}"
        )
        row["max_train_rows"] = MAX_TRAIN_ROWS
        deciles = decile_spread_summary(
            df_model,
            score_col=score_col,
            target_col="target_winsorized_demeaned",
        )
        row["top_bottom_spread"] = deciles.loc[
            deciles["decile"] == "top_minus_bottom", "mean"
        ].iloc[0]
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
            "validation_years",
            "max_train_rows",
            "mean_ic",
            "std_ic",
            "t_stat",
            "n_days",
            "top_bottom_spread",
            "features",
        ]
    ].sort_values("mean_ic", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved Elastic Net feature ablation to: {OUTPUT_PATH}")
    print(result.to_string(index=False))
