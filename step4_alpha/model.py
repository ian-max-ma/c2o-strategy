import pandas as pd


def build_baseline_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the main 9-feature IC-aligned heuristic baseline alpha score.

    Higher score means higher expected next overnight return.

    The signs are chosen using single-feature IC diagnostics:
    - positive IC features enter with a positive sign
    - negative IC features enter with a negative sign

    This baseline is used as the transparent benchmark for later ML models.
    """
    df = df.copy()

    df["score_baseline"] = (
        -df["z_feat_r_id_lag1"]          # negative IC: intraday reversal
        -df["z_feat_r_cc_lag1"]          # negative IC: close-to-close reversal
        +df["z_feat_vol20_lag1"]         # positive IC: overnight risk premium
        +df["z_feat_log_adv20_lag1"]     # positive IC: liquidity / tradability
        +df["z_feat_log_mcap_lag1"]      # positive IC: large-cap tilt
        +df["z_feat_dsi_lag1"]           # positive IC: short-interest ratio
        -df["z_feat_dtcn_lag1"]          # negative IC: crowded short pressure
        -df["z_feat_ddtcn_lag1"]         # negative IC: rising crowding pressure
        -df["z_feat_htb_flag_lag1"]      # negative IC: hard-to-borrow penalty
    )

    return df

