from pathlib import Path

import numpy as np
import pandas as pd

from step4_alpha.model import build_baseline_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "panel_step3.parquet"
ALL_DATA_PATH = PROJECT_ROOT / "data" / "all_data.parquet"
CHEAPNESS_PATH = PROJECT_ROOT / "data" / "cheapness_scores.parquet"
DOWNGRADE_PATH = PROJECT_ROOT / "data" / "rolling_scores_downgrade.csv"
UPGRADE_PATH = PROJECT_ROOT / "data" / "rolling_scores_upgrade.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"


ALL_DATA_FEATURE_COLS = [
    "piot_norm",
    "short_interest",
    "asset_turnover_ratio",
    "current_liabilities",
    "ev_to_ebit",
    "gross_profit_margin",
    "interest_expenses_net",
    "long_term_debt",
    "net_cash_flow_oper",
    "net_debt_to_equity",
    "net_income_before_extr",
    "price_to_book",
    "total_assets",
    "total_curr_assets",
    "epsp",
    "epsf",
    "reps1",
    "repsf4",
    "sue",
    "inesp",
    "inesn",
    "reps41",
    "repsfs",
    "repsfl",
    "nspc5",
    "deps",
    "value_mean_eps",
    "value_smart_eps",
    "value_split_adj_mean_eps",
    "value_split_adj_smart_eps",
]

CHEAPNESS_FEATURE_COLS = [
    "valuation_score",
    "quality_score",
    "health_score",
    "momentum_score",
    "final_score_clean",
    "score_velocity",
    "score_acceleration",
    "regime_break",
    "value_trap",
]

# Realised y_true rating outcomes are deliberately excluded.
CREDIT_FEATURE_COLS = [
    "downgrade_prob_1m",
    "downgrade_prob_2m",
    "downgrade_prob_3m",
    "downgrade_prob_6m",
    "upgrade_prob_1m",
    "upgrade_prob_2m",
    "upgrade_prob_3m",
    "upgrade_prob_6m",
]

PANEL_COLS = [
    "date",
    "ticker",
    "instrument_id",
    "high",
    "low",
    "close",
    "volume",
    "market_cap",
    "r_ON",
    "r_ID",
    "r_CC",
    "r_CC_lag1",
    "r_ID_lag1",
    "market_cap_lag1",
    "vol20",
    "adv20",
    "eligible",
    "dsi",
    "dtcn",
    "ddtcn",
    "htb_tier",
    "htb_flag",
]


def load_step3_panel() -> pd.DataFrame:
    """Load only the Step 3 columns needed by Step 4."""
    return pd.read_parquet(INPUT_PATH, columns=PANEL_COLS)


def add_all_data_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge one-day-lagged all_data signals onto eligible stock-days."""
    all_data = pd.read_parquet(
        ALL_DATA_PATH,
        columns=["stock_id", "date"] + ALL_DATA_FEATURE_COLS,
    ).rename(columns={"stock_id": "instrument_id"})
    all_data["date"] = pd.to_datetime(all_data["date"])
    all_data = all_data.sort_values(["instrument_id", "date"])

    lagged = all_data.groupby("instrument_id")[ALL_DATA_FEATURE_COLS].shift(1)
    lagged.columns = [f"feat_{col}_lag1" for col in ALL_DATA_FEATURE_COLS]
    all_data = pd.concat(
        [
            all_data[["instrument_id", "date"]].reset_index(drop=True),
            lagged.reset_index(drop=True),
        ],
        axis=1,
    )
    return df.merge(
        all_data,
        on=["instrument_id", "date"],
        how="left",
        validate="many_to_one",
    )


def add_cheapness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge one-day-lagged valuation, quality and momentum scores."""
    cheapness = pd.read_parquet(
        CHEAPNESS_PATH,
        columns=["instrument_id", "date", "ticker"] + CHEAPNESS_FEATURE_COLS,
    )
    cheapness["date"] = pd.to_datetime(cheapness["date"])
    cheapness["value_trap"] = cheapness["value_trap"].astype(float)
    cheapness = cheapness.sort_values(["instrument_id", "ticker", "date"])

    lagged = cheapness.groupby(
        ["instrument_id", "ticker"]
    )[CHEAPNESS_FEATURE_COLS].shift(1)
    lagged.columns = [f"feat_{col}_lag1" for col in CHEAPNESS_FEATURE_COLS]
    cheapness = pd.concat(
        [
            cheapness[["instrument_id", "date", "ticker"]].reset_index(drop=True),
            lagged.reset_index(drop=True),
        ],
        axis=1,
    )
    return df.merge(
        cheapness,
        on=["instrument_id", "date", "ticker"],
        how="left",
        validate="one_to_one",
    )


def _load_credit_probabilities(path: Path, prefix: str) -> pd.DataFrame:
    """Pivot four rating forecast horizons to one row per ticker-date."""
    scores = pd.read_csv(path, usecols=["ticker", "date", "target", "prob"])
    scores["date"] = pd.to_datetime(scores["date"])
    scores["horizon"] = scores["target"].str.extract(r"_(1m|2m|3m|6m)$")
    scores = (
        scores.pivot(index=["ticker", "date"], columns="horizon", values="prob")
        .rename(columns=lambda horizon: f"{prefix}_prob_{horizon}")
        .reset_index()
    )
    scores.columns.name = None
    return scores


def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """As-of merge the latest credit forecasts available before each date."""
    downgrade = _load_credit_probabilities(DOWNGRADE_PATH, "downgrade")
    upgrade = _load_credit_probabilities(UPGRADE_PATH, "upgrade")
    credit = downgrade.merge(
        upgrade,
        on=["ticker", "date"],
        how="outer",
        validate="one_to_one",
    )
    credit = credit.rename(columns={"date": "credit_date"})
    credit["credit_date"] = (
        credit["credit_date"] + pd.Timedelta(days=1)
    ).astype("datetime64[ns]")
    credit = credit.sort_values("credit_date")

    left = df.reset_index(names="_row_order").sort_values("date")
    merged = pd.merge_asof(
        left,
        credit,
        left_on="date",
        right_on="credit_date",
        by="ticker",
        direction="backward",
    )
    merged = merged.drop(columns=["credit_date"]).sort_values("_row_order")
    merged = merged.drop(columns=["_row_order"]).reset_index(drop=True)
    return merged.rename(
        columns={col: f"feat_{col}_lag1" for col in CREDIT_FEATURE_COLS}
    )


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build next-trading-day overnight targets on complete stock histories."""
    df = df.sort_values(["instrument_id", "date"]).copy()
    df["target_raw"] = df.groupby("instrument_id")["r_ON"].shift(-1)
    df["target_raw_demeaned"] = (
        df["target_raw"] - df.groupby("date")["target_raw"].transform("mean")
    )
    df["target_winsorized"] = df.groupby("date")["target_raw"].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
    )
    df["target_winsorized_demeaned"] = (
        df["target_winsorized"]
        - df.groupby("date")["target_winsorized"].transform("mean")
    )
    df["target_rank"] = df.groupby("date")["target_raw"].rank(pct=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build point-in-time features for the 15:50 decision on date t.

    r_ON,t is known at the open and may enter date-t rolling windows. Returns
    and prices requiring Close_t are shifted so their latest observation is
    date t-1.
    """
    df = df.sort_values(["instrument_id", "date"]).copy()
    grouped = df.groupby("instrument_id", sort=False)

    df["feat_r_cc_lag1"] = df["r_CC_lag1"]
    df["feat_r_id_lag1"] = df["r_ID_lag1"]
    df["feat_momentum_on_5"] = grouped["r_ON"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    df["feat_momentum_id_5"] = grouped["r_ID"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).mean()
    )

    # Step 2 vol20 on row t already uses r_CC observations through t-1.
    df["feat_vol20"] = df["vol20"]
    df["feat_vol20_on"] = grouped["r_ON"].transform(
        lambda s: s.rolling(20, min_periods=10).std() * np.sqrt(252)
    )
    df["feat_vol20_id"] = grouped["r_ID"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).std() * np.sqrt(252)
    )

    prev_high = grouped["high"].shift(1)
    prev_low = grouped["low"].shift(1)
    prev_close = grouped["close"].shift(1)
    safe_prev_close = prev_close.where(prev_close > 0)
    safe_vol20 = df["vol20"].where(df["vol20"] > 0)
    df["feat_range_lag1"] = (prev_high - prev_low) / safe_prev_close
    df["feat_jump_lag1"] = df["r_CC_lag1"].abs() / safe_vol20

    # Keep the original baseline liquidity and size terms unchanged.
    df["adv20_lag1"] = grouped["adv20"].shift(1)
    df["feat_log_adv20_lag1"] = np.log1p(df["adv20_lag1"])
    df["feat_log_mcap_lag1"] = np.log1p(df["market_cap_lag1"])

    # Screenshot liquidity definitions, all ending at t-1.
    safe_close = df["close"].where(df["close"] > 0)
    shares_outstanding = df["market_cap"] / safe_close
    daily_turnover = df["volume"] / shares_outstanding.where(
        shares_outstanding > 0
    )
    dollar_volume = df["close"] * df["volume"]
    daily_amihud = df["r_CC"].abs() / dollar_volume.where(dollar_volume > 0)
    df["feat_turnover20"] = daily_turnover.groupby(
        df["instrument_id"], sort=False
    ).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    df["feat_amihud20"] = daily_amihud.groupby(
        df["instrument_id"], sort=False
    ).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    df["feat_price_lag1"] = prev_close

    for col in ["dsi", "dtcn", "ddtcn", "htb_flag"]:
        if col in df.columns:
            df[f"feat_{col}_lag1"] = grouped[col].shift(1)
    if "feat_htb_flag_lag1" in df.columns:
        df["feat_htb_flag_lag1"] = df["feat_htb_flag_lag1"].astype(float)
    return df


def standardize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Cross-sectionally z-score features by date and neutral-fill missing."""
    df = df.copy()
    z_cols = [f"z_{col}" for col in feature_cols]
    values = df[feature_cols].astype("float32")
    grouped = values.groupby(df["date"])
    standardized = (values - grouped.transform("mean")) / grouped.transform("std")
    standardized = standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    standardized.columns = z_cols
    df[z_cols] = standardized.astype("float32")
    return df, z_cols


def _feature_columns() -> list[str]:
    """Return the ordered production feature list."""
    return [
        "feat_r_cc_lag1",
        "feat_r_id_lag1",
        "feat_momentum_on_5",
        "feat_momentum_id_5",
        "feat_vol20",
        "feat_vol20_on",
        "feat_vol20_id",
        "feat_range_lag1",
        "feat_jump_lag1",
        "feat_log_adv20_lag1",
        "feat_log_mcap_lag1",
        "feat_turnover20",
        "feat_amihud20",
        "feat_price_lag1",
        "feat_dsi_lag1",
        "feat_dtcn_lag1",
        "feat_ddtcn_lag1",
        "feat_htb_flag_lag1",
    ] + [
        f"feat_{col}_lag1"
        for col in (
            ALL_DATA_FEATURE_COLS
            + CHEAPNESS_FEATURE_COLS
            + CREDIT_FEATURE_COLS
        )
    ]


def build_alpha_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Build targets and 65 point-in-time standardized Step 4 features."""
    df = load_step3_panel()

    # Both shifts and rolling windows must run on complete trading histories.
    df = build_targets(df)
    df = build_features(df)

    df = df[df["eligible"]].copy()
    df = add_all_data_features(df)
    df = add_cheapness_features(df)
    df = add_credit_features(df)

    # Recompute cross-sectional target transforms on the tradable universe.
    df["target_winsorized"] = df.groupby("date")["target_raw"].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
    )
    df["target_winsorized_demeaned"] = (
        df["target_winsorized"]
        - df.groupby("date")["target_winsorized"].transform("mean")
    )
    df["target_rank"] = df.groupby("date")["target_raw"].rank(pct=True)

    feature_cols = [col for col in _feature_columns() if col in df.columns]
    df, z_cols = standardize_features(df, feature_cols)

    keep_cols = [
        "date",
        "ticker",
        "instrument_id",
        "target_raw",
        "target_winsorized_demeaned",
        "target_rank",
        "htb_tier",
        "htb_flag",
        "vol20",
        "adv20",
        "dsi",
        "dtcn",
        "ddtcn",
    ] + z_cols
    df = build_baseline_score(df[keep_cols])
    return df, z_cols


def feature_audit_table(feature_cols: list[str]) -> pd.DataFrame:
    """Document point-in-time timing for the Step 4 feature set."""
    rows = []
    for col in feature_cols:
        if col in {"feat_momentum_on_5", "feat_vol20_on"}:
            timing = "Includes r_ON,t, known at the opening auction on date t."
        elif col == "feat_vol20":
            timing = "Step 2 rolling volatility through date t-1."
        elif col.endswith("_lag1") or col in {
            "feat_momentum_id_5",
            "feat_vol20_id",
            "feat_turnover20",
            "feat_amihud20",
        }:
            timing = "Close-dependent inputs end at date t-1."
        else:
            timing = "Point-in-time source value, lagged before the Step 4 merge."
        rows.append(
            {
                "feature": col,
                "available_at_1550": True,
                "timing": timing,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frame, standardized_cols = build_alpha_dataset()
    print(frame.head())
    print(frame.shape)
    print("Standardized features:", standardized_cols)
