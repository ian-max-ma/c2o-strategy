import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


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


def fit_predict_elastic_net(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    alpha: float = 0.0001,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> pd.Series:
    """
    Fit Elastic Net on the training sample and predict scores for the test sample.

    The input features are already cross-sectionally standardized by date.
    A StandardScaler is still included in the pipeline to stabilise the model
    over the pooled training window.
    """
    train = train_df.dropna(subset=feature_cols + [target_col]).copy()
    test = test_df.dropna(subset=feature_cols).copy()

    model = make_pipeline(
        StandardScaler(),
        ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=5000,
            random_state=random_state,
        ),
    )

    model.fit(train[feature_cols], train[target_col])

    pred = pd.Series(
        model.predict(test[feature_cols]),
        index=test.index,
        name="score_elastic_net",
    )

    return pred


def expanding_window_elastic_net(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    first_pred_year: int = 2018,
    last_pred_year: int = 2024,
    alpha: float = 0.0001,
    l1_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Generate out-of-sample Elastic Net scores using an expanding-window scheme.

    For each prediction year Y, the model is trained only on observations with
    date year < Y, then used to predict year Y. This avoids random-split
    look-ahead bias and mimics a real forecasting setting.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["score_elastic_net"] = np.nan

    for pred_year in range(first_pred_year, last_pred_year + 1):
        train_df = df[df["year"] < pred_year]
        test_df = df[df["year"] == pred_year]

        if train_df.empty or test_df.empty:
            print(f"[Elastic Net] Skipping {pred_year}: empty train or test sample.")
            continue

        print(
            f"[Elastic Net] Train <= {pred_year - 1}, "
            f"predict {pred_year}: "
            f"{len(train_df):,} train rows, {len(test_df):,} test rows"
        )

        pred = fit_predict_elastic_net(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )

        df.loc[pred.index, "score_elastic_net"] = pred

    df = df.drop(columns=["year"])

    return df


def fit_predict_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 500,
    max_features: str = "sqrt",
    random_state: int = 42,
) -> pd.Series:
    """
    Fit Random Forest on the training sample and predict scores for the test sample.

    Random Forest is used to capture non-linear effects and feature interactions.
    Conservative tree depth and large leaf size are used to reduce overfitting
    on noisy daily stock-return data.
    """
    train = train_df.dropna(subset=feature_cols + [target_col]).copy()
    test = test_df.dropna(subset=feature_cols).copy()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(train[feature_cols], train[target_col])

    pred = pd.Series(
        model.predict(test[feature_cols]),
        index=test.index,
        name="score_random_forest",
    )

    return pred


def expanding_window_random_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    first_pred_year: int = 2018,
    last_pred_year: int = 2024,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 500,
    max_features: str = "sqrt",
) -> pd.DataFrame:
    """
    Generate out-of-sample Random Forest scores using an expanding-window scheme.

    For each prediction year Y, the model is trained only on observations with
    date year < Y, then used to predict year Y. This matches the timing logic
    used for Elastic Net.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["score_random_forest"] = np.nan

    for pred_year in range(first_pred_year, last_pred_year + 1):
        train_df = df[df["year"] < pred_year]
        test_df = df[df["year"] == pred_year]

        if train_df.empty or test_df.empty:
            print(f"[Random Forest] Skipping {pred_year}: empty train or test sample.")
            continue

        print(
            f"[Random Forest] Train <= {pred_year - 1}, "
            f"predict {pred_year}: "
            f"{len(train_df):,} train rows, {len(test_df):,} test rows"
        )

        pred = fit_predict_random_forest(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

        df.loc[pred.index, "score_random_forest"] = pred

    df = df.drop(columns=["year"])

    return df