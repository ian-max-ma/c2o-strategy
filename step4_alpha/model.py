import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet


def build_baseline_score(df: pd.DataFrame) -> pd.DataFrame:
    """Build the original transparent nine-feature baseline score."""
    df = df.copy()
    df["score_baseline"] = (
        -df["z_feat_r_id_lag1"]
        - df["z_feat_r_cc_lag1"]
        + df["z_feat_vol20"]
        + df["z_feat_log_adv20_lag1"]
        + df["z_feat_log_mcap_lag1"]
        + df["z_feat_dsi_lag1"]
        - df["z_feat_dtcn_lag1"]
        - df["z_feat_ddtcn_lag1"]
        - df["z_feat_htb_flag_lag1"]
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
    max_train_rows: int | None = None,
) -> pd.Series:
    """Fit Elastic Net and return predictions indexed like test_df."""
    train = train_df.dropna(subset=[target_col])
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=random_state)

    x_train = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train[target_col].to_numpy(dtype=np.float32, copy=False)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=5000,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return pd.Series(
        model.predict(x_test),
        index=test_df.index,
        name="score_elastic_net",
    )


def expanding_window_elastic_net(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    first_pred_year: int = 2018,
    last_pred_year: int = 2024,
    alpha: float = 0.0001,
    l1_ratio: float = 0.5,
    max_train_rows: int | None = None,
) -> pd.DataFrame:
    """Generate annual expanding-window Elastic Net predictions."""
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
            f"[Elastic Net] Train <= {pred_year - 1}, predict {pred_year}: "
            f"{len(train_df):,} available train rows, "
            f"up to {max_train_rows or len(train_df):,} used, "
            f"{len(test_df):,} test rows"
        )
        pred = fit_predict_elastic_net(
            train_df,
            test_df,
            feature_cols,
            target_col=target_col,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_train_rows=max_train_rows,
        )
        df.loc[pred.index, "score_elastic_net"] = pred
    return df.drop(columns=["year"])


def fit_predict_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 500,
    max_features: str | float = "sqrt",
    random_state: int = 42,
    max_train_rows: int | None = 750_000,
) -> pd.Series:
    """Fit Random Forest and return predictions indexed like test_df."""
    train = train_df.dropna(subset=[target_col])
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=random_state)

    x_train = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train[target_col].to_numpy(dtype=np.float32, copy=False)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return pd.Series(
        model.predict(x_test),
        index=test_df.index,
        name="score_random_forest",
    )


def expanding_window_random_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_winsorized_demeaned",
    first_pred_year: int = 2018,
    last_pred_year: int = 2024,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 500,
    max_features: str | float = "sqrt",
    max_train_rows: int | None = 750_000,
) -> pd.DataFrame:
    """Generate annual expanding-window Random Forest predictions."""
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
            f"[Random Forest] Train <= {pred_year - 1}, predict {pred_year}: "
            f"{len(train_df):,} available train rows, "
            f"up to {max_train_rows or len(train_df):,} used, "
            f"{len(test_df):,} test rows"
        )
        pred = fit_predict_random_forest(
            train_df,
            test_df,
            feature_cols,
            target_col=target_col,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_train_rows=max_train_rows,
        )
        df.loc[pred.index, "score_random_forest"] = pred
    return df.drop(columns=["year"])
