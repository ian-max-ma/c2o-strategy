import numpy as np
import pandas as pd

from step4_alpha.alpha import build_features


def test_return_and_risk_features_obey_the_1550_cutoff() -> None:
    dates = pd.bdate_range("2020-01-01", periods=25)
    panel = pd.DataFrame(
        {
            "date": dates,
            "instrument_id": 1,
            "r_ON": np.arange(1, 26, dtype=float) / 1000,
            "r_ID": np.arange(101, 126, dtype=float) / 1000,
            "r_CC": np.arange(201, 226, dtype=float) / 1000,
            "r_CC_lag1": np.r_[np.arange(200, 224), -224] / 1000,
            "r_ID_lag1": np.arange(100, 125, dtype=float) / 1000,
            "high": np.arange(111, 136, dtype=float),
            "low": np.arange(91, 116, dtype=float),
            "close": np.arange(101, 126, dtype=float),
            "volume": np.arange(1_000, 1_025, dtype=float),
            "market_cap": np.arange(10_000_000, 10_000_025, dtype=float),
            "vol20": 0.20,
            "adv20": 1_000_000.0,
            "market_cap_lag1": 9_999_999.0,
        }
    )

    features = build_features(panel)
    row = features.iloc[-1]

    assert np.isclose(row["feat_momentum_on_5"], panel["r_ON"].iloc[-5:].mean())
    assert np.isclose(
        row["feat_momentum_id_5"],
        panel["r_ID"].iloc[-6:-1].mean(),
    )
    assert np.isclose(row["feat_vol20"], panel["vol20"].iloc[-1])
    assert np.isclose(
        row["feat_vol20_id"],
        panel["r_ID"].iloc[-21:-1].std(ddof=1) * np.sqrt(252),
    )
    assert np.isclose(
        row["feat_range_lag1"],
        (panel["high"].iloc[-2] - panel["low"].iloc[-2])
        / panel["close"].iloc[-2],
    )
    assert np.isclose(
        row["feat_jump_lag1"],
        abs(panel["r_CC_lag1"].iloc[-1]) / panel["vol20"].iloc[-1],
    )


def test_liquidity_features_use_only_dates_through_t_minus_one() -> None:
    dates = pd.bdate_range("2020-01-01", periods=25)
    panel = pd.DataFrame(
        {
            "date": dates,
            "instrument_id": 1,
            "r_ON": 0.001,
            "r_ID": 0.002,
            "r_CC": np.arange(1, 26, dtype=float) / 1000,
            "r_CC_lag1": np.arange(25, dtype=float) / 1000,
            "r_ID_lag1": 0.002,
            "high": 12.0,
            "low": 8.0,
            "close": np.arange(10, 35, dtype=float),
            "volume": np.arange(1_000, 1_025, dtype=float),
            "market_cap": np.arange(10_000_000, 10_000_025, dtype=float),
            "vol20": 0.20,
            "adv20": 1_000_000.0,
            "market_cap_lag1": 9_999_999.0,
        }
    )

    features = build_features(panel)
    row = features.iloc[-1]
    prior = panel.iloc[-21:-1]
    prior_turnover = prior["volume"] / (
        prior["market_cap"] / prior["close"]
    )
    prior_amihud = prior["r_CC"].abs() / (
        prior["close"] * prior["volume"]
    )

    assert np.isclose(row["feat_turnover20"], prior_turnover.mean())
    assert np.isclose(row["feat_amihud20"], prior_amihud.mean())
    assert np.isclose(row["feat_price_lag1"], panel["close"].iloc[-2])
