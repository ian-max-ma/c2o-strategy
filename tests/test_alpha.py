import pandas as pd

from step4_alpha.alpha import build_features


def test_same_day_overnight_return_is_used_without_a_lag() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "instrument_id": [1, 1],
            "r_ON": [0.01, 0.02],
            "r_CC_lag1": [0.03, 0.04],
            "r_ID_lag1": [0.02, 0.02],
            "vol20": [0.20, 0.21],
            "adv20": [10_000_000.0, 11_000_000.0],
            "market_cap_lag1": [1_000_000_000.0, 1_100_000_000.0],
        }
    )

    features = build_features(panel)

    assert features["feat_r_on_today"].tolist() == [0.01, 0.02]
