import numpy as np
import pandas as pd

from config import ROUNDTRIP_BPS, TRADING_DAYS
from step4_alpha.evaluation import (
    cost_aware_decile_summary,
    daily_score_spread,
)


def _two_day_decile_panel() -> pd.DataFrame:
    rows = []
    for date, offset in [
        (pd.Timestamp("2024-01-02"), 0.0),
        (pd.Timestamp("2024-01-03"), 0.001),
    ]:
        for score in range(10):
            rows.append(
                {
                    "date": date,
                    "score": float(score),
                    "target_raw": offset + score / 10_000,
                    "htb_tier": "C" if score == 0 else "A",
                }
            )
    return pd.DataFrame(rows)


def test_daily_score_spread_uses_top_and_bottom_deciles() -> None:
    spread = daily_score_spread(_two_day_decile_panel(), "score")

    assert len(spread) == 2
    assert np.allclose(spread["top_minus_bottom"], 9 / 10_000)
    assert spread["n_top"].tolist() == [1, 1]
    assert spread["n_bottom"].tolist() == [1, 1]


def test_cost_aware_decile_summary_applies_trading_and_borrow_costs() -> None:
    result = cost_aware_decile_summary(_two_day_decile_panel(), "score").iloc[0]

    expected_gross = 0.5 * 9 / 10_000
    expected_trading = ROUNDTRIP_BPS / 10_000
    expected_borrow = 0.5 * 800 / 10_000 / TRADING_DAYS

    assert result["n_days"] == 2
    assert np.isclose(result["mean_gross_return"], expected_gross)
    assert np.isclose(result["mean_trading_cost_return"], expected_trading)
    assert np.isclose(result["mean_borrow_cost_return"], expected_borrow)
    assert np.isclose(
        result["mean_net_return"],
        expected_gross - expected_trading - expected_borrow,
    )
