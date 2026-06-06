import numpy as np
import pandas as pd

from step5_portfolio.portfolio import (
    PortfolioConfig,
    backtest_positions,
    build_daily_positions,
    borrow_sensitivity_analysis,
    cap_sensitivity_analysis,
    cost_schedule_summary,
    prepare_step5_input,
    robustness_diagnostics,
    run_aum_backtests,
)


def test_cost_schedule_roundtrip_is_four_bps() -> None:
    config = PortfolioConfig(aum=1_000_000)
    schedule = cost_schedule_summary(config)

    roundtrip = schedule.loc[
        schedule["item"] == "roundtrip_commission_slippage", "bps"
    ].iloc[0]

    assert roundtrip == 4.0
    assert config.per_leg_cost_bps == 2.0


def test_daily_trading_cost_is_four_bps_at_full_gross() -> None:
    config = PortfolioConfig(aum=1_000_000)
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "position": [500_000.0, -500_000.0],
            "target_raw": [0.01, -0.01],
            "adv20": [100_000_000.0, 100_000_000.0],
            "htb_tier": ["A", "A"],
        }
    )

    daily = backtest_positions(positions, config)

    assert np.isclose(daily["gross_exposure"].iloc[0], 1.0)
    assert np.isclose(daily["roundtrip_turnover"].iloc[0], 2.0)
    assert np.isclose(daily["trading_cost_return"].iloc[0], 0.0004)
    assert np.isclose(daily["commission_return"].iloc[0], 0.0001)
    assert np.isclose(daily["slippage_return"].iloc[0], 0.0003)


def test_daily_positions_are_dollar_neutral_and_respect_adv_cap() -> None:
    config = PortfolioConfig(
        aum=1_000_000,
        score_col="score_baseline",
        basket_quantile=0.25,
        participation_cap=0.05,
    )
    day = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 8),
            "instrument_id": range(8),
            "score_baseline": [8, 7, 6, 5, 4, 3, 2, 1],
            "target_raw": [0.0] * 8,
            "adv20": [20_000_000.0] * 8,
            "htb_tier": ["A"] * 8,
        }
    )

    positions = build_daily_positions(day, config)

    assert np.isclose(positions["position"].sum(), 0.0)
    assert (positions["position"].abs() <= positions["adv20"] * 0.05 + 1e-8).all()
    assert (positions["side"] == "long").sum() == 2
    assert (positions["side"] == "short").sum() == 2


def test_target_raw_uses_next_day_overnight_return() -> None:
    alpha = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "instrument_id": [1, 1, 1],
            "score_baseline": [0.1, 0.2, 0.3],
        }
    )
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "instrument_id": [1, 1, 1],
            "eligible": [True, True, True],
            "adv20": [100_000_000.0, 100_000_000.0, 100_000_000.0],
            "r_ON": [0.01, 0.02, 0.03],
            "htb_tier": ["A", "A", "A"],
        }
    )

    out = prepare_step5_input(alpha, panel).sort_values("date")

    assert np.isclose(
        out.loc[out["date"] == pd.Timestamp("2020-01-01"), "target_raw"].iloc[0],
        0.02,
    )
    assert np.isclose(
        out.loc[out["date"] == pd.Timestamp("2020-01-02"), "target_raw"].iloc[0],
        0.03,
    )


def test_robustness_diagnostics_include_lo_and_rolling_checks() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=300),
            "net_return": np.linspace(-0.001, 0.001, 300),
        }
    )

    out = robustness_diagnostics(daily)

    assert "lo_adjusted_net_sharpe_lag5" in set(out["diagnostic"])
    assert "worst_rolling_3m_return" in set(out["diagnostic"])
    assert "worst_rolling_6m_return" in set(out["diagnostic"])
    assert "worst_rolling_12m_return" in set(out["diagnostic"])
    assert out["diagnostic"].str.startswith("annual_net_sharpe_").any()


def test_borrow_sensitivity_outputs_hard_exclusion_scenarios() -> None:
    rows = []
    dates = pd.bdate_range("2020-01-01", periods=8)
    for date in dates:
        for i in range(20):
            rows.append(
                {
                    "date": date,
                    "instrument_id": i,
                    "score_baseline": 20 - i,
                    "target_raw": 0.001 * (1 if i < 10 else -1),
                    "adv20": 100_000_000.0,
                    "htb_tier": "C" if i in {0, 19} else "A",
                    "dsi": 0.15 if i in {18, 19} else 0.05,
                }
            )
    step5_df = pd.DataFrame(rows)
    summary, _, positions_by_aum = run_aum_backtests(
        step5_df,
        score_col="score_baseline",
        aum_levels={"250M": 250_000_000},
    )

    out = borrow_sensitivity_analysis(
        step5_df,
        summary,
        positions_by_aum["250M"],
        "score_baseline",
        aum=250_000_000,
    )

    assert "hard_exclude_tier_C" in set(out["scenario"])
    assert "hard_exclude_tier_BC" in set(out["scenario"])
    assert "short_interest_gt_10pct_short_book_contribution" in set(out["scenario"])


def test_cap_sensitivity_compares_headline_and_loose_cap() -> None:
    rows = []
    dates = pd.bdate_range("2020-01-01", periods=5)
    for date in dates:
        for i in range(20):
            rows.append(
                {
                    "date": date,
                    "instrument_id": i,
                    "score_baseline": 20 - i,
                    "target_raw": 0.001 * (1 if i < 10 else -1),
                    "adv20": 1_000_000.0,
                    "htb_tier": "A",
                }
            )
    step5_df = pd.DataFrame(rows)

    out = cap_sensitivity_analysis(
        step5_df,
        score_col="score_baseline",
        aum_levels={"250M": 250_000_000},
        headline_cap=0.05,
        loose_cap=1.0,
    )

    assert "headline_5pct_adv_cap" in set(out["scenario"])
    assert "loose_100pct_adv_reference" in set(out["scenario"])
    assert out.loc[
        out["scenario"] == "headline_5pct_adv_cap",
        "avg_gross_exposure",
    ].iloc[0] < out.loc[
        out["scenario"] == "loose_100pct_adv_reference",
        "avg_gross_exposure",
    ].iloc[0]
