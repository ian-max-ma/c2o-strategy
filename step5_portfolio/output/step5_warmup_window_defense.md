# Step 5 warm-up window defense

The official report window remains 2010-2024. Dates before the first expanding-window ML prediction are retained in the daily return series as zero-exposure days rather than dropped.
This is conservative because it prevents the headline Sharpe from being inflated by reporting only the active 2018-2024 trading period.

First available ML score date: 2018-01-02.
Last available ML score date: 2024-12-31.
Maximum gross exposure before the first ML score date: 0.000000.

Why the 2010-2017 period is retained:
1. Step 4 uses an expanding-window design; the model is not allowed to trade until enough past data exist to train it causality-safely.
2. The flat warm-up dates are included in `daily_returns_250M.csv`, `performance_summary.csv` and the QuantStats tear-sheet, so the official 2010-2024 headline is not cherry-picked.
3. The active 2018-2024 period is reported separately in diagnostics and stress-window tables to make deployment behaviour transparent.
