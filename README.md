# c2o-strategy

C2O overnight strategy for the Machine Learning in Finance coursework.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the coursework data files in `data/`:

```
data/
  prices.parquet
  sp500_tr.parquet
  sp500_constituents.parquet
  earnings_calendar.parquet
  short_interest_transfo.parquet
  all_data.parquet
  cheapness_scores.parquet
  rolling_scores_downgrade.csv
  rolling_scores_upgrade.csv
  regime.parquet          (optional — used for regime IC diagnostics only)
```

Then run:

```bash
python run_all.py
```

This runs every stage from Step 1 through Step 5, including the Step 4
evaluation, group and risk-feature ablations, Elastic Net and Random Forest
tuning, and the 2024 tuning holdout.

For a faster production-only rebuild without the report sanity tables:

```bash
python run_all.py --skip-sanity
```

The required Step 5 portfolio outputs are written to `step5_portfolio/output/`.
The 250M QuantStats tear-sheet is generated as
`step5_portfolio/output/quantstats_250M_SP500_TR.html`.

Step 5 defaults to report mode, which filters portfolio outputs to the
development cutoff in `config.py`:

```bash
python run_step5.py --mode report
```

If the marker supplies post-2024 scores for held-out evaluation, Step 5 can be
run without the development-window filter:

```bash
python run_step5.py --mode heldout
```

## Held-out evaluation (marker instructions)

To evaluate on the 2025–2026 held-out window:

1. Place the extended data files (covering through 2026) in `data/`, replacing
   the 2010–2024 versions.

2. In `run_step4.py`, change the two `last_pred_year=2024` arguments to
   `last_pred_year=2026` (one for Elastic Net, one for Random Forest).
   The loader in `step1_panel/loader.py` enforces `TRAIN_END` as a hard
   cutoff — set `TRAIN_END = "2026-12-31"` in `config.py` to allow
   the panel and scores to extend through 2026.  The model training loop
   is expanding-window, so no future targets enter the training set.

3. Run the full pipeline:

```bash
python run_all.py
```

4. Or, if only the held-out portfolio results are needed:

```bash
python run_step5.py --mode heldout
```

## Pipeline

- `run_step2.py` rebuilds the Step 1-3 daily panel at `outputs/panel_step3.parquet`.
- `run_step4.py` creates baseline and ML alpha scores.
- `run_step4_eval.py` writes IC and decile-spread diagnostics.
- `run_step5.py` builds the 50M, 250M and 1B portfolios, applies the fixed Table
  6.1 costs, and writes all Step 5 tables, charts, captions and self-checks.

The development cutoff is fixed in `config.py` as `TRAIN_END = "2024-12-31"`.
The final Step 5 score is fixed in `config.py` as
`FINAL_SCORE_COL = "score_random_forest"`; other score columns are reported only
as diagnostics.
