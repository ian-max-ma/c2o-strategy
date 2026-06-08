# c2o-strategy

C2O overnight strategy for the Machine Learning in Finance coursework.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the five coursework data files in `data/`:

```
data/
  prices.parquet
  sp500_tr.parquet
  sp500_constituents.parquet
  earnings_calendar.parquet
  short_interest_transfo.parquet
```

Then run:

```bash
python run_all.py
```

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
