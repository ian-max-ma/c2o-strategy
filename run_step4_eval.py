from pathlib import Path
import pandas as pd
from step4_alpha.evaluation import (
    ic_summary,
    decile_spread_summary,
    plot_decile_spread,
    plot_ic_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"
OUTPUT_DIR = PROJECT_ROOT / "step4_alpha" / "eval_output"
OUTPUT_PATH = OUTPUT_DIR / "ic_summary.csv"

IC_PLOT_PATH = OUTPUT_DIR / "ic_summary.png"
DECILE_PLOT_PATH = OUTPUT_DIR / "decile_spread.png"


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs/alpha_scores.parquet. "
            "Please run `python3 run_step4.py` first."
        )

    df = pd.read_parquet(INPUT_PATH)

    score_cols = [c for c in df.columns if c.startswith("z_feat_")]
    score_cols.append("score_baseline")

    ic = ic_summary(df, score_cols)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ic.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved IC summary to: {OUTPUT_PATH}")
    print(ic.to_string(index=False))

    # decile spread check

    DECILE_OUTPUT_PATH = OUTPUT_DIR / "decile_spread.csv"

    decile = decile_spread_summary(df, score_col="score_baseline")
    decile.to_csv(DECILE_OUTPUT_PATH, index=False)

    print(f"\nSaved decile spread to: {DECILE_OUTPUT_PATH}")
    print(decile.to_string(index=False))


    # plot
    plot_ic_summary(ic, IC_PLOT_PATH)
    plot_decile_spread(decile, DECILE_PLOT_PATH)

    print(f"Saved IC plot to: {IC_PLOT_PATH}")
    print(f"Saved decile plot to: {DECILE_PLOT_PATH}")

