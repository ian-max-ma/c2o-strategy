from pathlib import Path
from step4_alpha.alpha import build_alpha_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "alpha_scores.parquet"


if __name__ == "__main__":
    df, z_cols = build_alpha_dataset()

    output_cols = [
        "date",
        "ticker",
        "instrument_id",
        "target_raw",
        "target_winsorized_demeaned",
        "target_rank",
        "score_baseline",
        "htb_tier",
        "htb_flag",
    ] + z_cols

    output_cols = [col for col in output_cols if col in df.columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[output_cols].to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved alpha scores to: {OUTPUT_PATH}")
    print("Output shape:", df[output_cols].shape)
    print("Output columns:", output_cols)