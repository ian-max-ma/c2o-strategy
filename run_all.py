"""
Run the C2O coursework pipeline from Step 1 through Step 5.

Usage:
    python run_all.py

The default run rebuilds the report sanity outputs, the Step 1-3 panel, Step 4
alpha scores and Step 5 portfolio outputs. Feature ablation and model tuning
are optional because they are exploratory model-selection audits.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_stage(script_name: str, env: dict[str, str]) -> None:
    """Run one pipeline script with the current Python interpreter."""
    print("\n" + "=" * 78)
    print(f"Running {script_name}")
    print("=" * 78)
    subprocess.run(
        [sys.executable, script_name],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C2O coursework pipeline.")
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip report sanity scripts and run only production Step 1-5 artifacts.",
    )
    parser.add_argument(
        "--include-tuning",
        action="store_true",
        help="Also run Step 4 feature ablation and model-tuning audits.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    mpl_cache = PROJECT_ROOT / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))

    scripts: list[str] = []
    if not args.skip_sanity:
        scripts.extend(
            [
                "run_step1_sanity.py",
                "run_step2_sanity.py",
            ]
        )
    scripts.append("run_step2.py")
    if not args.skip_sanity:
        scripts.append("run_step3_sanity.py")
    scripts.extend(
        [
            "run_step4.py",
            "run_step4_eval.py",
        ]
    )
    if args.include_tuning:
        scripts.extend(
            [
                "run_step4_ablation.py",
                "run_step4_elastic_net_tuning.py",
                "run_step4_random_forest_tuning.py",
                "run_step4_tuning_holdout.py",
            ]
        )
    scripts.append("run_step5.py")

    for script in scripts:
        run_stage(script, env)

    print("\nPipeline complete. Step 5 outputs are in step5_portfolio/output/.")


if __name__ == "__main__":
    main()
