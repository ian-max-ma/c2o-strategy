"""
Run the C2O coursework pipeline from Step 1 through Step 5.

Usage:
    python run_all.py

The default run rebuilds the report sanity outputs, the Step 1-3 panel, Step 4
alpha scores, evaluation, feature-ablation and model-tuning audits, and Step 5
portfolio outputs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_stage(command: tuple[str, ...], env: dict[str, str]) -> None:
    """Run one pipeline script with the current Python interpreter."""
    stage_name = " ".join(command)
    print("\n" + "=" * 78)
    print(f"Running {stage_name}")
    print("=" * 78)
    subprocess.run(
        [sys.executable, *command],
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
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    env = os.environ.copy()
    mpl_cache = PROJECT_ROOT / ".cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))

    stages: list[tuple[str, ...]] = []
    if not args.skip_sanity:
        stages.extend(
            [
                ("run_step1_sanity.py",),
                ("run_step2_sanity.py",),
            ]
        )
    stages.append(("run_step2.py",))
    if not args.skip_sanity:
        stages.append(("run_step3_sanity.py",))
    stages.extend(
        [
            ("run_step4.py",),
            ("run_step4_eval.py",),
            ("run_step4_ablation.py", "--mode", "groups"),
            ("run_step4_ablation.py", "--mode", "risk"),
            ("run_step4_elastic_net_tuning.py",),
            ("run_step4_random_forest_tuning.py",),
            ("run_step4_tuning_holdout.py",),
            ("run_step5.py",),
        ]
    )

    for command in stages:
        run_stage(command, env)

    print("\nPipeline complete. Step 5 outputs are in step5_portfolio/output/.")


if __name__ == "__main__":
    main()
