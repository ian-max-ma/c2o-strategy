"""
Step 3 sanity check for the hard-to-borrow / borrow-cost module.

This script assumes that `python run_step2.py` has already been run and that
`outputs/panel_step3.parquet` exists. It then runs the Step 3 borrow summary
checks, including:
- HTB tier distribution
- borrow cost schedule
- yearly HTB share
- affected eligible-universe fraction
- gross vs net borrow proxy
"""

from pathlib import Path

from step3_borrow.borrow_step3_summary import main


PANEL_PATH = Path("outputs/panel_step3.parquet")


if __name__ == "__main__":
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs/panel_step3.parquet. "
            "Please run `python run_step2.py` first."
        )

    main()