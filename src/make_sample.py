"""Create a small shareable sample of the raw Kaggle weather CSV.

The raw file is git-ignored, but reviewers still need a tiny reproducible
sample in the repository. This script takes a random sample (fixed seed) and,
when possible, sorts it by the time column for readability.

Usage:
    python src/make_sample.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_CSV, SAMPLE_CSV, RANDOM_STATE  # noqa: E402
from src.inspect_data import pick_time_column  # noqa: E402

SAMPLE_ROWS = 300


def main() -> int:
    if not RAW_CSV.exists():
        print(
            f"Raw CSV not found: {RAW_CSV}\n"
            "Download it from Kaggle first (see data/README.md)."
        )
        return 1

    df = pd.read_csv(RAW_CSV, low_memory=False)
    n = min(SAMPLE_ROWS, len(df))
    sample = df.sample(n=n, random_state=RANDOM_STATE)

    time_col = pick_time_column(df)
    if time_col is not None:
        sample = sample.sort_values(time_col)

    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"Wrote {len(sample):,} rows x {sample.shape[1]} columns to {SAMPLE_CSV}")
    print("This file is small enough to commit; the raw CSV stays git-ignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
