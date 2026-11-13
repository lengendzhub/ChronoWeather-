"""Shared paths and constants for the PM Accelerator weather project.

Keeping paths in one place means notebooks and scripts never disagree about
where the raw data, figures, or reports live.

Import example (notebooks add the project root to sys.path first):

    from src.config import RAW_CSV, FIG_DIR, RANDOM_STATE
"""

from pathlib import Path

# parents[1] -> the project root (src/config.py -> src -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "GlobalWeatherRepository.csv"
SAMPLE_CSV = DATA_DIR / "sample.csv"

# Kaggle source. Used by src/load_data.py when the CSV is not downloaded yet.
KAGGLE_DATASET = "nelgiriyewithana/global-weather-repository"
KAGGLE_FILE = "GlobalWeatherRepository.csv"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# One seed used everywhere so results are reproducible.
RANDOM_STATE = 42


def ensure_directories() -> None:
    """Create output folders if they do not exist yet."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
