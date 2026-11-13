"""Resolve the raw Global Weather Repository CSV.

Two supported routes, in order of convenience:

1. **KaggleHub (automatic).** If ``data/GlobalWeatherRepository.csv`` is
   missing, download the public dataset with KaggleHub, locate the CSV inside
   the cached dataset folder, and copy it into ``data/``.
2. **Manual download.** Place the CSV at the expected path yourself. The
   function returns it immediately without touching the network.

The raw file is never modified. ``data/`` is git-ignored, so the downloaded
CSV never ends up in the repository.

Usage
-----
    python src/load_data.py
    python src/load_data.py --force

Or from Python / a notebook:

    from src.load_data import ensure_raw_csv
    csv_path = ensure_raw_csv()
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import KAGGLE_DATASET, KAGGLE_FILE, RAW_CSV  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 fingerprint of a file, read in chunks.

    Recording the hash makes it possible to tell exactly which data version a
    profile or model was trained on.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_csv(dataset_dir: Path, preferred_name: str) -> Path:
    """Find the dataset CSV without silently guessing between candidates."""
    exact = [
        p
        for p in dataset_dir.rglob("*")
        if p.is_file() and p.name.lower() == preferred_name.lower()
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        # Prefer the shallowest copy (usually the canonical one at the root).
        exact.sort(key=lambda p: (len(p.parts), str(p).lower()))
        if len({p.name.lower() for p in exact}) == 1:
            return exact[0]
        raise RuntimeError(
            "Multiple files match "
            f"{preferred_name!r} in {dataset_dir}:\n"
            + "\n".join(f"  - {p}" for p in exact)
        )

    csvs = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"]
    if len(csvs) == 1:
        print(f"Warning: {preferred_name!r} not found; using the only CSV: {csvs[0].name}")
        return csvs[0]
    if not csvs:
        raise RuntimeError(
            f"No CSV files found inside the downloaded dataset: {dataset_dir}\n"
            "This usually means Kaggle returned an empty archive because the "
            "dataset requires sign-in/subscription, or the download was interrupted.\n"
            "Fix it by signing in and downloading the CSV manually to\n"
            f"  {RAW_CSV}\n"
            "or by running `python src/load_data.py --force` after authenticating "
            "KaggleHub with `kagglehub.login()`."
        )

    raise RuntimeError(
        "Could not identify the dataset CSV. Found:\n"
        + "\n".join(f"  - {p.relative_to(dataset_dir)}" for p in csvs)
        + f"\nExpected a file named {preferred_name!r}."
    )


def ensure_raw_csv(force: bool = False) -> Path:
    """Return the raw CSV path, downloading it via KaggleHub if necessary.

    Parameters
    ----------
    force:
        Replace an existing ``data/GlobalWeatherRepository.csv`` with a fresh
        copy from the KaggleHub cache.
    """
    if RAW_CSV.exists() and RAW_CSV.stat().st_size > 0 and not force:
        return RAW_CSV

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "KaggleHub is not installed. Run:\n"
            "  pip install -r requirements.txt\n"
            "or download the CSV manually and place it at:\n"
            f"  {RAW_CSV}"
        ) from exc

    print(f"Downloading '{KAGGLE_DATASET}' via KaggleHub (cached after the first run)...")
    try:
        # force=True also ignores a stale/empty KaggleHub cache entry.
        cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET, force_download=force))
    except Exception as exc:
        raise RuntimeError(
            "KaggleHub could not download the dataset.\n"
            "Options:\n"
            "  1. Check your internet connection and retry.\n"
            "  2. If Kaggle asks for authentication, run `python -c \"import kagglehub; kagglehub.login()\"`\n"
            "     and try again. Credentials are stored outside this repository.\n"
            "  3. Download the CSV manually from the Kaggle page and place it at:\n"
            f"     {RAW_CSV}\n"
            f"Original error: {exc}"
        ) from exc

    source = _find_csv(cache_dir, KAGGLE_FILE)
    if source.stat().st_size == 0:
        raise RuntimeError(f"Downloaded CSV is empty: {source}")

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, RAW_CSV)

    size_mb = RAW_CSV.stat().st_size / 1024**2
    print(f"Source cache: {source}")
    print(f"Copied to:    {RAW_CSV} ({size_mb:.2f} MB)")
    return RAW_CSV


def main(argv: list[str]) -> int:
    force = "--force" in argv
    try:
        path = ensure_raw_csv(force=force)
    except RuntimeError as exc:
        print(exc)
        return 1
    print(f"CSV path: {path}")
    print(f"SHA-256:  {sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
