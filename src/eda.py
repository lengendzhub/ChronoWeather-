"""Phase C: reusable loading and summary helpers for exploratory analysis.

The EDA notebook should not re-implement the same aggregation logic in every
cell. This module keeps the definitions of the analysis subsets and the
descriptive calculations in one place.

No modelling, forecasting, or imputation is performed here. The only global
series used for headline trends is the equal-weighted core-location series
with ``coverage_ok == True``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIG_DIR  # noqa: E402

CITY_DAILY_CSV = PROJECT_ROOT / "data" / "processed" / "weather_daily_city.csv.gz"
GLOBAL_DAILY_CSV = PROJECT_ROOT / "data" / "processed" / "weather_daily_global_core.csv"

CORE_TIER = "core"
TIER_ORDER = ["core", "partial", "sparse"]
TIER_COLORS = {
    "core": "#2A6F97",
    "partial": "#E9A23B",
    "sparse": "#B44C5E",
}

# Variables used in the correlation analysis. Long raw names are mapped to
# compact display labels only in the plotting layer.
CORRELATION_COLUMNS = [
    "temperature_celsius",
    "humidity",
    "cloud",
    "wind_kph",
    "pressure_mb",
    "precip_mm",
    "visibility_km",
    "uv_index",
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
]

SHORT_LABELS = {
    "temperature_celsius": "Temp",
    "feels_like_celsius": "Feels-like",
    "humidity": "Humidity",
    "cloud": "Cloud",
    "wind_kph": "Wind",
    "gust_kph": "Gust",
    "pressure_mb": "Pressure",
    "precip_mm": "Precip",
    "visibility_km": "Visibility",
    "uv_index": "UV",
    "moon_illumination": "Moon illum.",
    "air_quality_Carbon_Monoxide": "CO",
    "air_quality_Ozone": "O$_3$",
    "air_quality_Nitrogen_dioxide": "NO$_2$",
    "air_quality_Sulphur_dioxide": "SO$_2$",
    "air_quality_PM2.5": "PM$_{2.5}$",
    "air_quality_PM10": "PM$_{10}$",
    "air_quality_us_epa_index": "US EPA",
    "air_quality_gb_defra_index": "GB DEFRA",
}


def load_eda_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two processed Phase C analysis tables.

    Returns
    -------
    city_daily
        One row per country + city + local date.
    global_daily
        One row per UTC date, equal-weighted across core cities.
    """
    city_daily = pd.read_csv(
        CITY_DAILY_CSV,
        low_memory=False,
        parse_dates=["date_local", "last_updated", "timestamp_utc"],
    )
    global_daily = pd.read_csv(GLOBAL_DAILY_CSV, parse_dates=["date_utc"])

    required_city = {
        "country",
        "city",
        "date_local",
        "location_tier",
        "temperature_celsius",
        "humidity",
        "wind_kph",
        "pressure_mb",
        "condition_text",
        "air_quality_PM2.5",
    }
    required_global = {
        "date_utc",
        "temperature_celsius",
        "n_cities",
        "coverage_ok",
    }
    missing_city = sorted(required_city.difference(city_daily.columns))
    missing_global = sorted(required_global.difference(global_daily.columns))
    if missing_city or missing_global:
        raise ValueError(
            f"Missing columns in processed data: city={missing_city}, "
            f"global={missing_global}"
        )
    return city_daily, global_daily


def core_city_days(city_daily: pd.DataFrame) -> pd.DataFrame:
    """Return only the 183 high-coverage city series."""
    return city_daily.loc[city_daily["location_tier"].eq(CORE_TIER)].copy()


def tier_summary(city_daily: pd.DataFrame) -> pd.DataFrame:
    """Rows and unique city groups by coverage tier."""
    group_sizes = (
        city_daily.groupby(["location_tier", "country", "city"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    summary = (
        group_sizes.groupby("location_tier")
        .agg(rows=("rows", "sum"), groups=("rows", "size"))
        .reindex(TIER_ORDER)
        .reset_index()
    )
    return summary


def monthly_global(global_daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly means of the usable equal-weighted global core series."""
    usable = global_daily.loc[global_daily["coverage_ok"]].copy()
    usable["month"] = usable["date_utc"].dt.to_period("M")
    monthly = (
        usable.groupby("month")
        .agg(
            n_days=("temperature_celsius", "size"),
            mean_temp=("temperature_celsius", "mean"),
            min_temp=("temperature_celsius", "min"),
            max_temp=("temperature_celsius", "max"),
            mean_cities=("n_cities", "mean"),
        )
        .reset_index()
    )
    monthly["month_start"] = monthly["month"].dt.to_timestamp()
    monthly["month_label"] = monthly["month"].dt.strftime("%Y-%m")
    monthly["month_num"] = monthly["month"].dt.month
    monthly["month_name"] = monthly["month"].dt.strftime("%b")
    return monthly


def city_temperature_summary(city_daily: pd.DataFrame) -> pd.DataFrame:
    """Per-core-city descriptive temperature and location summary."""
    core = core_city_days(city_daily)
    summary = (
        core.groupby(["country", "city"], dropna=False)
        .agg(
            n_days=("temperature_celsius", "size"),
            mean_temp=("temperature_celsius", "mean"),
            median_temp=("temperature_celsius", "median"),
            min_temp=("temperature_celsius", "min"),
            max_temp=("temperature_celsius", "max"),
            lat=("latitude", "median"),
            lon=("longitude", "median"),
        )
        .reset_index()
    )
    summary["temp_range"] = summary["max_temp"] - summary["min_temp"]
    return summary


def monthly_city_temperature(
    city_daily: pd.DataFrame, min_days_per_month: int = 20
) -> pd.DataFrame:
    """Monthly mean temperature for each core city, with month counts."""
    core = core_city_days(city_daily).copy()
    core["month"] = core["date_local"].dt.to_period("M")
    monthly = (
        core.groupby(["country", "city", "month"], dropna=False)
        .agg(
            mean_temp=("temperature_celsius", "mean"),
            n_days=("temperature_celsius", "size"),
        )
        .reset_index()
    )
    return monthly.loc[monthly["n_days"] >= min_days_per_month].copy()


def seasonal_amplitude(
    city_daily: pd.DataFrame,
    min_days_per_month: int = 20,
    min_months: int = 12,
) -> pd.DataFrame:
    """Difference between each city's warmest and coolest monthly means."""
    monthly = monthly_city_temperature(city_daily, min_days_per_month)
    amplitude = (
        monthly.groupby(["country", "city"], dropna=False)
        .agg(
            n_months=("month", "nunique"),
            coolest_month=("mean_temp", "min"),
            warmest_month=("mean_temp", "max"),
        )
        .reset_index()
    )
    amplitude["amplitude"] = amplitude["warmest_month"] - amplitude["coolest_month"]
    amplitude = amplitude.loc[amplitude["n_months"] >= min_months].copy()

    city_summary = city_temperature_summary(city_daily)[
        ["country", "city", "lat", "lon", "mean_temp"]
    ]
    return amplitude.merge(city_summary, on=["country", "city"], how="left")


def within_city_spearman(
    city_daily: pd.DataFrame,
    columns: Sequence[str] = CORRELATION_COLUMNS,
    min_rows: int = 100,
) -> pd.DataFrame:
    """Median Spearman correlation across core-city time series.

    Pooling all cities can create misleading cross-city associations (for
    example, altitude and latitude can drive a pooled temperature-pressure
    correlation). The median within-city matrix is a more defensible
    descriptive view, while still not being a causal estimate.
    """
    core = core_city_days(city_daily)
    matrices: list[pd.DataFrame] = []
    for _, sub in core.groupby(["country", "city"], dropna=False):
        if len(sub) < min_rows:
            continue
        matrices.append(sub[list(columns)].corr(method="spearman"))

    if not matrices:
        raise ValueError("No city had enough rows for a correlation matrix.")

    stacked = pd.concat([matrix.stack() for matrix in matrices], axis=1)
    median = stacked.median(axis=1).unstack()
    return median.reindex(index=columns, columns=columns)


def set_plot_style() -> None:
    """One consistent visual style for all Phase C figures."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="deep",
        font_scale=1.0,
    )
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 170,
            "savefig.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.autolayout": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
        }
    )


def save_figure(fig: plt.Figure, filename: str, dpi: int = 170) -> Path:
    """Save a figure under reports/figures and return its path."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path
