"""Phase F: climate-region, air-quality, and map helpers.

The analysis in this module is descriptive. Climate zones are derived from
absolute latitude with fixed bands because the source data has no official
climate-region field. Regions are a transparent proxy based on the first
component of the IANA timezone string (for example ``Asia/Kolkata`` -> Asia).

Air-quality values in the source dataset have no documented concentration
units. They are therefore labelled as dataset units throughout, and no
regulatory or health-threshold claim is made.

Map outlines come from the public-domain Natural Earth 1:110m land layer stored
in ``assets/ne_110m_land.geojson``. The outlines are context only; all plotted
values come from the weather panel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WORLD_LAND_PATH = PROJECT_ROOT / "assets" / "ne_110m_land.geojson"

CLIMATE_ZONE_ORDER = ["Tropical", "Subtropical", "Temperate", "High-latitude"]
CLIMATE_ZONE_COLORS = {
    "Tropical": "#2A9D8F",
    "Subtropical": "#E9A23B",
    "Temperate": "#2A6F97",
    "High-latitude": "#6A4C93",
}
CLIMATE_ZONE_BOUNDS = [
    (0.0, 23.5, "Tropical"),
    (23.5, 35.0, "Subtropical"),
    (35.0, 50.0, "Temperate"),
    (50.0, 90.0, "High-latitude"),
]

REGION_ORDER = [
    "Africa",
    "Asia",
    "Europe",
    "Americas",
    "Pacific",
    "Indian Ocean",
    "Atlantic",
    "Australia",
]

POLLUTANTS = {
    "air_quality_PM2.5": "PM2.5",
    "air_quality_PM10": "PM10",
    "air_quality_Ozone": "Ozone",
    "air_quality_Nitrogen_dioxide": "Nitrogen dioxide",
    "air_quality_Sulphur_dioxide": "Sulphur dioxide",
    "air_quality_Carbon_Monoxide": "Carbon monoxide",
}

PM25_WEATHER_FEATURES = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm",
    "cloud",
    "visibility_km",
    "uv_index",
]


def climate_zone_from_latitude(latitude: float) -> str:
    """Map a latitude to one of four documented absolute-latitude bands."""
    absolute = abs(float(latitude))
    if absolute < 23.5:
        return "Tropical"
    if absolute < 35.0:
        return "Subtropical"
    if absolute < 50.0:
        return "Temperate"
    return "High-latitude"


def region_from_timezone(timezone: str) -> str:
    """Map an IANA timezone to a broad region for descriptive grouping.

    This is a proxy, not an official geographic classification. It is used only
    because the source data does not contain a continent or region column.
    """
    prefix = str(timezone).split("/")[0]
    aliases = {
        "America": "Americas",
        "Indian": "Indian Ocean",
        "Atlantic": "Atlantic",
        "Australia": "Australia",
        "Pacific": "Pacific",
    }
    return aliases.get(prefix, prefix)


def city_climate_summary(
    core: pd.DataFrame,
    min_days_per_month: int = 20,
    min_months: int = 12,
) -> pd.DataFrame:
    """One descriptive row per core city with climate and air-quality metrics.

    Seasonal amplitude is the difference between the warmest and coolest
    monthly mean temperature. It is computed only for cities with at least
    ``min_months`` usable calendar months. Values are not resampled or imputed.
    """
    required = {
        "country",
        "city",
        "date_local",
        "latitude",
        "longitude",
        "timezone",
        "temperature_celsius",
        "humidity",
        "wind_kph",
        "pressure_mb",
        "precip_mm",
        "cloud",
        "visibility_km",
        "uv_index",
        "air_quality_PM2.5",
    }
    missing = sorted(required.difference(core.columns))
    if missing:
        raise ValueError(f"Missing columns for city climate summary: {missing}")

    work = core.copy()
    work["month"] = work["date_local"].dt.to_period("M")

    city_month = (
        work.groupby(["country", "city", "month"], dropna=False)
        .agg(
            mean_temp_c=("temperature_celsius", "mean"),
            n_days=("temperature_celsius", "count"),
        )
        .reset_index()
    )
    city_month = city_month.loc[city_month["n_days"].ge(min_days_per_month)].copy()

    amplitude = (
        city_month.groupby(["country", "city"], dropna=False)
        .agg(
            n_months=("month", "nunique"),
            coolest_month_c=("mean_temp_c", "min"),
            warmest_month_c=("mean_temp_c", "max"),
        )
        .reset_index()
    )
    amplitude["seasonal_amplitude_c"] = (
        amplitude["warmest_month_c"] - amplitude["coolest_month_c"]
    )
    amplitude = amplitude.loc[amplitude["n_months"].ge(min_months), [
        "country",
        "city",
        "n_months",
        "coolest_month_c",
        "warmest_month_c",
        "seasonal_amplitude_c",
    ]]

    summary = (
        work.groupby(["country", "city"], dropna=False)
        .agg(
            n_days=("temperature_celsius", "count"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            timezone=("timezone", "first"),
            mean_temp_c=("temperature_celsius", "mean"),
            median_temp_c=("temperature_celsius", "median"),
            temp_p10_c=("temperature_celsius", lambda values: values.quantile(0.10)),
            temp_p90_c=("temperature_celsius", lambda values: values.quantile(0.90)),
            temp_range_c=("temperature_celsius", lambda values: values.max() - values.min()),
            mean_humidity_pct=("humidity", "mean"),
            mean_wind_kph=("wind_kph", "mean"),
            mean_pressure_mb=("pressure_mb", "mean"),
            mean_daily_precip_mm=("precip_mm", "mean"),
            mean_cloud_pct=("cloud", "mean"),
            mean_visibility_km=("visibility_km", "mean"),
            mean_uv_index=("uv_index", "mean"),
            median_pm25=("air_quality_PM2.5", "median"),
            mean_pm25=("air_quality_PM2.5", "mean"),
            p95_pm25=("air_quality_PM2.5", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )
    summary = summary.merge(amplitude, on=["country", "city"], how="left")
    summary["abs_latitude"] = summary["latitude"].abs()
    summary["hemisphere"] = np.where(
        summary["latitude"].ge(0), "Northern", "Southern"
    )
    summary["climate_zone"] = (
        summary["abs_latitude"].map(climate_zone_from_latitude).astype("string")
    )
    summary["region"] = (
        summary["timezone"].map(region_from_timezone).astype("string")
    )
    summary["climate_zone"] = pd.Categorical(
        summary["climate_zone"], categories=CLIMATE_ZONE_ORDER, ordered=True
    )
    return summary.sort_values(["climate_zone", "country", "city"]).reset_index(drop=True)


def zone_summary(city_summary: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight city aggregates for each derived climate zone."""
    summary = (
        city_summary.groupby("climate_zone", observed=True)
        .agg(
            n_cities=("city", "size"),
            city_days=("n_days", "sum"),
            mean_temp_c=("mean_temp_c", "mean"),
            median_city_temp_c=("mean_temp_c", "median"),
            seasonal_amplitude_median_c=("seasonal_amplitude_c", "median"),
            mean_humidity_pct=("mean_humidity_pct", "mean"),
            mean_uv_index=("mean_uv_index", "mean"),
            mean_wind_kph=("mean_wind_kph", "mean"),
            median_city_pm25=("median_pm25", "median"),
            pm25_q25=("median_pm25", lambda values: values.quantile(0.25)),
            pm25_q75=("median_pm25", lambda values: values.quantile(0.75)),
        )
        .reindex(CLIMATE_ZONE_ORDER)
        .reset_index()
    )
    return summary


def region_summary(city_summary: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight city aggregates for the timezone-region proxy."""
    summary = (
        city_summary.groupby("region", observed=True)
        .agg(
            n_cities=("city", "size"),
            city_days=("n_days", "sum"),
            mean_temp_c=("mean_temp_c", "mean"),
            seasonal_amplitude_median_c=("seasonal_amplitude_c", "median"),
            mean_humidity_pct=("mean_humidity_pct", "mean"),
            median_city_pm25=("median_pm25", "median"),
        )
        .reset_index()
    )
    return summary.sort_values("mean_temp_c", ascending=False).reset_index(drop=True)


def calendar_zone_climate(
    core: pd.DataFrame,
    city_summary: pd.DataFrame,
    min_days_per_month: int = 20,
) -> pd.DataFrame:
    """Calendar-month temperature climatology by derived climate zone."""
    work = core.copy()
    work["month_num"] = work["date_local"].dt.month
    work["month_name"] = work["date_local"].dt.strftime("%b")
    work = work.merge(
        city_summary[["country", "city", "climate_zone"]],
        on=["country", "city"],
        how="left",
        validate="many_to_one",
    )
    city_month = (
        work.groupby(
            ["country", "city", "climate_zone", "month_num", "month_name"],
            dropna=False,
            observed=True,
        )
        .agg(
            mean_temp_c=("temperature_celsius", "mean"),
            n_days=("temperature_celsius", "count"),
        )
        .reset_index()
    )
    city_month = city_month.loc[city_month["n_days"].ge(min_days_per_month)].copy()
    result = (
        city_month.groupby(["climate_zone", "month_num", "month_name"], observed=True)
        .agg(
            mean_temp_c=("mean_temp_c", "mean"),
            n_cities=("city", "size"),
        )
        .reset_index()
        .sort_values(["climate_zone", "month_num"])
        .reset_index(drop=True)
    )
    return result


def city_month_pm25(
    core: pd.DataFrame,
    min_days_per_month: int = 20,
) -> pd.DataFrame:
    """City-month median PM2.5 with a usable-day count."""
    work = core.copy()
    work["month"] = work["date_local"].dt.to_period("M")
    result = (
        work.groupby(["country", "city", "month"], dropna=False)
        .agg(
            pm25_median=("air_quality_PM2.5", "median"),
            n_days=("air_quality_PM2.5", "count"),
        )
        .reset_index()
    )
    result = result.loc[result["n_days"].ge(min_days_per_month)].copy()
    result["month_start"] = result["month"].dt.to_timestamp()
    return result.sort_values(["country", "city", "month"]).reset_index(drop=True)


def monthly_zone_pm25(
    core: pd.DataFrame,
    city_summary: pd.DataFrame,
    min_days_per_month: int = 20,
) -> pd.DataFrame:
    """Equal-weight city monthly PM2.5 medians by derived climate zone."""
    city_month = city_month_pm25(core, min_days_per_month=min_days_per_month)
    city_month = city_month.merge(
        city_summary[["country", "city", "climate_zone"]],
        on=["country", "city"],
        how="left",
        validate="many_to_one",
    )
    result = (
        city_month.groupby(["climate_zone", "month", "month_start"], observed=True)
        .agg(
            pm25_median=("pm25_median", "median"),
            pm25_q25=("pm25_median", lambda values: values.quantile(0.25)),
            pm25_q75=("pm25_median", lambda values: values.quantile(0.75)),
            n_cities=("city", "size"),
        )
        .reset_index()
        .sort_values(["climate_zone", "month"])
        .reset_index(drop=True)
    )
    return result


def pollutant_summary(core: pd.DataFrame) -> pd.DataFrame:
    """Distribution summary for each air-quality variable in dataset units."""
    rows = []
    for column, label in POLLUTANTS.items():
        if column not in core.columns:
            continue
        values = pd.to_numeric(core[column], errors="coerce")
        rows.append(
            {
                "pollutant": label,
                "units": "dataset units",
                "n": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def month_over_month_summary(
    city_month: pd.DataFrame,
    month_pairs: Sequence[tuple[str, str]] = (
        ("2024-09", "2024-08"),
        ("2025-09", "2025-08"),
        ("2026-09", "2026-08"),
    ),
) -> pd.DataFrame:
    """City-level PM2.5 month-over-month ratios for selected month pairs."""
    work = city_month.copy()
    work["month_str"] = work["month"].astype(str)
    rows = []
    for month, previous in month_pairs:
        current_values = (
            work.loc[work["month_str"].eq(month)]
            .set_index(["country", "city"])["pm25_median"]
        )
        previous_values = (
            work.loc[work["month_str"].eq(previous)]
            .set_index(["country", "city"])["pm25_median"]
        )
        ratio = (
            (current_values / previous_values)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        rows.append(
            {
                "month": month,
                "previous_month": previous,
                "n_cities": int(len(ratio)),
                "median_ratio": float(ratio.median()),
                "share_gt_1_5": float(ratio.gt(1.5).mean()),
                "share_gt_2": float(ratio.gt(2.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def within_city_pm25_correlations(
    core: pd.DataFrame,
    features: Sequence[str] = PM25_WEATHER_FEATURES,
    target: str = "air_quality_PM2.5",
    min_rows: int = 100,
) -> pd.DataFrame:
    """Spearman correlations between PM2.5 and weather within each city.

    Correlating the pooled panel would mix city-level climate differences with
    within-city day-to-day behaviour. Computing one correlation per city avoids
    that confounding, and the notebook reports the distribution across cities.
    """
    rows = []
    for (country, city), sub in core.groupby(["country", "city"], sort=False):
        for feature in features:
            pair = sub[[target, feature]].dropna()
            if len(pair) < min_rows:
                continue
            if pair[target].nunique() < 6 or pair[feature].nunique() < 6:
                continue
            result = spearmanr(pair[feature], pair[target])
            if not np.isfinite(result.statistic) or not np.isfinite(result.pvalue):
                continue
            rows.append(
                {
                    "country": country,
                    "city": city,
                    "variable": feature,
                    "n": int(len(pair)),
                    "rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No city had enough PM2.5/weather pairs for correlation.")
    return result


def correlation_summary(correlations: pd.DataFrame) -> pd.DataFrame:
    """Summarise city-level Spearman correlations for each weather variable."""
    work = correlations.copy()
    work["negative"] = work["rho"].lt(0)
    work["positive"] = work["rho"].gt(0)
    work["significant_negative"] = work["rho"].lt(0) & work["p_value"].lt(0.05)
    work["significant_positive"] = work["rho"].gt(0) & work["p_value"].lt(0.05)
    summary = (
        work.groupby("variable", sort=False)
        .agg(
            n_cities=("rho", "size"),
            median_r=("rho", "median"),
            q25_r=("rho", lambda values: values.quantile(0.25)),
            q75_r=("rho", lambda values: values.quantile(0.75)),
            share_negative=("negative", "mean"),
            share_positive=("positive", "mean"),
            share_significant_negative=("significant_negative", "mean"),
            share_significant_positive=("significant_positive", "mean"),
        )
        .reset_index()
    )
    return summary


def load_world_land(path: Path = WORLD_LAND_PATH) -> list[np.ndarray]:
    """Load exterior rings from the Natural Earth land GeoJSON.

    Antarctica is omitted because the plotted latitude extent is -60 to 85 and
    its ring crosses the antimeridian. The remaining polygons do not have that
    discontinuity at the 110m resolution.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rings: list[np.ndarray] = []
    for feature in data["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for polygon in polygons:
            ring = np.asarray(polygon[0], dtype=float)
            if ring[:, 1].max() < -60.0:
                continue
            rings.append(ring)
    return rings


def plot_world_land(
    ax,
    land: Sequence[np.ndarray],
    facecolor: str = "#E9ECEF",
    edgecolor: str = "#CED4DA",
    linewidth: float = 0.35,
) -> None:
    """Draw Natural Earth land polygons on an equirectangular axis."""
    collection = PolyCollection(
        list(land),
        closed=True,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=0,
    )
    ax.add_collection(collection)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    ax.set_xticklabels([f"{value:g}°" for value in np.arange(-180, 181, 60)])
    ax.set_yticklabels([f"{value:g}°" for value in np.arange(-60, 91, 30)])
    ax.grid(color="#DEE2E6", linewidth=0.6, alpha=0.9, zorder=1)
    ax.axhline(0, color="#ADB5BD", linewidth=0.7, linestyle="--", zorder=1)
    for latitude in (23.5, 35.0, 50.0, -23.5, -35.0, -50.0):
        ax.axhline(
            latitude,
            color="#6C757D",
            linewidth=0.55,
            linestyle=(0, (2, 2)),
            alpha=0.55,
            zorder=1,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_facecolor("#F8F9FA")
