"""Phase B: clean the Global Weather Repository data.

The raw CSV is never modified. This module applies only rules that were
justified by the Phase A profile, records how many rows/cells each rule
affected, and writes derived datasets under ``data/processed/``.

Design choices
--------------
* ``last_updated`` is the required local timestamp. ``timestamp_utc`` is
  derived from ``last_updated_epoch`` for globally aligned aggregation.
* The entity key is country + city, because city names repeat across countries.
* Impossible values are set to missing and flagged; real weather extremes are
  not deleted merely because they are statistically unusual.
* Unit-duplicate columns are removed only after the metric version is kept.
* The cleaned row table is not scaled. Scaling must be fitted on training data
  inside the modelling pipeline.

Usage:
    python src/clean_data.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RAW_CSV, REPORT_DIR  # noqa: E402

PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_ROWS_CSV = PROCESSED_DIR / "weather_clean_rows.csv.gz"
DAILY_CITY_CSV = PROCESSED_DIR / "weather_daily_city.csv.gz"
DAILY_GLOBAL_CSV = PROCESSED_DIR / "weather_daily_global_core.csv"

CLEANING_REPORT = REPORT_DIR / "cleaning_summary.txt"
OUTLIER_CSV = REPORT_DIR / "outlier_summary.csv"
OUTLIER_CITY_CSV = REPORT_DIR / "outlier_by_city_temperature.csv"
SCALING_CSV = REPORT_DIR / "scaling_summary.csv"
ALIAS_CSV = REPORT_DIR / "alias_mapping.csv"

# A group needs ~91% of the 880-day span to be treated as a complete city
# series. 50 rows is the minimum for a descriptive/partial series.
CORE_MIN_ROWS = 800
PARTIAL_MIN_ROWS = 50

# Physically defensible guards. They are deliberately wide: real weather
# extremes should survive; only clearly impossible values are masked.
TEMP_C_BOUNDS = (-90.0, 60.0)
PRESSURE_MB_BOUNDS = (650.0, 1100.0)
WIND_KPH_MAX = 500.0
WIND_EXTREME_KPH = 200.0
# The global core series has 183 cities. Require at least 150 (~82%) on a
# UTC date before treating that date as a usable global aggregate.
GLOBAL_MIN_CITIES = 150

# Country labels that are translations, truncations, or alternate spellings.
# The original label is retained in country_raw.
COUNTRY_ALIASES = {
    "Bélgica": "Belgium",
    "Congo": "Republic of the Congo",
    "Estonie": "Estonia",
    "Fiji Islands": "Fiji",
    "Inde": "India",
    "Jemen": "Yemen",
    "Komoren": "Comoros",
    "Kyrghyzstan": "Kyrgyzstan",
    "Letonia": "Latvia",
    "Macedonia": "North Macedonia",
    "Malásia": "Malaysia",
    "Marrocos": "Morocco",
    "Mexique": "Mexico",
    "Polônia": "Poland",
    "Saint-Vincent-et-les-Grenadines": "Saint Vincent and the Grenadines",
    "Saudi Arabien": "Saudi Arabia",
    "Seychelles Islands": "Seychelles",
    "Südkorea": "South Korea",
    "Swaziland": "Eswatini",
    "Turkménistan": "Turkmenistan",
    "USA United States of America": "United States of America",
    "Гватемала": "Guatemala",
    "Польша": "Poland",
    "Турция": "Turkey",
    "كولومبيا": "Colombia",
    "火鸡": "Turkey",
}

# Clear spelling/aliasing duplicates only. We do not "correct" unusual place
# names: many odd-looking names have coordinates inside the stated country and
# may be real small localities.
CITY_ALIASES = {
    "Addis Abeba": "Addis Ababa",
    "City Of San Marino": "San Marino",
    "Kuwait": "Kuwait City",
    "Nuku`Aloia": "Nuku'alofa",
    "Phnum Penh": "Phnom Penh",
    "Rangoon": "Yangon",
}

# Redundant unit columns. Keep Celsius, kph, mb, mm, and km.
REDUNDANT_UNIT_COLUMNS = [
    "temperature_fahrenheit",
    "feels_like_fahrenheit",
    "wind_mph",
    "gust_mph",
    "pressure_in",
    "precip_in",
    "visibility_miles",
]

# Pandas cannot read the hyphenated air-quality index names easily from code.
RENAME_COLUMNS = {
    "air_quality_us-epa-index": "air_quality_us_epa_index",
    "air_quality_gb-defra-index": "air_quality_gb_defra_index",
}

AQ_POLLUTANT_COLUMNS = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
]
AQ_INDEX_COLUMNS = [
    "air_quality_us_epa_index",
    "air_quality_gb_defra_index",
]

# Numeric columns kept after unit-column removal.
NUMERIC_CLEAN_COLUMNS = [
    "temperature_celsius",
    "feels_like_celsius",
    "wind_kph",
    "wind_degree",
    "gust_kph",
    "pressure_mb",
    "precip_mm",
    "humidity",
    "cloud",
    "visibility_km",
    "uv_index",
    "moon_illumination",
    *AQ_POLLUTANT_COLUMNS,
    *AQ_INDEX_COLUMNS,
]

# Features that are safe candidates for a train-only scaler demonstration.
# The target, coordinates, time fields, and ordinal AQ indices are excluded.
MODEL_FEATURE_COLUMNS = [
    "feels_like_celsius",
    "wind_kph",
    "gust_kph",
    "pressure_mb",
    "precip_mm",
    "humidity",
    "cloud",
    "visibility_km",
    "uv_index",
    *AQ_POLLUTANT_COLUMNS,
]

FLAG_COLUMNS = [
    "flag_temperature_impossible",
    "flag_pressure_impossible",
    "flag_wind_impossible",
    "flag_extreme_wind",
    "flag_air_quality_negative",
    "flag_visibility_zero",
]


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV without changing values."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {path}\n"
            "Download GlobalWeatherRepository.csv from Kaggle first; see "
            "data/README.md."
        )
    return pd.read_csv(path, low_memory=False)


def _mode(series: pd.Series) -> Any:
    """Most frequent non-null value, or NaN if the group is empty."""
    values = series.dropna().mode()
    return values.iloc[0] if not values.empty else np.nan


def _changed_rows(before: pd.Series, after: pd.Series) -> int:
    return int((before.astype(str) != after.astype(str)).sum())


def alias_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    """List every country/city label that was actually changed.

    Counts are rows in the cleaned file, so the table can be audited directly.
    """
    records = []
    for scope, raw_col, clean_col in [
        ("country", "country_raw", "country"),
        ("city", "city_raw", "city"),
    ]:
        counts = (
            cleaned.loc[
                cleaned[raw_col].astype(str) != cleaned[clean_col].astype(str),
                [raw_col, clean_col],
            ]
            .groupby([raw_col, clean_col], dropna=False)
            .size()
            .reset_index(name="rows")
        )
        for _, row in counts.iterrows():
            records.append(
                {
                    "scope": scope,
                    "raw_label": row[raw_col],
                    "clean_label": row[clean_col],
                    "rows": int(row["rows"]),
                }
            )
    return pd.DataFrame(records).sort_values(["scope", "rows"], ascending=[True, False])


def clean_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the documented row-level cleaning rules.

    Returns the cleaned DataFrame and a stats dictionary used by the report.
    """
    stats: dict[str, Any] = {
        "rows_input": int(len(raw)),
        "columns_input": int(raw.shape[1]),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "null_cells_input": int(raw.isna().sum().sum()),
    }

    required = {
        "country",
        "location_name",
        "last_updated",
        "last_updated_epoch",
        "temperature_celsius",
        "feels_like_celsius",
        "pressure_mb",
        "wind_kph",
        "gust_kph",
        "condition_text",
        "visibility_km",
        *AQ_POLLUTANT_COLUMNS,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    df = raw.copy()

    # ---------------------------------------------------------------
    # 1. Time
    # ---------------------------------------------------------------
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(
        df["last_updated_epoch"], unit="s", errors="coerce", utc=True
    )
    if df["last_updated"].isna().any() or df["timestamp_utc"].isna().any():
        raise ValueError("Time parsing produced missing timestamps; refusing to clean.")
    df["date_local"] = df["last_updated"].dt.date
    df["date_utc"] = df["timestamp_utc"].dt.date
    stats["time_min_local"] = str(df["last_updated"].min())
    stats["time_max_local"] = str(df["last_updated"].max())
    stats["time_min_utc"] = str(df["timestamp_utc"].min())
    stats["time_max_utc"] = str(df["timestamp_utc"].max())

    # ---------------------------------------------------------------
    # 2. Entity labels
    # ---------------------------------------------------------------
    df = df.rename(columns={"country": "country_raw", "location_name": "city_raw"})
    df["country"] = df["country_raw"].astype(object).str.strip()
    df["city"] = df["city_raw"].astype(object).str.strip()

    country_before = df["country"].copy()
    city_before = df["city"].copy()
    df["country"] = df["country"].replace(COUNTRY_ALIASES)
    df["city"] = df["city"].replace(CITY_ALIASES)

    stats["country_alias_rows"] = _changed_rows(country_before, df["country"])
    stats["city_alias_rows"] = _changed_rows(city_before, df["city"])
    stats["country_unique_raw"] = int(country_before.nunique())
    stats["country_unique_clean"] = int(df["country"].nunique())
    stats["city_unique_raw"] = int(city_before.nunique())
    stats["city_unique_clean"] = int(df["city"].nunique())
    stats["groups_raw"] = int(
        raw.groupby(["country", "location_name"], dropna=False).ngroups
    )
    stats["groups_clean"] = int(df.groupby(["country", "city"], dropna=False).ngroups)

    # ---------------------------------------------------------------
    # 3. Deduplicate the entity+timestamp key
    # ---------------------------------------------------------------
    key = ["country", "city", "last_updated"]
    duplicate_mask = df.duplicated(subset=key, keep="first")
    stats["duplicate_key_rows"] = int(duplicate_mask.sum())
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, key]
            .head(5)
            .astype(str)
            .to_dict(orient="records")
        )
        stats["duplicate_key_examples"] = examples
    df = df.loc[~duplicate_mask].reset_index(drop=True)
    stats["rows_after_dedupe"] = int(len(df))

    # ---------------------------------------------------------------
    # 4. Coverage tier (computed after alias merges and deduplication)
    # ---------------------------------------------------------------
    group_counts = (
        df.groupby(["country", "city"], dropna=False)
        .size()
        .rename("location_n_rows")
    )
    df = df.merge(
        group_counts,
        left_on=["country", "city"],
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    df["location_tier"] = np.select(
        [
            df["location_n_rows"] >= CORE_MIN_ROWS,
            df["location_n_rows"] >= PARTIAL_MIN_ROWS,
        ],
        ["core", "partial"],
        default="sparse",
    )
    stats["groups_after_dedupe"] = int(df.groupby(["country", "city"]).ngroups)
    stats["tier_rows"] = {
        str(k): int(v) for k, v in df.groupby("location_tier").size().items()
    }
    stats["tier_groups"] = {
        str(k): int(v)
        for k, v in df.groupby(["location_tier", "country", "city"])
        .size()
        .groupby(level=0)
        .size()
        .items()
    }

    # ---------------------------------------------------------------
    # 5. Impossible values -> missing, with flags
    # ---------------------------------------------------------------
    temp_bad = df["temperature_celsius"].lt(TEMP_C_BOUNDS[0]) | df[
        "temperature_celsius"
    ].gt(TEMP_C_BOUNDS[1])
    feels_bad = df["feels_like_celsius"].lt(TEMP_C_BOUNDS[0]) | df[
        "feels_like_celsius"
    ].gt(TEMP_C_BOUNDS[1])
    pressure_bad = df["pressure_mb"].lt(PRESSURE_MB_BOUNDS[0]) | df[
        "pressure_mb"
    ].gt(PRESSURE_MB_BOUNDS[1])
    wind_impossible = df["wind_kph"].gt(WIND_KPH_MAX) | df["gust_kph"].gt(
        WIND_KPH_MAX
    )
    extreme_wind = df["wind_kph"].gt(WIND_EXTREME_KPH) | df["gust_kph"].gt(
        WIND_EXTREME_KPH
    )
    aq_negative = df[AQ_POLLUTANT_COLUMNS].lt(0)
    aq_negative_rows = aq_negative.any(axis=1)
    aq_negative_cells = int(aq_negative.sum().sum())

    stats["temperature_impossible_rows"] = int(temp_bad.sum())
    stats["feels_like_impossible_rows"] = int(feels_bad.sum())
    stats["pressure_impossible_rows"] = int(pressure_bad.sum())
    stats["wind_impossible_rows"] = int(wind_impossible.sum())
    stats["extreme_wind_rows"] = int(extreme_wind.sum())
    stats["air_quality_negative_rows"] = int(aq_negative_rows.sum())
    stats["air_quality_negative_cells"] = aq_negative_cells

    # Mask both unit representations before dropping the redundant ones.
    df.loc[temp_bad, ["temperature_celsius", "temperature_fahrenheit"]] = np.nan
    df.loc[feels_bad, ["feels_like_celsius", "feels_like_fahrenheit"]] = np.nan
    df.loc[pressure_bad, ["pressure_mb", "pressure_in"]] = np.nan
    df.loc[
        wind_impossible,
        ["wind_kph", "wind_mph", "gust_kph", "gust_mph"],
    ] = np.nan
    df[AQ_POLLUTANT_COLUMNS] = df[AQ_POLLUTANT_COLUMNS].mask(aq_negative)

    # Keep flags so masked values are auditable.
    df["flag_temperature_impossible"] = (temp_bad | feels_bad).fillna(False)
    df["flag_pressure_impossible"] = pressure_bad.fillna(False)
    df["flag_wind_impossible"] = wind_impossible.fillna(False)
    df["flag_extreme_wind"] = extreme_wind.fillna(False)
    df["flag_air_quality_negative"] = aq_negative_rows.fillna(False)
    df["flag_visibility_zero"] = df["visibility_km"].eq(0).fillna(False)
    df["quality_flag_count"] = df[FLAG_COLUMNS].sum(axis=1).astype(int)

    stats["visibility_zero_rows"] = int(df["flag_visibility_zero"].sum())

    # ---------------------------------------------------------------
    # 6. Categorical normalization and schema cleanup
    # ---------------------------------------------------------------
    condition_before = int(df["condition_text"].nunique())
    df["condition_text"] = (
        df["condition_text"].astype(object).str.strip().str.lower().str.capitalize()
    )
    stats["condition_unique_before"] = condition_before
    stats["condition_unique_after"] = int(df["condition_text"].nunique())
    df["wind_direction"] = df["wind_direction"].astype(object).str.strip().str.upper()
    df["moon_phase"] = df["moon_phase"].astype(object).str.strip()
    df["timezone"] = df["timezone"].astype(object).str.strip()

    df = df.rename(columns=RENAME_COLUMNS)
    dropped = [c for c in REDUNDANT_UNIT_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    stats["dropped_unit_columns"] = dropped

    # Stable column order: identifiers, time, quality metadata, then the rest.
    first = [
        "country",
        "city",
        "country_raw",
        "city_raw",
        "latitude",
        "longitude",
        "timezone",
        "last_updated",
        "timestamp_utc",
        "date_local",
        "date_utc",
        "location_tier",
        "location_n_rows",
        "quality_flag_count",
        *FLAG_COLUMNS,
    ]
    rest = [c for c in df.columns if c not in first]
    df = df[first + rest].reset_index(drop=True)

    stats["rows_output"] = int(len(df))
    stats["columns_output"] = int(df.shape[1])
    stats["null_cells_output"] = int(df.isna().sum().sum())
    stats["duplicate_key_rows_remaining"] = int(
        df.duplicated(subset=["country", "city", "last_updated"], keep=False).sum()
    )
    return df, stats


def build_daily_city(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per country + city + local date.

    Numeric weather values are averaged (or median for ordinal AQ indices).
    Categorical fields use the mode. Wind direction uses a circular mean.
    Two readings on the same local date are therefore collapsed explicitly.
    """
    work = rows.copy()
    work["_wind_sin"] = np.sin(np.deg2rad(work["wind_degree"]))
    work["_wind_cos"] = np.cos(np.deg2rad(work["wind_degree"]))

    mean_cols = [
        "temperature_celsius",
        "feels_like_celsius",
        "wind_kph",
        "gust_kph",
        "pressure_mb",
        "precip_mm",
        "humidity",
        "cloud",
        "visibility_km",
        "uv_index",
        "moon_illumination",
        *AQ_POLLUTANT_COLUMNS,
    ]
    agg_spec: dict[str, pd.NamedAgg] = {
        col: pd.NamedAgg(column=col, aggfunc="mean") for col in mean_cols
    }
    agg_spec.update(
        {
            "latitude": pd.NamedAgg(column="latitude", aggfunc="median"),
            "longitude": pd.NamedAgg(column="longitude", aggfunc="median"),
            "timezone": pd.NamedAgg(column="timezone", aggfunc="first"),
            "condition_text": pd.NamedAgg(column="condition_text", aggfunc=_mode),
            "wind_direction": pd.NamedAgg(column="wind_direction", aggfunc=_mode),
            "moon_phase": pd.NamedAgg(column="moon_phase", aggfunc=_mode),
            "last_updated": pd.NamedAgg(column="last_updated", aggfunc="max"),
            "timestamp_utc": pd.NamedAgg(column="timestamp_utc", aggfunc="max"),
            "country_raw": pd.NamedAgg(column="country_raw", aggfunc="first"),
            "city_raw": pd.NamedAgg(column="city_raw", aggfunc="first"),
            "location_tier": pd.NamedAgg(column="location_tier", aggfunc="first"),
            "location_n_rows": pd.NamedAgg(
                column="location_n_rows", aggfunc="first"
            ),
            "n_readings": pd.NamedAgg(column="last_updated", aggfunc="size"),
            "n_flagged_readings": pd.NamedAgg(
                column="quality_flag_count", aggfunc=lambda s: int((s > 0).sum())
            ),
            "_wind_sin": pd.NamedAgg(column="_wind_sin", aggfunc="mean"),
            "_wind_cos": pd.NamedAgg(column="_wind_cos", aggfunc="mean"),
        }
    )
    for col in AQ_INDEX_COLUMNS:
        agg_spec[col] = pd.NamedAgg(column=col, aggfunc="median")

    daily = (
        work.groupby(["country", "city", "date_local"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )
    daily["wind_degree"] = (
        np.degrees(np.arctan2(daily["_wind_sin"], daily["_wind_cos"])) % 360
    ).round(1)
    daily = daily.drop(columns=["_wind_sin", "_wind_cos"])

    first = [
        "country",
        "city",
        "country_raw",
        "city_raw",
        "date_local",
        "last_updated",
        "timestamp_utc",
        "location_tier",
        "location_n_rows",
        "n_readings",
        "n_flagged_readings",
        "latitude",
        "longitude",
        "timezone",
    ]
    rest = [c for c in daily.columns if c not in first]
    return daily[first + rest]


def build_daily_global_core(rows: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted global daily series using core locations and UTC dates.

    First aggregate each city to one value per UTC date, then average cities.
    This prevents a city with two readings on a date from being double-counted.
    """
    core = rows.loc[rows["location_tier"].eq("core")].copy()
    mean_cols = [
        "temperature_celsius",
        "feels_like_celsius",
        "wind_kph",
        "gust_kph",
        "pressure_mb",
        "precip_mm",
        "humidity",
        "cloud",
        "visibility_km",
        "uv_index",
        "moon_illumination",
        *AQ_POLLUTANT_COLUMNS,
    ]
    city_day = (
        core.groupby(["country", "city", "date_utc"], dropna=False)
        .agg(**{col: pd.NamedAgg(column=col, aggfunc="mean") for col in mean_cols})
        .reset_index()
    )
    global_daily = (
        city_day.groupby("date_utc", dropna=False)
        .agg(
            **{
                col: pd.NamedAgg(column=col, aggfunc="mean")
                for col in mean_cols
            },
            n_cities=pd.NamedAgg(column="city", aggfunc="size"),
        )
        .reset_index()
    )
    global_daily["coverage_ok"] = global_daily["n_cities"] >= GLOBAL_MIN_CITIES
    return global_daily


def outlier_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Global IQR / z-score counts for diagnostics, not deletion rules."""
    records = []
    for col in NUMERIC_CLEAN_COLUMNS:
        s = rows[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else pd.Series(0, index=s.index)
        records.append(
            {
                "column": col,
                "n": int(s.size),
                "q1": float(q1),
                "q3": float(q3),
                "iqr_low": float(low),
                "iqr_high": float(high),
                "iqr_outliers": int(((s < low) | (s > high)).sum()),
                "z3_outliers": int((z.abs() > 3).sum()),
                "min": float(s.min()),
                "max": float(s.max()),
                "note": "Keep; diagnostic only",
            }
        )
    return pd.DataFrame(records)


def outlier_by_city(rows: pd.DataFrame) -> pd.DataFrame:
    """Within-city IQR outliers for core locations, using the cleaned rows."""
    core = rows.loc[rows["location_tier"].eq("core")]
    records = []
    for (country, city), sub in core.groupby(["country", "city"]):
        rec = {"country": country, "city": city, "n": int(len(sub))}
        for col, short in [
            ("temperature_celsius", "temperature"),
            ("wind_kph", "wind"),
            ("pressure_mb", "pressure"),
        ]:
            s = sub[col].dropna()
            if s.empty:
                rec[f"{short}_outliers"] = 0
                rec[f"{short}_share"] = np.nan
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
            rec[f"{short}_outliers"] = int(mask.sum())
            rec[f"{short}_share"] = float(mask.mean())
        records.append(rec)
    return pd.DataFrame(records).sort_values(
        ["temperature_outliers", "wind_outliers"], ascending=False
    )


def scaling_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Scale statistics; the saved cleaned data itself is not scaled."""
    records = []
    for col in NUMERIC_CLEAN_COLUMNS:
        s = rows[col].dropna()
        if s.empty:
            continue
        records.append(
            {
                "column": col,
                "n": int(s.size),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.size > 1 else np.nan,
                "min": float(s.min()),
                "max": float(s.max()),
                "range": float(s.max() - s.min()),
                "recommended_use": (
                    "do_not_scale_target"
                    if col == "temperature_celsius"
                    else "circular_encode"
                    if col == "wind_degree"
                    else "treat_as_ordinal"
                    if col in AQ_INDEX_COLUMNS
                    else "standardise_if_linear_or_distance_model"
                ),
            }
        )
    return pd.DataFrame(records)


def _fmt_dict(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def write_cleaning_report(
    stats: dict[str, Any],
    outliers: pd.DataFrame,
    by_city: pd.DataFrame,
    scaling: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    """Write a human-readable audit trail with actual counts and rules."""
    temp_outlier_total = int(by_city["temperature_outliers"].sum())
    temp_n_total = int(by_city["n"].sum())
    temp_cities = int((by_city["temperature_outliers"] > 0).sum())
    lines = [
        "Phase B cleaning summary",
        "======================",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Input and output",
        "----------------",
        f"Rows in: {stats['rows_input']:,} | columns in: {stats['columns_input']}",
        f"Rows out: {stats['rows_output']:,} | columns out: {stats['columns_output']}",
        f"Exact duplicate rows: {stats['exact_duplicate_rows']}",
        f"Duplicate country+city+timestamp rows removed: {stats['duplicate_key_rows']}",
        f"Null cells in raw file: {stats['null_cells_input']:,}",
        f"Null cells in cleaned file: {stats['null_cells_output']:,}",
        f"Time range (local): {stats['time_min_local']} -> {stats['time_max_local']}",
        f"Time range (UTC):   {stats['time_min_utc']} -> {stats['time_max_utc']}",
        "",
        "Entity and coverage rules",
        "-------------------------",
        f"Country labels: {stats['country_unique_raw']} raw -> {stats['country_unique_clean']} cleaned",
        f"City names: {stats['city_unique_raw']} raw -> {stats['city_unique_clean']} cleaned",
        f"Country+city groups: {stats['groups_raw']} raw -> {stats['groups_clean']} cleaned"
        f" -> {stats['groups_after_dedupe']} after dedupe",
        f"Country alias rows changed: {stats['country_alias_rows']:,}",
        f"City alias rows changed: {stats['city_alias_rows']:,}",
        f"Rows after dedupe: {stats['rows_after_dedupe']:,}",
        f"Location tiers (rows): {_fmt_dict(stats['tier_rows'])}",
        f"Location tiers (groups): {_fmt_dict(stats['tier_groups'])}",
        f"Duplicate local city-day keys collapsed: "
        f"{stats['duplicate_city_day_keys']:,} keys "
        f"({stats['duplicate_city_day_extra_rows']:,} extra rows)",
        f"Daily city rows: {stats['daily_city_rows']:,}",
        f"Global core UTC dates: {stats['daily_global_rows']:,}; "
        f"usable (>= {GLOBAL_MIN_CITIES} cities): "
        f"{stats['global_coverage_ok_dates']:,}; "
        f"cities/date min/median/max: "
        f"{stats['global_min_cities']}/"
        f"{stats['global_median_cities']:.1f}/"
        f"{stats['global_max_cities']}",
        "",
        "Impossible values set to missing and flagged",
        "-------------------------------------------",
        f"Temperature/feels-like impossible rows: "
        f"{stats['temperature_impossible_rows']} / {stats['feels_like_impossible_rows']}",
        f"Pressure outside {PRESSURE_MB_BOUNDS} mb: {stats['pressure_impossible_rows']}",
        f"Wind/gust above {WIND_KPH_MAX:g} kph: {stats['wind_impossible_rows']}",
        f"Extreme wind/gust above {WIND_EXTREME_KPH:g} kph (kept if < "
        f"{WIND_KPH_MAX:g}): {stats['extreme_wind_rows']} rows",
        f"Negative air-quality rows/cells: "
        f"{stats['air_quality_negative_rows']} / {stats['air_quality_negative_cells']}",
        f"Zero visibility rows (kept; mostly fog): {stats['visibility_zero_rows']:,}",
        "",
        "Schema cleanup",
        "--------------",
        f"Dropped redundant unit columns: {', '.join(stats['dropped_unit_columns'])}",
        f"Condition labels: {stats['condition_unique_before']} exact -> "
        f"{stats['condition_unique_after']} after lowercase+capitalise",
        "Astronomy fields are retained as raw metadata and are not used as model features.",
        "",
        "Outlier analysis (diagnostic only; no weather extremes deleted)",
        "---------------------------------------------------------------",
        f"Global IQR outliers by variable (see {OUTLIER_CSV.name}):",
        outliers[
            ["column", "n", "iqr_outliers", "z3_outliers", "min", "max"]
        ].to_string(index=False),
        "",
        f"Within-city temperature IQR outliers (core locations only): "
        f"{temp_outlier_total:,} / {temp_n_total:,} rows "
        f"({temp_outlier_total / temp_n_total:.2%}); "
        f"{temp_cities} of {len(by_city)} core cities have at least one.",
        f"Within-city wind IQR outliers: {int(by_city['wind_outliers'].sum()):,} "
        f"({by_city['wind_outliers'].sum() / by_city['n'].sum():.2%}); "
        f"pressure: {int(by_city['pressure_outliers'].sum()):,} "
        f"({by_city['pressure_outliers'].sum() / by_city['n'].sum():.2%}).",
        "These are kept: they may be real heatwaves/cold snaps and are useful "
        "signals for anomaly detection.",
        "IQR is not meaningful for zero-inflated precipitation or capped "
        "visibility; those counts are reported for transparency only.",
        "",
        "Scaling decision",
        "----------------",
        "The cleaned output is deliberately unscaled. Scale inside the model "
        "pipeline, fitting on the training period only.",
        "Use StandardScaler (or RobustScaler if retained outliers dominate) for "
        "linear, distance-based, and neural models. Tree models do not need "
        "scaling. Keep the target in degrees Celsius for interpretable MAE/RMSE.",
        f"Scale statistics: {SCALING_CSV.name}",
        "",
        "Output files",
        "------------",
        f"Row-level cleaned data: {paths['clean_rows']}",
        f"City-day data:          {paths['daily_city']}",
        f"Global daily core data: {paths['daily_global']}",
        f"Global outliers:        {paths['outliers']}",
        f"City outliers:          {paths['outlier_city']}",
        f"Scaling statistics:     {paths['scaling']}",
        f"Alias mapping:          {paths['aliases']}",
        "",
        "This report contains actual counts from the executed run, not estimates.",
    ]
    CLEANING_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    cleaned: pd.DataFrame,
    daily_city: pd.DataFrame,
    daily_global: pd.DataFrame,
    outliers: pd.DataFrame,
    by_city: pd.DataFrame,
    scaling: pd.DataFrame,
) -> dict[str, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(CLEAN_ROWS_CSV, index=False)
    daily_city.to_csv(DAILY_CITY_CSV, index=False)
    daily_global.to_csv(DAILY_GLOBAL_CSV, index=False)
    outliers.to_csv(OUTLIER_CSV, index=False)
    by_city.to_csv(OUTLIER_CITY_CSV, index=False)
    scaling.to_csv(SCALING_CSV, index=False)
    alias_table(cleaned).to_csv(ALIAS_CSV, index=False)
    return {
        "clean_rows": CLEAN_ROWS_CSV,
        "daily_city": DAILY_CITY_CSV,
        "daily_global": DAILY_GLOBAL_CSV,
        "outliers": OUTLIER_CSV,
        "outlier_city": OUTLIER_CITY_CSV,
        "scaling": SCALING_CSV,
        "aliases": ALIAS_CSV,
    }


def main() -> int:
    raw = load_raw()
    cleaned, stats = clean_rows(raw)
    daily_city = build_daily_city(cleaned)
    daily_global = build_daily_global_core(cleaned)
    outliers = outlier_summary(cleaned)
    by_city = outlier_by_city(cleaned)
    scaling = scaling_summary(cleaned)

    duplicate_city_day = (
        cleaned.groupby(["country", "city", "date_local"], dropna=False)
        .size()
        .rename("n")
    )
    stats["duplicate_city_day_keys"] = int((duplicate_city_day > 1).sum())
    stats["duplicate_city_day_extra_rows"] = int(
        (duplicate_city_day - 1).clip(lower=0).sum()
    )
    stats["daily_city_rows"] = int(len(daily_city))
    stats["daily_global_rows"] = int(len(daily_global))
    stats["global_coverage_ok_dates"] = int(daily_global["coverage_ok"].sum())
    stats["global_min_cities"] = int(daily_global["n_cities"].min())
    stats["global_median_cities"] = float(daily_global["n_cities"].median())
    stats["global_max_cities"] = int(daily_global["n_cities"].max())

    paths = save_outputs(cleaned, daily_city, daily_global, outliers, by_city, scaling)
    write_cleaning_report(stats, outliers, by_city, scaling, paths)

    print("Phase B cleaning complete.")
    print(f"Rows: {stats['rows_input']:,} -> {stats['rows_output']:,}")
    print(
        "Rows removed: "
        f"{stats['rows_input'] - stats['rows_output']:,} "
        f"({stats['duplicate_key_rows']} duplicate key rows)"
    )
    print(f"Daily city rows: {stats['daily_city_rows']:,}")
    print(f"Daily global core rows: {stats['daily_global_rows']:,}")
    print(f"Cleaning report: {CLEANING_REPORT}")
    print(f"Outliers: {OUTLIER_CSV}")
    print(f"City temperature outliers: {OUTLIER_CITY_CSV}")
    print(f"Scaling summary: {SCALING_CSV}")
    print(f"Cleaned rows: {CLEAN_ROWS_CSV}")
    print(f"City-day data: {DAILY_CITY_CSV}")
    print(f"Global core data: {DAILY_GLOBAL_CSV}")
    print(f"Alias mapping: {ALIAS_CSV}")
    print(
        "Tier rows:", stats["tier_rows"],
        "| Tier groups:", stats["tier_groups"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
