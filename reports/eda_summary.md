# Phase C EDA summary

Generated from the executed `notebooks/02_eda.ipynb`.

## Scope

Descriptive analysis of the cleaned Global Weather Repository data. The global
headline series uses the 183 core locations, equal-weighted by city, and only
the 875 of 879 UTC dates with `coverage_ok == True`. No forecasting model was
trained and no forecast metric was calculated.

## Key findings

- **Coverage:** 183 core cities; 159,905 core city-days. The global
  core series has 875 usable UTC dates out
  of 879.
- **Global level:** the usable global daily mean is
  **21.48 °C**, with daily values from
  13.51 to
  27.80 °C. The minimum occurs on
  21 Jan 2026 and
  the maximum on
  19 Jul 2024.
- **Seasonality:** monthly means range from
  **14.56 °C** in 2026-01 to
  **27.02 °C** in 2024-07, an
  amplitude of 12.45 °C.
  This is a 2.5-year descriptive pattern, not a climate trend.
- **City differences:** Doha has the highest core-city mean
  (32.97 °C); Ulaanbaatar has the lowest
  (6.04 °C), a city-mean range of
  26.93 °C.
- **Seasonal amplitude:** Ulaanbaatar has the largest monthly amplitude
  (45.1 °C); Majuro has the smallest
  (2.0 °C). This contrast is consistent with
  continental versus tropical maritime climates.
- **Latitude:** the Spearman correlation between latitude and city mean
  temperature is **-0.53**, so latitude is informative but not
  sufficient; altitude and local climate matter (for example, Quito at the
  equator is much cooler than Singapore).
- **Within-city associations with temperature:** median Spearman ρ is
  -0.55 for humidity,
  0.63 for UV,
  -0.34 for pressure,
  0.32 for ozone, and
  -0.10 for PM2.5.
  These are descriptive associations, not causal effects.
- **Conditions:** 61.9% of core city-days are labelled Sunny or
  Partly cloudy; rain/drizzle/thunder labels account for 18.9%.
  Rare dust/sandstorm labels are associated with much warmer conditions, but the labels
  are not independent physical measurements.
- **Air quality:** PM2.5 has median 13.3, 90th percentile
  50.1, and 99th percentile 154.3
  (dataset units). Riyadh has the highest core-city mean
  (133.9); Honiara has the lowest
  (3.2). Units are not documented in the Kaggle schema, so
  absolute regulatory comparisons are avoided.

## Figures

| Figure | What it shows |
|---|---|
| `fig_01_coverage_summary.png` | City-day rows and groups by tier; global core coverage distribution |
| `fig_02_global_temperature_and_coverage.png` | Global daily core temperature, 30-day rolling mean, and city coverage |
| `fig_03_monthly_seasonal_cycle.png` | Monthly global series and calendar-month seasonal cycle |
| `fig_04_temperature_distributions.png` | Core city-day and city-mean temperature distributions |
| `fig_05_city_temperature_extremes.png` | Hottest and coolest core cities with observed ranges |
| `fig_06_seasonal_amplitude.png` | Largest and smallest city seasonal amplitudes |
| `fig_07_latitude_and_temperature.png` | City mean temperature versus latitude, colored by amplitude |
| `fig_08_within_city_correlations.png` | Median within-city Spearman correlations |
| `fig_09_weather_relationships.png` | Pooled hexbin relationships plus a city-mean PM2.5/PM10 panel |
| `fig_10_condition_frequency.png` | Most frequent condition labels and their mean temperatures |
| `fig_11_pm25_air_quality.png` | PM2.5 distribution and highest city means |

## Caveats for modelling

- The global series is an **equal-weighted panel average**, not a population-
  or area-weighted global temperature.
- The record is only May 2024–October 2026; annual comparisons are incomplete
  because 2024 starts in May and 2026 ends in October.
- Only 183 of 269 country+city groups have near-complete coverage. Sparse
  groups must not be used for city-level forecasting.
- Four global UTC dates have severe coverage dips and are excluded from the
  headline series via `coverage_ok`.
- City labels and country labels are only partially normalised; unusual place
  names were retained when coordinates were consistent with the stated country.
- Air-quality units are not documented in the source schema.
- These EDA associations do not establish causality and do not select a
  forecasting model.
