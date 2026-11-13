# Phase E–F — Advanced analysis: anomalies, models, climate, air quality, and maps

**Status:** Complete for review through Phase F. Phase E covers anomaly
detection and model comparison; Phase F adds climate/region, air-quality,
feature-importance, and map analysis.

## Scope and inputs

- **Global target:** equal-weighted daily mean temperature across the 183 core
  locations (`temperature_celsius`).
- **Global calendar coverage:** 881 daily dates from
  2024-05-16 to 2026-10-13;
  875 observed dates.
- **City-day panel:** 159,905 core city-days across
  183 cities.
- **Evaluation:** 6 chronological rolling-origin folds, each with a
  30-day test window; **180 predictions per model**.
- **Extended models:** ElasticNet + Fourier, HistGradientBoosting, and three
  fixed ensembles: Ridge + ElasticNet mean, all-model mean, all-model median.
- **Anomaly detection:** descriptive only; no anomaly is removed from the
  forecast evaluation.

## Anomaly detection

### Global temperature screen

The global screen uses a trailing 31-observed-day median and a
`1.4826 × MAD` scale; the current day is excluded from its own baseline. A day
is flagged when `|z| >= 4`. The observed global series has
**15** flagged days out of
**875**.

| Date (UTC) | Global mean (°C) | Prior 31-day median (°C) | Robust scale (°C) | z | Flagged |
|---|---|---|---|---|---|
| 2026-08-17 | 17.543 | 23.504 | 0.564 | -10.572 | True |
| 2025-03-06 | 20.016 | 17.186 | 0.452 | 6.259 | True |
| 2025-03-07 | 19.784 | 17.186 | 0.483 | 5.380 | True |
| 2024-09-17 | 24.677 | 26.668 | 0.394 | -5.059 | True |
| 2025-03-05 | 19.453 | 17.186 | 0.452 | 5.015 | True |
| 2024-09-16 | 24.812 | 26.744 | 0.394 | -4.907 | True |
| 2025-03-10 | 20.207 | 17.457 | 0.568 | 4.837 | True |
| 2025-03-08 | 19.846 | 17.347 | 0.517 | 4.835 | True |
| 2024-09-15 | 24.863 | 26.757 | 0.394 | -4.810 | True |
| 2025-03-09 | 19.874 | 17.378 | 0.527 | 4.739 | True |

**Largest event:** 2026-08-17 has a global mean of
**17.54 °C**, versus a prior-31-day median of
**23.50 °C**; **z = -10.57** and the robust scale is
**0.56 °C**.

### City-day methods

The city temperature screen flags **2,219**
of **159,905** city-days
(**1.39%**). The Isolation Forest
(300 trees, 1% contamination) flags
**1,600** city-days
(**1.00%**). It is fitted on
within-city median/MAD-standardised temperature, humidity, pressure, wind,
cloud, precipitation, visibility, and UV features.

| Category | City-days |
|---|---|
| Both methods | 94 |
| Temperature z only | 2125 |
| Isolation Forest only | 1506 |

The two detectors agree on **94** city-days. The temperature screen
contributes **2,125** unique flags and the Isolation Forest
**1,506**. They answer different questions: a single-variable
temperature departure versus an unusual multivariate weather combination.

### The August 2026 event

On **2026-08-17**, the temperature screen flags
**107 of 182** core cities;
the Isolation Forest flags **24**; **17**
are flagged by both and **114** by at least one.
The next-highest combined date, **2024-09-13**, has
only **17** flags. The largest single
departure on the event date is **Monaco, Monaco**
at **5.3 °C** with a temperature z-score of
**-47.21**.

| Country | City | Temperature (°C) | Temperature z | IF score | Temperature flag | IF flag |
|---|---|---|---|---|---|---|
| Monaco | Monaco | 5.300 | -47.214 | -0.002 | True | True |
| Barbados | Bridgetown | 20.800 | -43.167 | 0.004 | True | False |
| Mauritius | Port Louis | 34.100 | 18.717 | 0.014 | True | False |
| Turkmenistan | Ashgabat | 13.200 | -17.621 | 0.109 | True | False |
| Dominican Republic | Santo Domingo | 12.100 | -17.402 | 0.142 | True | False |
| Saint Vincent and the Grenadines | Kingstown | 22.600 | -16.188 | -0.038 | True | True |
| Dominica | Roseau | 20.000 | -15.513 | -0.040 | True | True |
| Norway | Oslo | -7.300 | -15.363 | 0.027 | True | False |
| Djibouti | Djibouti | 24.600 | -14.670 | 0.105 | True | False |
| Nicaragua | Managua | 19.900 | -14.614 | -0.006 | True | True |

The event is broad and physically implausible as a one-day global temperature
change; the city pattern is dominated by extremely low readings in several
locations alongside a few high outliers. It is consistent with a source-data or
timestamp artefact. It is **not** removed from the backtest.

## Extended model comparison

### Setup

- Same 15 lag/Fourier features and recursive forecast rule as Phase D.
- ElasticNet: `StandardScaler` + `ElasticNetCV(l1_ratio=0.5, alphas=[0.01,
  0.03, 0.1, 0.3, 1.0])`, with a three-split chronological inner validation
  inside each outer training fold. Alpha was selected as 0.01 in all six
  folds; the outer test windows were not used for selection.
- HistGradientBoosting: `max_iter=200`, `learning_rate=0.05`,
  `max_leaf_nodes=15`, `random_state=42`.
- Ensembles are fixed combination rules, not meta-models fitted on the test
  windows.

### Aggregate metrics

| Model | Predictions | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) |
|---|---|---|---|---|---|
| ElasticNet + Fourier | 180 | 0.4705 | 0.7232 | 2.1559 | 2.1049 |
| Ridge + ElasticNet mean | 180 | 0.4812 | 0.7294 | 2.2090 | 2.1576 |
| Ridge + Fourier | 180 | 0.5012 | 0.7463 | 2.3032 | 2.2508 |
| All-model mean | 180 | 0.5474 | 0.7788 | 2.5015 | 2.4432 |
| All-model median | 180 | 0.5485 | 0.7923 | 2.5078 | 2.4449 |
| HistGradientBoosting | 180 | 0.7449 | 0.9782 | 3.3617 | 3.3308 |
| SARIMAX + Fourier | 180 | 0.9580 | 1.1936 | 4.3539 | 4.2103 |
| Persistence | 180 | 1.0368 | 1.2494 | 4.6777 | 4.6767 |
| Seasonal naive | 180 | 1.8513 | 2.0141 | 8.4135 | 8.0042 |

**Best aggregate model:** **ElasticNet + Fourier** with MAE **0.4705 °C**,
RMSE **0.7232 °C**, MAPE **2.156%**, and
sMAPE **2.105%**. It is **54.6%**
lower in MAE than persistence and **6.1%** lower than
Ridge + Fourier.

**Ensembles:** Ridge + ElasticNet mean has MAE
**0.4812 °C**; the all-model mean
**0.5474 °C**; the all-model median
**0.5485 °C**. None beats ElasticNet. The Ridge +
ElasticNet mean beats Ridge alone but not ElasticNet.

### Per-fold MAE

| Fold | Persistence | Seasonal naive | SARIMAX + Fourier | Ridge + Fourier | ElasticNet + Fourier | HistGradientBoosting | Ridge + ElasticNet mean | All-model mean | All-model median |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.655 | 2.335 | 1.269 | 0.624 | 0.434 | 0.550 | 0.526 | 0.495 | 0.526 |
| 2 | 1.609 | 1.868 | 0.665 | 0.570 | 0.480 | 0.781 | 0.516 | 0.523 | 0.521 |
| 3 | 1.066 | 1.940 | 0.793 | 0.496 | 0.541 | 0.697 | 0.516 | 0.477 | 0.536 |
| 4 | 0.717 | 1.930 | 1.328 | 0.570 | 0.592 | 1.031 | 0.575 | 0.748 | 0.739 |
| 5 | 1.463 | 1.904 | 1.349 | 0.449 | 0.490 | 0.911 | 0.462 | 0.747 | 0.674 |
| 6 | 0.711 | 1.131 | 0.345 | 0.298 | 0.286 | 0.499 | 0.291 | 0.295 | 0.295 |

Per-fold winners: **ElasticNet + Fourier: 3; Ridge + Fourier: 2; All-model mean: 1**. ElasticNet wins
**3 of 6** folds; no model wins every fold.

### MAE at selected horizons

| Horizon (days) | Ridge + Fourier | ElasticNet + Fourier | HistGradientBoosting | Ridge + ElasticNet mean | All-model mean | All-model median |
|---|---|---|---|---|---|---|
| 1 | 0.214 | 0.233 | 0.281 | 0.224 | 0.204 | 0.200 |
| 7 | 0.449 | 0.480 | 0.829 | 0.464 | 0.613 | 0.550 |
| 14 | 0.453 | 0.533 | 1.064 | 0.493 | 0.676 | 0.654 |
| 21 | 0.621 | 0.468 | 0.542 | 0.545 | 0.440 | 0.520 |
| 30 | 0.722 | 0.594 | 0.886 | 0.652 | 0.666 | 0.743 |

The all-model median is best at day 1 (0.200 °C), Ridge is best at days 7 and 14
(0.449 and 0.453 °C), the all-model mean is best at day 21 (0.440 °C), and
ElasticNet is best at day 30 (0.594 °C). The horizon figure includes a large
day-3 error spike from the 2026-08-17 event in fold 5; the event is retained in
the aggregate and per-fold results, and the selected horizon table follows the
Phase D checkpoints rather than removing the anomaly.

### Last holdout

Last test fold: **2026-09-14 to 2026-10-13**.

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) |
|---|---|---|---|---|
| ElasticNet + Fourier | 0.286 | 0.357 | 1.362 | 1.365 |
| Ridge + ElasticNet mean | 0.291 | 0.363 | 1.389 | 1.393 |
| All-model median | 0.295 | 0.359 | 1.404 | 1.405 |
| All-model mean | 0.295 | 0.373 | 1.406 | 1.400 |
| Ridge + Fourier | 0.298 | 0.370 | 1.419 | 1.424 |
| SARIMAX + Fourier | 0.345 | 0.410 | 1.653 | 1.638 |
| HistGradientBoosting | 0.499 | 0.617 | 2.377 | 2.352 |
| Persistence | 0.711 | 0.812 | 3.424 | 3.353 |
| Seasonal naive | 1.131 | 1.259 | 5.329 | 5.163 |

## Findings

- **The August 2026 event is the dominant joint anomaly.** It is the largest
  global robust-z departure and the largest city-day cluster by a wide margin.
  It is best treated as a likely data-quality event, not a real one-day global
  temperature collapse.
- **The two anomaly methods are complementary.** Most flags are unique to one
  method. Agreement on 94 city-days identifies the strongest joint
  cases; disagreement is expected given the different inputs.
- **ElasticNet is the best aggregate forecaster in this extended comparison.**
  Its MAE of 0.4705 °C is lower than Ridge's
  0.5012 °C, SARIMAX's 0.9580 °C, and
  HistGradientBoosting's 0.7449 °C.
- **Fixed ensembles do not beat the best single model.** Equal weighting gives
  too much influence to weaker base models. A learned stack would need an
  inner time-series validation design; it is not claimed here.
- **Horizon performance is not uniform.** Ridge is strongest at days 7 and 14,
  while ElasticNet is strongest at day 30. This is a reason to report the full
  horizon profile rather than only the aggregate.

## Leakage controls

- All models use the same six chronological folds; each training set ends one
  day before its test window.
- Base models are refitted inside each training fold. ElasticNet scaling is
  fitted on training rows only. No test target is used to fit a scaler, feature,
  or model.
- Recursive forecasts use prior predictions as history; actual test values are
  never inserted into the recursive path.
- Fourier terms are deterministic calendar features and do not use the target.
- The ensembles are fixed arithmetic combinations of base-model predictions;
  no ensemble weight is estimated on the test window.
- The six non-observed global dates are forward-filled only for model
  continuity; all are before the first test window and none is an evaluation
  target.
- Anomaly flags are descriptive and are never used to filter or reweight the
  forecast targets.

## Caveats

- The global target is an equal-weighted average of core locations, not a
  population- or area-weighted global temperature.
- The record spans a little over two years; it can describe seasonal behaviour
  in this panel but cannot establish a long-term climate trend.
- The anomaly thresholds are analytical choices, not ground truth. The
  Isolation Forest uses within-city standardisation over the full available
  history, so it is descriptive rather than a live real-time detector.
- The Isolation Forest's high-count May 2024 cluster is not confirmed as a data
  error. It may reflect unusual weather combinations near the start of the
  record or properties of the detector.
- The August 2026 event is retained in fold 5. That is deliberate: removing it
  after observing model scores would bias the evaluation.
- Model selection used the same six folds that are reported. The metrics are a
  model-selection estimate, not an untouched final holdout.
- HistGradientBoosting is weaker than the linear models in this recursive
  setup. The comparison is reported as observed; no post-hoc tuning was done on
  the test windows.
- Phase E did not include climate-region analysis, air-quality analysis,
  feature importance with two methods, or maps. Phase F adds those analyses;
  the final report and demo script remain Phase G work.
- The official PM Accelerator mission statement is still a placeholder pending
  the user's exact text; it is not invented here.

## Figures

- `reports/figures/fig_16_global_and_city_anomalies.png`
- `reports/figures/fig_17_anomaly_dates_and_event.png`
- `reports/figures/fig_18_advanced_model_comparison.png`
- `reports/figures/fig_19_advanced_error_by_horizon.png`

<!-- PHASE F START -->
## Phase F — Climate, region, air quality, feature importance, and maps

**Status:** complete for review. This section is descriptive; no new forecast
metric is claimed. The Phase E anomaly and model-comparison results above are
unchanged.

### Scope and derived groupings

- **Panel:** 183 core cities and 159,905 city-day rows;
  159,904 city-days have a temperature value.
- **Climate zone:** absolute-latitude bands (Tropical <23.5°, Subtropical
  23.5–<35°, Temperate 35–<50°, High-latitude ≥50°). This is a transparent
  proxy, not an official Köppen classification.
- **Region:** first component of the IANA timezone string. It is a descriptive
  proxy, not an official continent mapping.
- **Air quality:** all six pollutant variables are in **dataset units** because
  the source does not document concentration units. No regulatory or health
  threshold is applied.
- **Maps:** the public-domain Natural Earth 1:110m land layer is a backdrop
  only; all plotted values come from the weather panel.
- **Feature importance:** standardised ElasticNet coefficients plus
  permutation importance on one-step-ahead test rows. The latter is not the
  recursive 30-day backtest.

The 2026-08-17 temperature event remains in the panel. Its effect on
multi-year city means is negligible, but it is not removed.

### Climate and regional analysis

| Climate zone | Cities | City-days | Mean temp (°C) | Median city temp (°C) | Median seasonal amplitude (°C) | Mean humidity (%) | Mean UV index | Mean wind (kph) | Median city PM2.5 | PM2.5 Q1 | PM2.5 Q3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tropical | 90 | 78540 | 25.32 | 25.97 | 7.54 | 72.54 | 3.12 | 13.08 | 12.59 | 7.95 | 21.46 |
| Subtropical | 27 | 23646 | 23.99 | 23.36 | 20.94 | 52.17 | 4.65 | 13.14 | 24.35 | 13.82 | 46.72 |
| Temperate | 48 | 41950 | 16.75 | 17.28 | 30.21 | 61.36 | 2.63 | 11.21 | 12.80 | 10.46 | 16.21 |
| High-latitude | 18 | 15768 | 11.07 | 10.84 | 30.09 | 73.64 | 1.78 | 14.73 | 9.58 | 7.26 | 11.63 |

The tropical band contains **90**
cities with a mean city temperature of
**25.32 °C**
and a median seasonal amplitude of
**7.54 °C**.
The temperate and high-latitude bands have median amplitudes near
**30.2 °C**
and
**30.1 °C**,
respectively. The zone temperature difference is significant in a descriptive
Kruskal-Wallis test (H = **99.01**, p = **2.54e-21**), but
this is not a causal or trend claim.

Absolute latitude has a negative city-level Spearman association with mean
temperature (**rho = -0.733**, p = **4.20e-32**) and a
positive association with seasonal amplitude
(**rho = 0.816**, p = **5.16e-45**). The hottest city mean
is **Doha, Qatar** at
**32.97 °C**; the coldest is
**Ulaanbaatar, Mongolia** at
**6.04 °C**. The largest seasonal amplitude is
**Ulaanbaatar, Mongolia**
(**45.1 °C**); the smallest is
**Majuro, Marshall Islands**
(**2.0 °C**).

| Region | Cities | City-days | Mean temp (°C) | Median seasonal amplitude (°C) | Mean humidity (%) | Median city PM2.5 |
|---|---|---|---|---|---|---|
| Indian Ocean | 5 | 4341 | 26.35 | 6.70 | 69.45 | 7.65 |
| Pacific | 11 | 9595 | 25.50 | 4.45 | 82.15 | 5.00 |
| Asia | 46 | 40243 | 24.73 | 22.21 | 50.66 | 20.72 |
| Africa | 43 | 37671 | 24.62 | 11.75 | 63.49 | 19.06 |
| Atlantic | 1 | 877 | 23.99 | 8.57 | 73.17 | 18.45 |
| Americas | 32 | 27766 | 20.86 | 6.55 | 83.96 | 8.88 |
| Australia | 1 | 874 | 15.21 | 24.49 | 62.74 | 5.18 |
| Europe | 44 | 38537 | 13.98 | 29.60 | 69.87 | 11.84 |

The region table is a timezone proxy, and the groups are unevenly sampled:
Asia has 46
core cities and Europe has
44,
while the Atlantic and Australia groups have one city each. The regional
contrasts should therefore be read as panel descriptions, not population- or
area-weighted continent estimates.

### Air-quality analysis

| Pollutant | Units | Rows | Missing | Median | Mean | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|
| PM2.5 | dataset units | 159905 | 0 | 13.32 | 23.08 | 74.37 | 154.29 | 1614.10 |
| PM10 | dataset units | 159903 | 2 | 19.06 | 46.80 | 150.90 | 470.46 | 6037.29 |
| Ozone | dataset units | 159905 | 0 | 54.40 | 58.18 | 113.00 | 161.70 | 480.70 |
| Nitrogen dioxide | dataset units | 159905 | 0 | 5.73 | 14.38 | 60.44 | 109.89 | 427.70 |
| Sulphur dioxide | dataset units | 159904 | 1 | 2.35 | 9.95 | 48.10 | 109.52 | 521.33 |
| Carbon monoxide | dataset units | 159904 | 1 | 264.85 | 419.46 | 1118.72 | 2963.65 | 38879.40 |

The **median of the 183 city medians** is
**12.77** dataset units; the **pooled
city-day median** is
**13.32**.
The pooled 95th percentile is
**74.37**,
the 99th percentile is
**154.29**,
and the maximum city-day value is
**1614.10**.
The distribution is strongly right-skewed; values are not treated as normally
distributed.

| Country | City | Climate zone | Region | Median PM2.5 | P95 PM2.5 |
|---|---|---|---|---|---|
| Saudi Arabia | Riyadh | Subtropical | Asia | 99.99 | 368.24 |
| Indonesia | Jakarta | Tropical | Asia | 99.34 | 301.14 |
| China | Beijing | Temperate | Asia | 95.65 | 286.34 |
| India | New Delhi | Subtropical | Asia | 91.01 | 216.96 |
| Kuwait | Kuwait City | Subtropical | Asia | 81.55 | 189.95 |
| Chile | Santiago | Subtropical | Americas | 60.50 | 597.29 |
| Bahrain | Manama | Subtropical | Asia | 58.09 | 119.03 |
| Vietnam | Hanoi | Tropical | Asia | 54.67 | 155.91 |

The highest city-median values are **Riyadh, Saudi Arabia**
(**99.99**), **Jakarta, Indonesia**
(**99.34**), and **Beijing, China**
(**95.65**). The lowest city median is
**Honiara, Solomon Islands** at
**2.40**.

**Level shift:** the September 2024 median city-level PM2.5 ratio versus
August 2024 is **2.932**, with
**78.1%** of cities above 1.5x and
**64.5%** above 2x. The same September comparison
is **0.972** in 2025 and **0.937**
in 2026. This is better treated as a source/level discontinuity than as a
recurring seasonal effect. It is not used to make a causal or regulatory claim.

| Month | Previous month | Cities | Median city ratio | Share ratio >1.5 | Share ratio >2 |
|---|---|---|---|---|---|
| 2024-09 | 2024-08 | 183 | 2.932 | 0.781 | 0.645 |
| 2025-09 | 2025-08 | 183 | 0.972 | 0.049 | 0.000 |
| 2026-09 | 2026-08 | 183 | 0.937 | 0.049 | 0.016 |

| Weather variable | Cities | Median rho | Q1 rho | Q3 rho | Share negative | Share positive | Share significant negative | Share significant positive |
|---|---|---|---|---|---|---|---|---|
| Temp | 183 | -0.103 | -0.260 | 0.124 | 0.617 | 0.383 | 0.546 | 0.311 |
| Humidity | 183 | -0.045 | -0.201 | 0.135 | 0.546 | 0.454 | 0.432 | 0.372 |
| Wind | 183 | -0.167 | -0.310 | 0.098 | 0.694 | 0.306 | 0.639 | 0.273 |
| Pressure | 183 | 0.139 | -0.034 | 0.281 | 0.295 | 0.705 | 0.197 | 0.634 |
| Precip | 183 | -0.146 | -0.251 | -0.070 | 0.885 | 0.115 | 0.754 | 0.049 |
| Cloud | 183 | -0.107 | -0.224 | -0.020 | 0.781 | 0.219 | 0.596 | 0.098 |
| UV | 181 | -0.063 | -0.276 | 0.111 | 0.569 | 0.431 | 0.497 | 0.304 |
| Visibility | 172 | -0.085 | -0.219 | 0.030 | 0.680 | 0.320 | 0.535 | 0.157 |

Within-city Spearman correlations with PM2.5 are heterogeneous. The most
negative median associations are wind (**rho = -0.167**;
**63.9%** of cities significant
negative), precipitation (**rho = -0.146**;
**75.4%**), and cloud
(**rho = -0.107**;
**59.6%**). Pressure has a
positive median association (**rho = 0.139**;
**63.4%** significant
positive), and temperature is weaker and less consistent
(**rho = -0.103**). The signs are not universal
across cities, so these are descriptive local associations, not general
physical laws.

### Feature importance

| Feature | Mean coefficient | Fold SD | Mean absolute coefficient | Non-zero share |
|---|---|---|---|---|
| Lag 1 day | 1.8553 | 0.1480 | 1.8553 | 1.0000 |
| Lag 2 days | 0.4275 | 0.0709 | 0.4275 | 1.0000 |
| Lag 365 days | 0.4224 | 0.0245 | 0.4224 | 1.0000 |
| Rolling mean 7 days | 0.3438 | 0.0263 | 0.3438 | 1.0000 |
| Cos 1 (annual) | -0.2511 | 0.0631 | 0.2511 | 1.0000 |
| Lag 3 days | 0.1659 | 0.0981 | 0.1659 | 1.0000 |
| Rolling mean 30 days | 0.1443 | 0.0485 | 0.1443 | 1.0000 |
| Lag 14 days | 0.0633 | 0.0533 | 0.0633 | 1.0000 |
| Lag 7 days | 0.0409 | 0.0267 | 0.0409 | 0.8333 |
| Cos 3 (four-month) | -0.0369 | 0.0105 | 0.0369 | 1.0000 |

The largest mean absolute standardised coefficients are
**Lag 1 day** (**+1.855**),
**Lag 2 days** (**+0.428**),
**Lag 365 days** (**+0.422**),
and **Rolling mean 7 days**
(**+0.344**). The annual cosine term is negative and
non-trivial, so the model does use a seasonal shape even though lag-1 is
largest.

**ElasticNet one-step permutation importance:**

| Feature | Mean MAE increase (°C) | Fold SD | Folds | Mean within-fold SD |
|---|---|---|---|---|
| Lag 1 day | 0.1081 | 0.0454 | 6 | 0.0411 |
| Lag 365 days | 0.0022 | 0.0278 | 6 | 0.0116 |
| Lag 3 days | 0.0004 | 0.0081 | 6 | 0.0049 |
| Sin 2 (semiannual) | 0.0003 | 0.0007 | 6 | 0.0004 |
| Sin 1 (annual) | 0.0001 | 0.0001 | 6 | 0.0002 |
| Cos 2 (semiannual) | -0.0000 | 0.0002 | 6 | 0.0001 |
| Sin 3 (four-month) | -0.0001 | 0.0011 | 6 | 0.0011 |
| Lag 30 days | -0.0003 | 0.0005 | 6 | 0.0001 |

**HistGradientBoosting one-step permutation importance:**

| Feature | Mean MAE increase (°C) | Fold SD | Folds | Mean within-fold SD |
|---|---|---|---|---|
| Lag 1 day | 0.0758 | 0.0653 | 6 | 0.0373 |
| Lag 365 days | 0.0250 | 0.0710 | 6 | 0.0247 |
| Rolling mean 7 days | 0.0133 | 0.0628 | 6 | 0.0129 |
| Lag 2 days | 0.0055 | 0.0119 | 6 | 0.0122 |
| Lag 3 days | 0.0015 | 0.0046 | 6 | 0.0045 |
| Sin 1 (annual) | 0.0002 | 0.0012 | 6 | 0.0004 |
| Cos 1 (annual) | -0.0003 | 0.0008 | 6 | 0.0007 |
| Cos 2 (semiannual) | -0.0010 | 0.0026 | 6 | 0.0010 |

**Grouped permutation importance:**

| Feature group | Mean MAE increase (°C) | Fold SD | Folds | Mean within-fold SD | Model |
|---|---|---|---|---|---|
| Recent lags (1-3 d) | 0.1265 | 0.0595 | 6 | 0.0410 | ElasticNet |
| Annual lag (365 d) | 0.0049 | 0.0265 | 6 | 0.0114 | ElasticNet |
| Annual Fourier | -0.0008 | 0.0146 | 6 | 0.0064 | ElasticNet |
| Weekly/monthly lags (7-30 d) | -0.0021 | 0.0045 | 6 | 0.0033 | ElasticNet |
| Rolling means (7/30 d) | -0.0030 | 0.0138 | 6 | 0.0087 | ElasticNet |
| Recent lags (1-3 d) | 0.0914 | 0.0624 | 6 | 0.0353 | HGB |
| Annual lag (365 d) | 0.0312 | 0.0740 | 6 | 0.0253 | HGB |
| Rolling means (7/30 d) | 0.0158 | 0.0703 | 6 | 0.0121 | HGB |
| Annual Fourier | -0.0071 | 0.0111 | 6 | 0.0062 | HGB |
| Weekly/monthly lags (7-30 d) | -0.0081 | 0.0114 | 6 | 0.0103 | HGB |

The top one-step permutation result is **Lag 1 day**
for ElasticNet (**+0.1081 °C MAE increase**) and
**Lag 1 day** for HGB
(**+0.0758 °C**). Grouped permutation confirms that
recent lags dominate the one-step view: ElasticNet
**+0.1265 °C**, HGB
**+0.0914 °C**. The annual-lag group is smaller
(ElasticNet **+0.0049 °C**; HGB
**+0.0312 °C**), and the annual Fourier group
is close to zero or slightly negative in this one-step design (ElasticNet
**-0.0008 °C**; HGB
**-0.0071 °C**). Small negative values mean that permutation slightly improved the test MAE in
this sample; they are consistent with noise and should not be interpreted as
the feature being harmful.

**Interpretation caveat:** one-step permutation importance is not the same as
recursive multi-step importance. In the recursive backtest, lag-1 is replaced
by the model's own previous prediction, so the model must rely more on
seasonality, longer lags, and rolling means. No recursive feature-importance
claim is made here.

### Phase F caveats

- The climate zones are an absolute-latitude proxy; they do not encode
  elevation, continentality, ocean currents, or official Köppen classes.
- The region groups come from timezone strings and are unevenly sampled.
- Air-quality units are undocumented. The values are not compared with any
  regulatory threshold, and no health claim is made.
- The September 2024 PM2.5 level shift affects cross-time comparisons. It is
  documented but not removed or corrected.
- Within-city Spearman correlations are descriptive; p-values are not
  multiple-testing corrected. Signs vary across cities, so a single global
  relationship is not claimed.
- Feature-importance methods can disagree when predictors are correlated.
  Coefficients are model-specific and standardised; permutation importance is
  one-step and model-specific. Neither is a causal effect.
- The maps use an equirectangular projection. Marker size and colour encode
  different quantities in the two map figures, as stated in their titles.

### Phase F figures

- `reports/figures/fig_20_climate_zone_map.png`
- `reports/figures/fig_21_climate_zone_profiles.png`
- `reports/figures/fig_22_pm25_city_map.png`
- `reports/figures/fig_23_air_quality_patterns.png`
- `reports/figures/fig_24_feature_importance.png`

### Phase F checkpoint

Phase F is complete for review. The notebook executed from top to bottom and
wrote this summary from the actual dataframes. Stop here before Phase G: final
report, PDF export, mission statement insertion, README finalisation, and
`DEMO_SCRIPT.md`.

