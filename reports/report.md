# ChronoWeather: Global Meteorological Intelligence & Time-Series Forecasting Platform
### Technical Assessment Report • AI Engineer Intern
**Author:** Marimuthu  
**Organization:** Product Manager Accelerator (PM Accelerator)  
**Program:** AI Engineer Intern Technical Assessment  
**Date:** November 2026  
**Repository:** `https://github.com/lengendzhub/ChronoWeather-.git`  

---

## Executive Summary

This report delivers a rigorous, publication-grade meteorological analytics and time-series forecasting study conducted on Kaggle's **Global Weather Repository** dataset (170,928 raw records across 211 countries and 268 cities from May 16, 2024 to October 13, 2026). 

The primary objective is to evaluate global and regional atmospheric patterns, quantify within-city weather and air-quality associations, detect extreme multivariate anomalies, and develop robust, zero-leakage predictive models forecasting global mean temperature trajectories across multi-week horizons.

### Key Milestones & Quantitative Findings:
1. **Data Engineering & Quality Assurance (Phase A & B):** 
   - Audited the raw 41-column schema, resolving entity naming variants (191 cleaned countries, 262 cleaned cities) and establishing **three location tiers** (183 Core cities with $\ge 800$ daily observations, 33 Partial, and 53 Sparse).
   - Removed 1 duplicate record, collapsed 416 intra-day duplicate timestamps, eliminated redundant imperial unit columns, and masked physically impossible sensor readings while preserving authentic weather extremes.
2. **Exploratory Data Analysis (Phase C):**
   - The headline equal-weighted global core daily mean is **21.48 °C** (observed range: 13.51 °C to 27.80 °C) with an annual seasonal amplitude of **12.45 °C** (monthly means from 14.56 °C in Jan 2026 to 27.02 °C in Jul 2024).
   - Extreme city contrasts: **Doha, Qatar** is the warmest core city (mean 32.97 °C); **Ulaanbaatar, Mongolia** is the coldest (mean 6.04 °C) with the highest seasonal amplitude (45.1 °C). **Majuro, Marshall Islands** exhibits maritime stability with an amplitude of only 2.0 °C.
   - Temperature demonstrates strong negative association with humidity (Spearman $\rho = -0.55$) and pressure ($\rho = -0.34$), and strong positive association with UV index ($\rho = +0.63$).
3. **Time-Series Forecasting & Backtesting (Phase D & E):**
   - Implemented a **6-fold rolling-origin (walk-forward) backtest** evaluating 30-day forecast windows (180 out-of-sample predictions per model) strictly preventing data leakage.
   - **Champion Model:** **ElasticNet + Fourier** achieved an aggregate **MAE of 0.4705 °C** (RMSE 0.7303 °C, MAPE 2.15%), representing a **54.6% error reduction** over the persistence baseline (**1.0371 °C MAE**) and outperforming Ridge + Fourier (**0.5012 °C MAE**) and Seasonal Naive (**1.8510 °C MAE**).
4. **Multivariate Anomaly Detection (Phase E):**
   - Evaluated both univariate robust Z-scores (median/MAD) and a 300-tree multivariate **Isolation Forest**.
   - Identified and isolated an anomalous global dip on **August 17, 2026** ($-5.36\text{ °C}$ global swing, $z = -10.57$) driven by batching/timestamp errors across 107 cities. Rather than dropping the date to artificially inflate scores, it was transparently retained in Fold 5.
5. **Geospatial & Air Quality Analysis (Phase F):**
   - Classified locations into 5 macro-climate zones (Tropical, Arid, Temperate, Continental, Polar/Subpolar) mapped over Natural Earth landmasses.
   - Profiled global PM2.5 pollution, documenting a dataset-wide measurement level shift in September 2024 and identifying **Riyadh, Saudi Arabia** as the highest PM2.5 city mean (133.9).
   - Verified via Permutation and MDI Feature Importance that 7-day rolling temperature means and 1-day autoregressive lags drive over 55% of predictive power.

---

## 1. Product Manager Accelerator Mission & Vision

> **Official Mission Statement:**  
> *"The Product Manager Accelerator (PM Accelerator) is dedicated to empowering professionals to break into top-tier traditional and AI Product Management and AI Engineering roles. By bridging the gap between cutting-edge technology and business impact, PM Accelerator equips fellows with hands-on, end-to-end technical frameworks, portfolio-grade project execution, and strategic product leadership to accelerate high-impact tech careers and lead responsible AI transformations worldwide."*

This data analytics assessment embodies the PM Accelerator engineering philosophy: treating data with extreme intellectual honesty, building reproducible pipelines without look-ahead bias, understanding the business and domain implications of results, and communicating complex technical architectures with executive clarity.

---

## 2. Dataset Architecture & Preprocessing

### 2.1 Provenance & Integrity
- **Source:** Kaggle *Global Weather Repository* by nelgiriyewithana.
- **Physical Dataset:** `data/GlobalWeatherRepository.csv` (42.98 MB, SHA-256: `be39946635a451996421eaaacfba0d0b5727c6a599f3a57566f79ec2ac8ecb56`).
- **Dimensions:** 170,928 rows × 41 columns.
- **Temporal Span:** May 16, 2024 01:45 to October 13, 2026 18:45 (880 days, ~2.5 years).
- **Timezone Conventions:** `last_updated` represents local city time; `last_updated_epoch` represents UTC epoch timestamps. Both agree across 100% of rows upon timezone conversion.

### 2.2 Entity Harmonization & Location Tiering
The raw dataset contains naming redundancies and uneven reporting frequency. In Phase B, entity resolution consolidated:
- **Countries:** 211 raw names $\to$ 191 standard country labels (5,144 rows harmonized).
- **Cities:** 268 raw names $\to$ 262 standard city names (1,187 rows harmonized).

Locations were systematically partitioned into **three tiers**:
1. **Core Tier (183 locations / 160,285 rows):** Cities with near-complete coverage ($\ge 800$ observations). Form the basis of all global aggregates and time-series modeling.
2. **Partial Tier (33 locations / 10,007 rows):** Intermittent coverage (100–799 rows).
3. **Sparse Tier (53 locations / 635 rows):** Highly irregular reporting ($<100$ rows, 30 having only 1 row). Excluded from longitudinal modeling.

![Figure 1: Coverage Summary](figures/fig_01_coverage_summary.png)
*Figure 1: Distribution of observation counts across country+city groups and location tiers.*

### 2.3 Data Hygiene & Outlier Treatment
- **Duplicate Removal:** Removed 1 exact duplicate record; collapsed 416 intra-day duplicate city-date records into single daily observations.
- **Collinear Unit Pruning:** Dropped redundant imperial columns (`temperature_fahrenheit`, `wind_mph`, `pressure_in`, etc.) that were deterministic linear conversions.
- **Physical Boundary Verification:** Masked 1 impossible temperature/feels-like reading and 2 out-of-range barometric pressures ($<650$ or $>1100$ mb), while deliberately retaining genuine meteorological extremes (e.g. sub-zero winter temperatures in Siberia and extreme wind gusts during tropical cyclones).
- **Leakage Prevention:** StandardScalers are strictly fitted on training splits only; no global scaling or future-target information is introduced prior to modeling.

---

## 3. Exploratory Data Analysis & Meteorological Patterns

### 3.1 Global Temperature Baseline & Seasonal Cycles
The global headline series is computed as the equal-weighted daily average of the 183 core locations on UTC dates meeting a minimum coverage threshold of 150 cities (875 of 879 dates qualify).

- **Global Daily Mean:** **21.48 °C** (Standard Deviation: 3.42 °C).
- **Historical Envelope:** Absolute minimum of **13.51 °C** (January 21, 2026) and maximum of **27.80 °C** (July 19, 2024).
- **Seasonal Amplitude:** 12.45 °C spread between January (14.56 °C) and July (27.02 °C).

![Figure 2: Global Temperature & Coverage](figures/fig_02_global_temperature_and_coverage.png)
*Figure 2: Daily global core mean temperature, 30-day moving average, and daily station reporting volume.*

![Figure 3: Monthly Seasonal Cycle](figures/fig_03_monthly_seasonal_cycle.png)
*Figure 3: Monthly aggregated global temperature trajectories illustrating cyclical intra-annual harmonics.*

### 3.2 Geographical Variations & Extreme Climates
- **Warmest Core City:** Doha, Qatar ($32.97\text{ °C}$ mean).
- **Coldest Core City:** Ulaanbaatar, Mongolia ($6.04\text{ °C}$ mean).
- **Continental vs Maritime Extremes:** Ulaanbaatar exhibits the highest monthly thermal amplitude ($45.1\text{ °C}$), whereas Majuro (Marshall Islands) displays remarkable maritime stability ($2.0\text{ °C}$ amplitude).
- **Latitude Gradient:** City mean temperature correlates inversely with absolute latitude (Spearman $\rho = -0.53$). Deviations from this trend highlight the moderating impact of elevation and ocean currents (e.g. Quito at $0^\circ$ latitude maintains a cool $14.1\text{ °C}$ mean due to altitude).

![Figure 5: City Temperature Extremes](figures/fig_05_city_temperature_extremes.png)
*Figure 5: Hottest and coolest core global cities with observed historical min-max ranges.*

![Figure 7: Latitude vs Temperature Gradient](figures/fig_07_latitude_and_temperature.png)
*Figure 7: Relationship between station latitude and annual mean temperature, colored by seasonal amplitude.*

### 3.3 Inter-Variable Correlations & Air Quality
Evaluating within-city Spearman rank correlations reveals the physical coupling of meteorological forces:
- **Temperature & Humidity:** $\rho = -0.55$ (hotter air holds more moisture before saturation).
- **Temperature & UV Index:** $\rho = +0.63$ (direct correlation with solar zenith angle).
- **Temperature & Pressure:** $\rho = -0.34$ (thermal low-pressure formation).
- **Air Quality (PM2.5):** Median of 13.3, 90th percentile of 50.1, and 99th percentile of 154.3. Riyadh, Saudi Arabia recorded the highest annual mean (133.9), driven by frequent dust storm events.

![Figure 8: Within-City Correlation Matrix](figures/fig_08_within_city_correlations.png)
*Figure 8: Median within-city Spearman correlation coefficients between meteorological variables.*

![Figure 11: PM2.5 Air Quality Distribution](figures/fig_11_pm25_air_quality.png)
*Figure 11: Global distribution of PM2.5 concentrations and top polluted metropolitan areas.*

---

## 4. Time-Series Forecasting & Walk-Forward Backtesting

### 4.1 Problem Formulation & Backtest Design
To evaluate forecasting models realistically, we avoided random train/test splits (which cause severe temporal look-ahead leakage) and implemented a **chronological 6-fold rolling-origin backtest**.

- **Forecast Horizon ($H$):** 30 consecutive days per fold.
- **Evaluation Period:** April 17, 2026 to October 13, 2026 (180 total out-of-sample days).
- **Training Expansion:** Each fold trains on all observed history strictly ending 1 day prior to the test window ($N=701$ to $851$ days).
- **Multi-Step Prediction:** Autoregressive features are updated recursively during evaluation—predicted values become input lags for subsequent days.

![Figure 12: Rolling Forecast Backtest Windows](figures/fig_12_forecast_windows.png)
*Figure 12: Chronological allocation of expanding training sets and non-overlapping 30-day test folds.*

### 4.2 Candidate Models
1. **Persistence Benchmark:** Naive forecast assuming temperature remains constant at the last observed training value ($y_{t+h} = y_t$).
2. **Seasonal Naive Benchmark:** Assumes temperature equals the value observed exactly one year prior ($y_{t+h} = y_{t+h-365}$).
3. **Ridge Regression + Fourier:** Linear regularization with 3rd-order annual Fourier harmonics ($\sin, \cos$ pairs), target lags ($[1, 2, 3, 7, 14, 30, 365]$), and rolling moving averages.
4. **SARIMAX + Fourier:** Autoregressive Integrated Moving Average with exogenous annual Fourier drivers.
5. **ElasticNet + Fourier (Phase E):** L1/L2 regularized regression performing automatic feature sparsity selection.
6. **HistGradientBoosting (Phase E):** Non-linear gradient-boosted decision trees capable of modeling complex feature interactions.
7. **Fixed Model Ensembles:** Simple Average, Median, and Ridge + ElasticNet Blends.

### 4.3 Comprehensive Performance Results (180 Predictions Per Model)

| Rank | Model Architecture | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) | Improvement vs Persistence |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **ElasticNet + Fourier** | **0.4705** | **0.7303** | **2.15%** | **2.10%** | **+54.6%** |
| 🥈 | **Ridge + ElasticNet Ensemble** | **0.4740** | **0.7275** | **2.17%** | **2.12%** | **+54.3%** |
| 🥉 | **Ridge + Fourier** | **0.5012** | **0.7456** | **2.30%** | **2.25%** | **+51.7%** |
| 4 | All-Model Median Ensemble | 0.5337 | 0.7891 | 2.45% | 2.40% | +48.5% |
| 5 | All-Model Mean Ensemble | 0.5337 | 0.7937 | 2.45% | 2.40% | +48.5% |
| 6 | HistGradientBoosting | 0.6310 | 0.8654 | 2.87% | 2.80% | +39.2% |
| 7 | SARIMAX + Fourier | 0.9576 | 1.1942 | 4.35% | 4.21% | +7.7% |
| 8 | **Persistence Baseline** | **1.0371** | **1.2494** | **4.68%** | **4.68%** | Benchmark |
| 9 | **Seasonal Naive (365-day)** | **1.8510** | **2.0135** | **8.41%** | **8.00%** | -78.5% |

![Figure 13: Model Error Comparison](figures/fig_13_forecast_model_metrics.png)
*Figure 13: Aggregate MAE and RMSE performance comparison across candidate models.*

![Figure 18: Advanced Model Comparison](figures/fig_18_advanced_model_comparison.png)
*Figure 18: Detailed metric breakdown including gradient-boosted trees and fixed ensemble blends.*

### 4.4 Error Degradation Over Forecast Horizon
As the forecast horizon extends from 1 to 30 days, recursive error accumulates across all architectures. However, models regularized with Fourier seasonality resist unbounded divergence:

| Horizon ($h$) | Persistence MAE | Seasonal Naive MAE | Ridge + Fourier MAE | ElasticNet MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Day 1** | 0.285 °C | 1.926 °C | 0.214 °C | **0.208 °C** |
| **Day 7** | 0.907 °C | 1.953 °C | 0.449 °C | **0.421 °C** |
| **Day 14** | 1.092 °C | 2.032 °C | 0.453 °C | **0.435 °C** |
| **Day 21** | 0.952 °C | 1.671 °C | 0.621 °C | **0.589 °C** |
| **Day 30** | 1.337 °C | 1.842 °C | 0.722 °C | **0.684 °C** |

![Figure 14: Error Degradation by Forecast Horizon](figures/fig_14_forecast_error_by_horizon.png)
*Figure 14: Out-of-sample MAE progression over the 30-day forecast horizon.*

### 4.5 Post-Sample 14-Day Projection
Refitting the champion model on 100% of available data yields an out-of-sample projection for the subsequent fortnight (**October 14, 2026 to October 27, 2026**). Daily global means are projected to cool steadily from **20.34 °C down to 19.51 °C**, tracking the Northern Hemisphere autumnal decline, bounded by an empirical 80% residual interval $[-1.08\text{ °C}, +0.37\text{ °C}]$.

![Figure 15: Final Holdout & 14-Day Projection](figures/fig_15_holdout_and_projection.png)
*Figure 15: Last 30-day holdout predictions and 14-day post-sample trajectory with 80% confidence bands.*

---

## 5. Advanced Anomaly Detection & Diagnostic Audit

To distinguish physical extreme weather events from sensor corruption, Phase E applied two distinct anomaly detectors:
1. **Univariate Robust Z-Score:** Trailing 31-day median with Median Absolute Deviation ($1.4826 \times \text{MAD}$ scale). Flags days with $|z| \ge 4$.
2. **Multivariate Isolation Forest:** 300 isolation trees fitted on standardized 8-dimensional weather vectors (temperature, humidity, pressure, wind, cloud, rain, visibility, UV).

![Figure 16: Anomaly Detection Overview](figures/fig_16_global_and_city_anomalies.png)
*Figure 16: Time series of robust Z-scores and Isolation Forest anomaly scores across the panel.*

### 5.1 Case Study: The August 17, 2026 Global Collapse
On **August 17, 2026**, the global daily mean experienced an abrupt drop of **$-5.36\text{ °C}$** (from 22.90 °C to 17.54 °C, $z = -10.57$), followed by an immediate $+5.31\text{ °C}$ rebound on August 18.

- **Detector Reaction:** The univariate screen flagged **107 of 182 cities**; the Isolation Forest flagged 24 cities; 17 cities were flagged by both.
- **Physical Impossibility:** Monaco reported a summer temperature of $5.3\text{ °C}$ ($z = -47.21$); Bridgetown reported $20.8\text{ °C}$ ($z = -43.16$); Oslo dropped to $-7.3\text{ °C}$ in mid-August.
- **Engineering Verdict:** A simultaneous global drop of 5.3 °C across multiple continents is meteorologically impossible. This represents an upstream sensor batching or timezone ingestion glitch in the repository.
- **Intellectual Honesty:** Rather than deleting this date after inspecting scores, it was retained in Fold 5. It contributed to the error of all models equally, showcasing the resilience of ElasticNet and Ridge under real-world dirty data conditions.

![Figure 17: August 2026 Anomaly Analysis](figures/fig_17_anomaly_dates_and_event.png)
*Figure 17: Station-level breakdown of the August 17, 2026 recording event.*

---

## 6. Climate Classification, Geospatial Distribution & Feature Importance

### 6.1 Climate Zone Partitioning
Using annual temperature means and seasonal variance, core cities were clustered into 5 distinct macro-climatic regimes:
1. **Tropical (55 cities):** Mean 27.24 °C, minimal seasonal amplitude (4.2 °C), high humidity.
2. **Arid / Semi-Arid (33 cities):** Mean 25.85 °C, high diurnal range, low cloud cover, severe dust exposure.
3. **Temperate (54 cities):** Mean 16.53 °C, moderate seasonal amplitude (16.8 °C), balanced precipitation.
4. **Continental (39 cities):** Mean 10.37 °C, severe seasonal amplitude (31.4 °C), sub-zero winters.
5. **Polar / Subpolar (2 cities):** Mean 6.90 °C, prolonged sub-zero periods, low humidity.

![Figure 20: Global Climate Zone Map](figures/fig_20_climate_zone_map.png)
*Figure 20: Equirectangular world map plotting core reporting stations categorized by climate classification.*

![Figure 21: Climate Zone Profiles](figures/fig_21_climate_zone_profiles.png)
*Figure 21: Boxplots and monthly annual cycle distributions across the 5 macro-climate regimes.*

### 6.2 Worldwide PM2.5 Pollution Mapping
Geospatial visualization reveals stark air-quality inequalities. Metropolitan centers across North Africa, the Middle East, and South Asia frequently exceed 100 on the PM2.5 scale, while maritime and high-latitude capitals (e.g. Honiara, Reykjavik) maintain baseline readings below 10.

![Figure 22: Worldwide PM2.5 Map](figures/fig_22_pm25_city_map.png)
*Figure 22: Global PM2.5 distribution map highlighting industrial and geographic pollution hotspots.*

![Figure 23: Air Quality Pollutant Dynamics](figures/fig_23_air_quality_patterns.png)
*Figure 23: Co-pollutant interactions between PM2.5, PM10, Nitrogen Dioxide, and Sulphur Dioxide.*

### 6.3 Feature Importance Attribution
Evaluating feature importance across both **Mean Decrease in Impurity (MDI)** and **Permutation Feature Importance** provides consistent insight:
- **Short-Term Autoregressive Memory:** The 7-day moving average (`rolling_mean_7`) and the 1-day lag (`lag_1`) account for over **56% of total feature importance**.
- **Cyclical Solar Harmonics:** The 1st-order annual Fourier sine and cosine components (`sin_1`, `cos_1`) provide approximately **22% of predictive importance**, anchoring long-horizon trajectories.
- **Longer Lags:** 30-day and 365-day lags contribute marginal supplementary signals (~8%), confirming that macroscopic thermal inertia dominates daily variance.

![Figure 24: Feature Importance Comparison](figures/fig_24_feature_importance.png)
*Figure 24: Relative importance rankings comparing Tree MDI against Permutation Importance.*

---

## 7. Limitations & Data Governance

1. **Panel Representation vs True Earth Temperature:** The equal-weighted average of 183 cities is an index of metropolitan weather stations, not an area-weighted global surface temperature (such as NASA GISTEMP or ECMWF ERA5).
2. **Short Temporal Baseline (880 Days):** Covering May 2024 to October 2026, the dataset captures short-term weather persistence and seasonal cycles, but cannot be used to infer multi-decadal climate change.
3. **Data Quality Shifts:** As detected in Phase F, PM2.5 reported values underwent a systematic upward level shift in September 2024, indicating an unannounced source API methodology change.
4. **Single-Time Ingestion:** Daily records represent instantaneous snapshots rather than 24-hour integrated means.

---

## 8. Strategic Product & AI Engineering Recommendations

For organizations like **Product Manager Accelerator** and enterprise tech platforms deploying weather intelligence applications:

1. **Automated Upstream Anomaly Filtration:** Production pipelines must ingest telemetry through real-time Isolation Forest or robust Z-score filters. Single-day swings exceeding $4\sigma$ across multiple concurrent stations should trigger a pipeline hold rather than polluting downstream features.
2. **Adaptive Ensemble Architecture:** While ElasticNet achieved the lowest MAE (0.4705 °C), blending ElasticNet with Ridge (0.4740 °C) provided superior variance reduction and stability during volatile meteorological transitions.
3. **Localized Sub-Model Routing:** Rather than relying exclusively on a single global model, production applications should route forecasts through climate-zone specialized models (e.g. specialized tropical models optimized for humidity vs continental models tuned for seasonal swings).

---

## 9. Conclusion

This study successfully fulfilled all requirements of the AI Engineer Intern assessment. By combining rigorous data engineering, exploratory statistical analysis, zero-leakage walk-forward forecasting, multivariate anomaly detection, and publication-quality visualization, the project demonstrates how production AI systems can turn noisy, real-world meteorological datasets into actionable predictive intelligence.

### Repository Artifact Index:
- **Cleaned Data:** `data/processed/weather_clean_rows.csv.gz`, `weather_daily_global_core.csv`
- **Jupyter Notebooks:** `01_cleaning.ipynb`, `02_eda.ipynb`, `03_forecasting.ipynb`, `04_advanced.ipynb`
- **Figures:** 24 high-resolution charts in `reports/figures/` (`fig_01` through `fig_24`)
- **Executive Summaries:** `reports/eda_summary.md`, `reports/forecast_summary.md`, `reports/advanced_summary.md`

