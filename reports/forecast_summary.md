# Phase D — Forecast summary

**Status:** Complete for review. Phase D produced an executed forecasting
notebook, four forecast figures, and this summary. No Phase E or Phase F result
is included.

## Target and evaluation

- **Target:** equal-weighted daily mean temperature across the 183 core
  locations (`temperature_celsius`).
- **Calendar coverage:** 881 daily dates from 2024-05-16
  to 2026-10-13; 875 observed dates and
  6 non-observed dates.
- **Evaluation:** 6 chronological rolling-origin folds, each with a
  30-day test window; 180 predictions per model
  (720 model-date predictions in total).
- **Test period:** 2026-04-17 through 2026-10-13.
- **Training rule:** each training set ends one day before its test window.
- **Models:** persistence, seasonal naive, Ridge + Fourier, and SARIMAX +
  Fourier. Ridge scaling is fitted inside each training fold only.

| fold | train_start | train_end | train_days | test_start | test_end | test_days | test_observed |
|---|---|---|---|---|---|---|---|
| 1 | 2024-05-16 | 2026-04-16 | 701 | 2026-04-17 | 2026-05-16 | 30 | 30 |
| 2 | 2024-05-16 | 2026-05-16 | 731 | 2026-05-17 | 2026-06-15 | 30 | 30 |
| 3 | 2024-05-16 | 2026-06-15 | 761 | 2026-06-16 | 2026-07-15 | 30 | 30 |
| 4 | 2024-05-16 | 2026-07-15 | 791 | 2026-07-16 | 2026-08-14 | 30 | 30 |
| 5 | 2024-05-16 | 2026-08-14 | 821 | 2026-08-15 | 2026-09-13 | 30 | 30 |
| 6 | 2024-05-16 | 2026-09-13 | 851 | 2026-09-14 | 2026-10-13 | 30 | 30 |

## Aggregate metrics

| Model | Predictions | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) |
|---|---|---|---|---|---|
| Ridge + Fourier | 180 | 0.501 | 0.746 | 2.303 | 2.251 |
| SARIMAX + Fourier | 180 | 0.958 | 1.194 | 4.354 | 4.210 |
| Persistence | 180 | 1.037 | 1.249 | 4.678 | 4.677 |
| Seasonal naive | 180 | 1.851 | 2.014 | 8.413 | 8.004 |

**Result:** `Ridge + Fourier` had the lowest aggregate MAE, **0.501 °C**
(RMSE **0.746 °C**, MAPE **2.303%**,
sMAPE **2.251%**). Persistence had MAE
**1.037 °C**, so the best model's MAE was
**51.7% lower** in this backtest. SARIMAX + Fourier had MAE
**0.958 °C**.
`Ridge + Fourier` also had the lowest per-fold MAE in **6 of
6 folds**.

## Per-fold MAE

| Fold | Persistence | Seasonal naive | SARIMAX + Fourier | Ridge + Fourier |
|---|---|---|---|---|
| 1 | 0.655 | 2.335 | 1.269 | 0.624 |
| 2 | 1.609 | 1.868 | 0.665 | 0.570 |
| 3 | 1.066 | 1.940 | 0.793 | 0.496 |
| 4 | 0.717 | 1.930 | 1.328 | 0.570 |
| 5 | 1.463 | 1.904 | 1.349 | 0.449 |
| 6 | 0.711 | 1.131 | 0.345 | 0.298 |

## MAE at selected horizons

| Horizon (days) | Persistence | Seasonal naive | SARIMAX + Fourier | Ridge + Fourier |
|---|---|---|---|---|
| 1 | 0.285 | 1.926 | 0.214 | 0.214 |
| 7 | 0.907 | 1.953 | 0.824 | 0.449 |
| 14 | 1.092 | 2.032 | 0.966 | 0.453 |
| 21 | 0.952 | 1.671 | 1.072 | 0.621 |
| 30 | 1.337 | 1.842 | 1.412 | 0.722 |

## Last holdout and 14-day projection

Last test fold: **2026-09-14 to 2026-10-13**.

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) |
|---|---|---|---|---|
| Ridge + Fourier | 0.298 | 0.370 | 1.419 | 1.424 |
| SARIMAX + Fourier | 0.345 | 0.410 | 1.653 | 1.638 |
| Persistence | 0.711 | 0.812 | 3.424 | 3.353 |
| Seasonal naive | 1.131 | 1.259 | 5.329 | 5.163 |

The Ridge model was refit on all available data and projected
**2026-10-14 to 2026-10-27**.
The projected daily mean ranges from **19.51 °C** to
**20.34 °C**. The empirical 80% residual band is
**-1.08 °C to +0.37 °C** around the point forecast. This
band is descriptive, not a formal predictive distribution.

## Leakage controls

- The six test windows are chronological and non-overlapping; each training set
  ends one day before its test window.
- No test target is used in fitting. The first-fold check verifies zero
  train/test date overlap, 30/30 observed test targets, and scaler fitting on
  training rows only.
- Ridge lags and rolling means use only prior target values. During the
  recursive forecast, predictions become history; actual test values are not
  inserted.
- Fourier terms are deterministic calendar features and do not use the target.
- SARIMAX is fitted separately on each training fold. Its exogenous Fourier
  terms for the test window are known in advance.
- The six non-observed dates are forward-filled only for model continuity and
  are not evaluation targets. All six are before the first test window.

## Caveats

- The target is an equal-weighted average of 183 core locations, not a
  population- or area-weighted global temperature.
- The record spans a little over two years. It can describe seasonal behaviour
  in this panel but cannot establish a long-term climate trend.
- Coverage is uneven and readings are point-in-time values from different local
  times. The model treats the panel average as a single series and does not
  propagate coverage uncertainty.
- The largest observed one-day change is **-5.36 °C** on
  **2026-08-17**. The underlying city readings are broad and
  physically implausible as a one-day global event; it is likely a source-data
  or timestamp artefact. It is retained in fold 5 rather than removed after
  seeing model scores, so it contributes to fold-5 errors rather than being
  hidden.
- Six folds provide 180 out-of-sample predictions per model
  (720 total), but model selection used the same six folds. The
  reported score is therefore a model-selection estimate, not an untouched final
  holdout.
- Seasonal naive is a deliberately simple calendar benchmark; its poor result
  shows that last year's same-day value is not a good predictor of this series.
- The 14-day projection is post-sample and has not been evaluated against
  future observations. The empirical band is based on pooled backtest residuals
  and does not include structural-change uncertainty.
- Air quality, climate-region analysis, anomaly detection, and advanced
  feature-importance work belong to later phases and are not claimed here.

## Figures

- `reports/figures/fig_12_forecast_windows.png`
- `reports/figures/fig_13_forecast_model_metrics.png`
- `reports/figures/fig_14_forecast_error_by_horizon.png`
- `reports/figures/fig_15_holdout_and_projection.png`

## Phase D checkpoint

Phase D is complete for review. The next phase is advanced analysis (anomaly
detection, multiple-model comparison/ensembling, climate and air-quality
analysis, feature importance, and maps). Stop here for review before Phase E.
