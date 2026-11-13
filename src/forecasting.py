"""Phase D: reusable forecasting utilities for the global core series.

Design choices
--------------
* Target: daily equal-weighted mean temperature of the 183 core locations.
* Time index: UTC date. Dates with insufficient coverage (plus any calendar
  dates absent from the table) are treated as missing observations. They are
  filled forward only for model continuity and are never used as evaluation
  targets.
* Evaluation: chronological, rolling-origin backtest. Each fold trains only on
  data strictly before its 30-day test window.
* Baselines: persistence (last training value repeated) and seasonal naive
  (same calendar day one year earlier).
* Real models: Ridge regression on annual Fourier terms plus lag features, and
  a SARIMAX model with annual Fourier exogenous terms.
* No test-period information is used for fitting a scaler, feature set, or
  hyperparameter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GLOBAL_DAILY_CSV = PROJECT_ROOT / "data" / "processed" / "weather_daily_global_core.csv"
TARGET_COLUMN = "temperature_celsius"
DEFAULT_HORIZON = 30
DEFAULT_FOLDS = 6
FOURIER_ORDER = 3
RIDGE_ALPHA = 1.0
MODEL_NAMES = ["persistence", "seasonal_naive", "ridge_fourier", "sarimax_fourier"]
BASELINE_NAMES = ["persistence", "seasonal_naive"]
REAL_MODEL_NAMES = ["ridge_fourier", "sarimax_fourier"]
LAGS = [1, 2, 3, 7, 14, 30, 365]
ROLLING_WINDOWS = [7, 30]


def load_global_series() -> pd.DataFrame:
    """Load the core UTC series on a complete daily index.

    Returns
    -------
    DataFrame
        Columns: ``temperature_celsius`` (raw), ``coverage_ok``, ``observed``,
        ``target``. ``target`` is the observed value where coverage passes and
        is forward-filled only across non-observed dates so that the daily
        models have a continuous input series.
    """
    raw = pd.read_csv(GLOBAL_DAILY_CSV, parse_dates=["date_utc"])
    raw = raw.sort_values("date_utc").set_index("date_utc")
    raw = raw.asfreq("D")
    raw["observed"] = raw["coverage_ok"].eq(True)
    raw["target"] = raw[TARGET_COLUMN].where(raw["observed"])
    raw["target"] = raw["target"].ffill()
    if raw["target"].isna().any():
        raise ValueError("The target series contains a leading gap that ffill cannot fill.")
    return raw


def fourier_features(index: pd.DatetimeIndex, order: int = FOURIER_ORDER) -> pd.DataFrame:
    """Annual Fourier terms. They are known in advance and use no target data."""
    day_of_year = np.asarray(index.dayofyear, dtype=float)
    values: dict[str, np.ndarray] = {}
    for k in range(1, order + 1):
        angle = 2.0 * np.pi * k * (day_of_year - 1.0) / 365.25
        values[f"sin_{k}"] = np.sin(angle)
        values[f"cos_{k}"] = np.cos(angle)
    return pd.DataFrame(values, index=index)


def build_feature_frame(series: pd.Series) -> pd.DataFrame:
    """Build one-step-ahead features using only past target values.

    Each row may be used to predict the target on that row's date. Lags and
    rolling means are shifted, so the current target is never an input.
    """
    frame = series.to_frame("target").copy()
    fourier = fourier_features(frame.index)
    for column in fourier.columns:
        frame[column] = fourier[column]

    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["target"].shift(lag)

    shifted = frame["target"].shift(1)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = shifted.rolling(window).mean()

    return frame


FEATURE_COLUMNS = [
    *[f"sin_{k}" for k in range(1, FOURIER_ORDER + 1)],
    *[f"cos_{k}" for k in range(1, FOURIER_ORDER + 1)],
    *[f"lag_{lag}" for lag in LAGS],
    *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
]


def feature_row(date: pd.Timestamp, history: pd.Series) -> pd.DataFrame:
    """Create one feature row for a future date from an evolving history."""
    values: dict[str, float] = {}
    fourier = fourier_features(pd.DatetimeIndex([date]))
    for column in fourier.columns:
        values[column] = float(fourier.iloc[0][column])

    for lag in LAGS:
        lag_date = date - pd.Timedelta(days=lag)
        if lag_date in history.index:
            values[f"lag_{lag}"] = float(history.loc[lag_date])
        else:
            values[f"lag_{lag}"] = np.nan

    for window in ROLLING_WINDOWS:
        values[f"rolling_mean_{window}"] = float(history.iloc[-window:].mean())

    return pd.DataFrame([values], index=[date], columns=FEATURE_COLUMNS)


def make_folds(
    frame: pd.DataFrame,
    n_folds: int = DEFAULT_FOLDS,
    horizon: int = DEFAULT_HORIZON,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Create the most recent contiguous rolling-origin test windows.

    Fold 1 is the earliest window; the final fold ends on the last observed
    date. Each training set ends one day before its test window starts.
    """
    end = frame.index.max()
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for offset in reversed(range(n_folds)):
        test_end = end - pd.Timedelta(days=horizon * offset)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)
        train_index = frame.index[frame.index <= train_end]
        test_index = frame.index[(frame.index >= test_start) & (frame.index <= test_end)]
        if len(test_index) != horizon:
            raise ValueError(f"Fold ending {test_end.date()} has {len(test_index)} test rows.")
        folds.append((train_index, test_index))
    return folds


def fit_ridge(train: pd.DataFrame) -> Pipeline:
    """Fit scaling and Ridge only on the training fold."""
    features = build_feature_frame(train["target"])
    usable = features.loc[train.index, FEATURE_COLUMNS].dropna()
    if usable.empty:
        raise ValueError("No complete feature rows in the training fold.")
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )
    model.fit(usable, train.loc[usable.index, "target"])
    return model


def forecast_ridge(
    model: Pipeline,
    history: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Recursive multi-step forecast for the Ridge model."""
    work = history.copy()
    predictions: dict[pd.Timestamp, float] = {}
    for date in dates:
        row = feature_row(pd.Timestamp(date), work)
        if row.isna().any(axis=None):
            raise ValueError(f"Missing Ridge features for {date.date()}.")
        prediction = float(model.predict(row)[0])
        predictions[pd.Timestamp(date)] = prediction
        work.loc[pd.Timestamp(date)] = prediction
    return pd.Series(predictions, name="ridge_fourier")


def fit_sarimax(train: pd.DataFrame):
    """Fit SARIMAX with annual Fourier exogenous terms on one training fold.

    The default L-BFGS optimiser did not fully converge in one development
    fold. BFGS with a larger iteration cap converged in all six folds with no
    convergence warning, so that optimiser is used here.
    """
    exog = fourier_features(train.index)
    model = SARIMAX(
        train["target"],
        exog=exog,
        order=(1, 0, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, method="bfgs", maxiter=1000)
    if not bool(result.mle_retvals.get("converged", False)):
        raise RuntimeError("SARIMAX did not converge on this training fold.")
    return result


def forecast_sarimax(result, dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    """Forecast and return an 80% interval for the SARIMAX model."""
    exog = fourier_features(dates)
    forecast = result.get_forecast(steps=len(dates), exog=exog)
    mean = forecast.predicted_mean.copy()
    mean.index = dates
    mean.name = "sarimax_fourier"
    interval = forecast.conf_int(alpha=0.20)
    interval.index = dates
    return mean, interval


def seasonal_naive(history: pd.Series, dates: pd.DatetimeIndex, lag_days: int = 365) -> pd.Series:
    """Use the observed value from the same calendar day one year earlier."""
    values = []
    for date in dates:
        lag_date = pd.Timestamp(date) - pd.Timedelta(days=lag_days)
        if lag_date not in history.index:
            raise ValueError(f"Seasonal-naive history missing for {lag_date.date()}.")
        values.append(float(history.loc[lag_date]))
    return pd.Series(values, index=dates, name="seasonal_naive")


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean absolute percentage error, with a guard for zero targets."""
    denominator = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric MAPE in percent, robust to zero or near-zero values."""
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100.0)


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Regression metrics in target units and percent."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred)
    true = true[valid]
    pred = pred[valid]
    if len(true) == 0:
        raise ValueError("No finite pairs to evaluate.")
    return {
        "n": int(len(true)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mape_pct": mape(true, pred),
        "smape_pct": smape(true, pred),
    }


def run_backtest(
    frame: pd.DataFrame,
    n_folds: int = DEFAULT_FOLDS,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
    """Run the rolling-origin backtest and return long-form predictions."""
    folds = make_folds(frame, n_folds=n_folds, horizon=horizon)
    records: list[dict[str, object]] = []

    for fold_number, (train_index, test_index) in enumerate(folds, start=1):
        train = frame.loc[train_index].copy()
        test = frame.loc[test_index].copy()
        if not bool(test["observed"].all()):
            raise ValueError("A test window contains a non-observed target.")
        history = train["target"].copy()
        origin = train.index.max()

        predictions: dict[str, pd.Series] = {}
        predictions["persistence"] = pd.Series(
            float(train["target"].iloc[-1]),
            index=test_index,
            name="persistence",
        )
        predictions["seasonal_naive"] = seasonal_naive(history, test_index)

        ridge = fit_ridge(train)
        predictions["ridge_fourier"] = forecast_ridge(ridge, history, test_index)

        sarimax_result = fit_sarimax(train)
        sarimax_mean, _ = forecast_sarimax(sarimax_result, test_index)
        predictions["sarimax_fourier"] = sarimax_mean

        for model_name, prediction in predictions.items():
            for date in test_index:
                records.append(
                    {
                        "fold": fold_number,
                        "origin": origin,
                        "date": pd.Timestamp(date),
                        "horizon": int((pd.Timestamp(date) - origin).days),
                        "actual": float(test.loc[date, "target"]),
                        "model": model_name,
                        "prediction": float(prediction.loc[date]),
                        "observed": bool(test.loc[date, "observed"]),
                    }
                )

    predictions_df = pd.DataFrame.from_records(records)
    return predictions_df, folds


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across all rolling-origin folds."""
    rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        values = metrics(group["actual"], group["prediction"])
        values["model"] = model_name
        rows.append(values)
    columns = ["model", "n", "mae", "rmse", "mape_pct", "smape_pct"]
    return pd.DataFrame(rows)[columns].sort_values("mae").reset_index(drop=True)


def per_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics for each model and fold, useful for checking stability."""
    rows = []
    for (model_name, fold), group in predictions.groupby(["model", "fold"], sort=True):
        values = metrics(group["actual"], group["prediction"])
        values.update({"model": model_name, "fold": fold})
        rows.append(values)
    columns = ["model", "fold", "n", "mae", "rmse", "mape_pct", "smape_pct"]
    return pd.DataFrame(rows)[columns].sort_values(["model", "fold"]).reset_index(drop=True)


def horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Error by forecast horizon, aggregated across folds."""
    rows = []
    for (model_name, horizon), group in predictions.groupby(["model", "horizon"], sort=True):
        values = metrics(group["actual"], group["prediction"])
        values.update({"model": model_name, "horizon": int(horizon)})
        rows.append(values)
    columns = ["model", "horizon", "n", "mae", "rmse", "mape_pct", "smape_pct"]
    return pd.DataFrame(rows)[columns].sort_values(["model", "horizon"]).reset_index(drop=True)


def future_dates(last_date: pd.Timestamp, horizon: int = 14) -> pd.DatetimeIndex:
    """Dates strictly after the observed series end."""
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
