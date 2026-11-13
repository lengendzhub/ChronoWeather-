"""Phase E: anomaly detection and extended forecast-model comparison.

This module contains the reusable pieces for the advanced phase:

* trailing robust temperature anomalies at the global and city-day level;
* within-city standardised weather features and an Isolation Forest;
* ElasticNetCV and HistGradientBoosting forecasters on the Phase D feature set;
* equal-weight and median ensembles evaluated on the same chronological folds.

ElasticNet regularisation is selected with an inner time-series split inside
each outer training fold, so the test window is never used for alpha selection.

The anomaly methods are descriptive. No anomaly is removed from the forecast
evaluation in this module; the already-documented 2026-08-17 event remains in
fold 5.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import core_city_days, load_eda_data  # noqa: E402
from src.forecasting import (  # noqa: E402
    FEATURE_COLUMNS,
    build_feature_frame,
    feature_row,
    make_folds,
    metrics,
    run_backtest,
)

CORE_WEATHER_FEATURES = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "cloud",
    "precip_mm",
    "visibility_km",
    "uv_index",
]
ANOMALY_WINDOW = 31
ANOMALY_MIN_PERIODS = 20
ANOMALY_Z_THRESHOLD = 4.0
IF_CONTAMINATION = 0.01
RANDOM_STATE = 42

ADVANCED_MODEL_NAMES = ["elastic_net", "hist_gradient_boosting"]
ENSEMBLE_NAMES = [
    "ensemble_ridge_enet",
    "ensemble_all_equal",
    "ensemble_all_median",
]

FEATURE_LABELS = {
    "sin_1": "Sin 1 (annual)",
    "cos_1": "Cos 1 (annual)",
    "sin_2": "Sin 2 (semiannual)",
    "cos_2": "Cos 2 (semiannual)",
    "sin_3": "Sin 3 (four-month)",
    "cos_3": "Cos 3 (four-month)",
    "lag_1": "Lag 1 day",
    "lag_2": "Lag 2 days",
    "lag_3": "Lag 3 days",
    "lag_7": "Lag 7 days",
    "lag_14": "Lag 14 days",
    "lag_30": "Lag 30 days",
    "lag_365": "Lag 365 days",
    "rolling_mean_7": "Rolling mean 7 days",
    "rolling_mean_30": "Rolling mean 30 days",
}
FEATURE_GROUPS = {
    "Annual Fourier": [
        "sin_1",
        "cos_1",
        "sin_2",
        "cos_2",
        "sin_3",
        "cos_3",
    ],
    "Recent lags (1-3 d)": ["lag_1", "lag_2", "lag_3"],
    "Weekly/monthly lags (7-30 d)": ["lag_7", "lag_14", "lag_30"],
    "Annual lag (365 d)": ["lag_365"],
    "Rolling means (7/30 d)": ["rolling_mean_7", "rolling_mean_30"],
}


def load_core_city_days() -> pd.DataFrame:
    """Load the core city-day panel used for anomaly detection."""
    city_daily, _ = load_eda_data()
    core = core_city_days(city_daily)
    return core.sort_values(["country", "city", "date_local"]).reset_index(drop=True)


def rolling_robust_z(
    series: pd.Series,
    window: int = ANOMALY_WINDOW,
    min_periods: int = ANOMALY_MIN_PERIODS,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Trailing robust z-score using only prior observations.

    The baseline is the median of the previous ``window`` values, and the
    scale is ``1.4826 * MAD`` of those same prior values. The current value is
    never included in its own baseline, which keeps the score causal.
    """
    past = series.shift(1)
    baseline = past.rolling(window, min_periods=min_periods).median()
    mad = (past - baseline).abs().rolling(window, min_periods=min_periods).median()
    scale = (1.4826 * mad).replace(0, np.nan)
    z = (series - baseline) / scale
    return z, baseline, scale


def global_temperature_anomalies(
    frame: pd.DataFrame,
    threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """Flag unusual days in the observed global core series."""
    observed = frame.loc[frame["observed"], "target"].copy()
    z, baseline, scale = rolling_robust_z(observed)
    result = pd.DataFrame(
        {
            "target": observed,
            "rolling_median": baseline,
            "rolling_scale": scale,
            "z": z,
        }
    )
    result.index.name = "date_utc"
    result["anomaly"] = result["z"].abs().ge(threshold)
    return result


def city_temperature_anomalies(
    core: pd.DataFrame,
    threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """Flag unusual city-day temperatures relative to each city's recent past."""
    records: list[pd.DataFrame] = []
    for _, sub in core.groupby(["country", "city"], sort=False):
        sub = sub.sort_values("date_local").copy()
        z, baseline, scale = rolling_robust_z(sub["temperature_celsius"])
        sub["temp_rolling_median"] = baseline
        sub["temp_rolling_scale"] = scale
        sub["temp_z"] = z
        sub["temp_anomaly"] = z.abs().ge(threshold)
        records.append(
            sub[
                [
                    "country",
                    "city",
                    "date_local",
                    "temperature_celsius",
                    "temp_rolling_median",
                    "temp_rolling_scale",
                    "temp_z",
                    "temp_anomaly",
                ]
            ]
        )
    return pd.concat(records, ignore_index=True)


def within_city_robust_zscores(
    core: pd.DataFrame,
    features: Sequence[str] = CORE_WEATHER_FEATURES,
) -> pd.DataFrame:
    """Standardise each weather feature within each city using median/MAD.

    This removes cross-city climate differences before the multivariate
    anomaly model. It is a descriptive transformation over the available
    history, not a train/test feature used by the forecast models.
    """
    records: list[pd.DataFrame] = []
    for _, sub in core.groupby(["country", "city"], sort=False):
        sub = sub.copy()
        for column in features:
            values = pd.to_numeric(sub[column], errors="coerce")
            median = values.median()
            mad = (values - median).abs().median()
            scale = 1.4826 * mad
            if not np.isfinite(scale) or scale <= 0:
                sub[f"z_{column}"] = 0.0
            else:
                sub[f"z_{column}"] = (values - median) / scale
        records.append(sub[["country", "city", "date_local", *features, *[f"z_{c}" for c in features]]])
    return pd.concat(records, ignore_index=True)


def city_multivariate_anomalies(
    core: pd.DataFrame,
    features: Sequence[str] = CORE_WEATHER_FEATURES,
    contamination: float = IF_CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, IsolationForest]:
    """Isolation Forest on within-city standardised weather combinations."""
    standardised = within_city_robust_zscores(core, features)
    z_columns = [f"z_{column}" for column in features]
    matrix = standardised[z_columns].fillna(0.0)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(matrix)
    scores = model.decision_function(matrix)
    flags = model.predict(matrix) == -1
    result = standardised[["country", "city", "date_local", *z_columns]].copy()
    result["if_score"] = scores
    result["if_anomaly"] = flags
    return result, model


def combine_anomaly_flags(
    city_temperature: pd.DataFrame,
    city_multivariate: pd.DataFrame,
) -> pd.DataFrame:
    """Merge temperature and multivariate flags into one city-day table."""
    merged = city_temperature.merge(
        city_multivariate[
            ["country", "city", "date_local", "if_score", "if_anomaly"]
        ],
        on=["country", "city", "date_local"],
        how="inner",
    )
    merged["both_flags"] = merged["temp_anomaly"] & merged["if_anomaly"]
    merged["either_flag"] = merged["temp_anomaly"] | merged["if_anomaly"]
    return merged


def anomaly_method_counts(flags: pd.DataFrame) -> pd.DataFrame:
    """Count agreement and disagreement between the two anomaly methods."""
    both = int(flags["both_flags"].sum())
    stat_only = int((flags["temp_anomaly"] & ~flags["if_anomaly"]).sum())
    if_only = int((~flags["temp_anomaly"] & flags["if_anomaly"]).sum())
    return pd.DataFrame(
        [
            {"category": "Both methods", "city_days": both},
            {"category": "Temperature z only", "city_days": stat_only},
            {"category": "Isolation Forest only", "city_days": if_only},
        ]
    )


def daily_anomaly_counts(flags: pd.DataFrame) -> pd.DataFrame:
    """Aggregate anomaly flags by local date."""
    return (
        flags.groupby("date_local")
        .agg(
            n_cities=("country", "size"),
            temp_flags=("temp_anomaly", "sum"),
            if_flags=("if_anomaly", "sum"),
            both_flags=("both_flags", "sum"),
            either_flags=("either_flag", "sum"),
        )
        .reset_index()
        .sort_values("date_local")
        .reset_index(drop=True)
    )


def fit_elastic_net(train: pd.DataFrame) -> Pipeline:
    """Fit scaled ElasticNetCV on the Phase D lag/Fourier feature set.

    Alpha is selected by a three-split chronological inner validation on the
    outer training fold only. The outer test window is never passed to the
    selector.
    """
    features = build_feature_frame(train["target"])
    usable = features.loc[train.index, FEATURE_COLUMNS].dropna()
    if usable.empty:
        raise ValueError("No complete feature rows in the ElasticNet training fold.")
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "elastic_net",
                ElasticNetCV(
                    l1_ratio=0.5,
                    alphas=[0.01, 0.03, 0.1, 0.3, 1.0],
                    cv=TimeSeriesSplit(n_splits=3),
                    max_iter=50000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(usable, train.loc[usable.index, "target"])
    return model


def fit_hist_gradient_boosting(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Fit a small HistGradientBoosting model on the same feature set."""
    features = build_feature_frame(train["target"])
    usable = features.loc[train.index, FEATURE_COLUMNS].dropna()
    if usable.empty:
        raise ValueError("No complete feature rows in the HGB training fold.")
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=RANDOM_STATE,
    )
    model.fit(usable, train.loc[usable.index, "target"])
    return model


def forecast_recursive(
    model,
    history: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Recursive multi-step forecast using the shared feature-row helper."""
    work = history.copy()
    predictions: dict[pd.Timestamp, float] = {}
    for date in dates:
        row = feature_row(pd.Timestamp(date), work)
        if row.isna().any(axis=None):
            raise ValueError(f"Missing recursive features for {date.date()}.")
        prediction = float(model.predict(row)[0])
        predictions[pd.Timestamp(date)] = prediction
        work.loc[pd.Timestamp(date)] = prediction
    return pd.Series(predictions)


def run_advanced_backtest(
    frame: pd.DataFrame,
    n_folds: int = 6,
    horizon: int = 30,
) -> tuple[pd.DataFrame, list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
    """Run Phase D base models plus ElasticNet, HGB, and fixed ensembles."""
    base_predictions, folds = run_backtest(frame, n_folds=n_folds, horizon=horizon)
    records: list[dict[str, object]] = []

    for fold_number, (train_index, test_index) in enumerate(folds, start=1):
        train = frame.loc[train_index].copy()
        test = frame.loc[test_index].copy()
        history = train["target"].copy()
        origin = train.index.max()

        advanced_models = {
            "elastic_net": fit_elastic_net(train),
            "hist_gradient_boosting": fit_hist_gradient_boosting(train),
        }
        selected_alpha = float(
            advanced_models["elastic_net"].named_steps["elastic_net"].alpha_
        )
        for model_name, model in advanced_models.items():
            prediction = forecast_recursive(model, history, test_index)
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
                        "alpha": selected_alpha if model_name == "elastic_net" else np.nan,
                        "observed": bool(test.loc[date, "observed"]),
                    }
                )

    advanced_predictions = pd.DataFrame.from_records(records)
    all_predictions = pd.concat(
        [base_predictions, advanced_predictions],
        ignore_index=True,
    )

    wide = all_predictions.pivot_table(
        index=["fold", "date"],
        columns="model",
        values="prediction",
    )
    actual = (
        all_predictions.drop_duplicates(["fold", "date"])
        .set_index(["fold", "date"])["actual"]
    )

    base_models = [
        "ridge_fourier",
        "sarimax_fourier",
        "elastic_net",
        "hist_gradient_boosting",
    ]
    ensembles = {
        "ensemble_ridge_enet": wide[["ridge_fourier", "elastic_net"]].mean(axis=1),
        "ensemble_all_equal": wide[base_models].mean(axis=1),
        "ensemble_all_median": wide[base_models].median(axis=1),
    }

    metadata = (
        all_predictions.drop_duplicates(["fold", "date"])
        .set_index(["fold", "date"])[
            ["origin", "horizon", "actual", "observed"]
        ]
    )
    ensemble_frames = []
    for model_name, values in ensembles.items():
        frame_out = metadata.copy()
        frame_out["model"] = model_name
        frame_out["prediction"] = values.reindex(frame_out.index)
        ensemble_frames.append(frame_out.reset_index())

    all_predictions = pd.concat(
        [all_predictions, *ensemble_frames],
        ignore_index=True,
    )
    return all_predictions, folds


def summarize_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics for every base model and ensemble, sorted by MAE."""
    rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        values = metrics(group["actual"], group["prediction"])
        values["model"] = model_name
        rows.append(values)
    columns = ["model", "n", "mae", "rmse", "mape_pct", "smape_pct"]
    return pd.DataFrame(rows)[columns].sort_values("mae").reset_index(drop=True)


def feature_importance_tables(
    frame: pd.DataFrame,
    folds: Sequence[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    n_repeats: int = 20,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Compute coefficient and permutation-importance tables across folds.

    The coefficients come from the already-selected ElasticNet model; because
    the pipeline standardises its inputs, they are comparable across features.
    Permutation importance is measured separately for ElasticNet and
    HistGradientBoosting on **one-step-ahead** test rows: at date *t* the model
    sees lagged values from before *t* and predicts the observed target at *t*.
    It is therefore an interpretation of the feature set, not the recursive
    30-day backtest metric reported in Phase E.

    Grouped permutation importance permutes all columns in a feature family
    together. This is less affected by collinearity between individual lags,
    but it is still a descriptive model-sensitivity measure, not a causal
    effect.
    """
    features = build_feature_frame(frame["target"])
    coefficient_rows: list[dict[str, object]] = []
    individual_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []

    fitters = {
        "elastic_net": fit_elastic_net,
        "hist_gradient_boosting": fit_hist_gradient_boosting,
    }

    for fold_number, (train_index, test_index) in enumerate(folds, start=1):
        train = frame.loc[train_index].copy()
        X_test = features.loc[test_index, FEATURE_COLUMNS].dropna()
        if X_test.empty:
            raise ValueError(f"No complete one-step test rows in fold {fold_number}.")
        y_test = frame.loc[X_test.index, "target"]

        for model_name, fitter in fitters.items():
            model = fitter(train)
            baseline_mae = float(np.mean(np.abs(y_test.to_numpy() - model.predict(X_test))))
            baseline_rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "n": int(len(X_test)),
                    "one_step_mae": baseline_mae,
                }
            )

            if model_name == "elastic_net":
                coefficients = model.named_steps["elastic_net"].coef_
                for feature, value in zip(FEATURE_COLUMNS, coefficients):
                    coefficient_rows.append(
                        {
                            "fold": fold_number,
                            "feature": feature,
                            "coefficient": float(value),
                        }
                    )

            result = permutation_importance(
                model,
                X_test,
                y_test,
                scoring="neg_mean_absolute_error",
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=1,
            )
            for feature, mean_value, std_value in zip(
                FEATURE_COLUMNS,
                result.importances_mean,
                result.importances_std,
            ):
                individual_rows.append(
                    {
                        "fold": fold_number,
                        "model": model_name,
                        "feature": feature,
                        "importance": float(mean_value),
                        "importance_within_fold_sd": float(std_value),
                    }
                )

            rng = np.random.default_rng(random_state + fold_number)
            for group_name, columns in FEATURE_GROUPS.items():
                deltas = []
                for _ in range(n_repeats):
                    perturbed = X_test.copy()
                    for column in columns:
                        perturbed[column] = rng.permutation(
                            perturbed[column].to_numpy()
                        )
                    permuted_mae = float(
                        np.mean(
                            np.abs(
                                y_test.to_numpy() - model.predict(perturbed)
                            )
                        )
                    )
                    deltas.append(permuted_mae - baseline_mae)
                group_rows.append(
                    {
                        "fold": fold_number,
                        "model": model_name,
                        "group": group_name,
                        "importance": float(np.mean(deltas)),
                        "importance_within_fold_sd": float(np.std(deltas, ddof=1)),
                    }
                )

    return {
        "coefficients": pd.DataFrame(coefficient_rows),
        "individual": pd.DataFrame(individual_rows),
        "groups": pd.DataFrame(group_rows),
        "baselines": pd.DataFrame(baseline_rows),
    }


def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Mean standardised coefficient and fold-to-fold variability per feature."""
    summary = (
        coefficients.groupby("feature")
        .agg(
            mean_coef=("coefficient", "mean"),
            sd_coef=("coefficient", "std"),
            mean_abs_coef=("coefficient", lambda values: np.mean(np.abs(values))),
            nonzero_share=(
                "coefficient",
                lambda values: np.mean(np.abs(values) > 1e-10),
            ),
        )
        .reset_index()
        .sort_values("mean_abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def summarize_feature_importance(
    importance: pd.DataFrame,
    group_column: str = "feature",
) -> pd.DataFrame:
    """Mean and fold-to-fold standard deviation of permutation importance."""
    summary = (
        importance.groupby(group_column)
        .agg(
            mean_importance=("importance", "mean"),
            sd_importance=("importance", "std"),
            n_folds=("importance", "size"),
            mean_within_fold_sd=("importance_within_fold_sd", "mean"),
        )
        .reset_index()
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    return summary
