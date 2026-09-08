"""Reliability statistics for lifetime, censoring, and accelerated tests."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from semiyield.reliability.data import validate_lifetime_data

BOLTZMANN_EV_PER_K = 8.617333262145e-5
DEFAULT_USE_TEMPERATURES_C = (55.0, 85.0, 105.0, 125.0)
RSF_BASELINE_FEATURES = (
    "baseline_temperature_c_mean",
    "baseline_rds_on_ohm_mean",
    "baseline_rds_on_ohm_slope",
)
RSF_FEATURE_ALIASES = {
    "baseline_temperature_c_mean": (
        "baseline_temperature_c_mean",
        "temperature_c",
    ),
    "baseline_rds_on_ohm_mean": (
        "baseline_rds_on_ohm_mean",
        "initial_rds_on_ohm",
    ),
    "baseline_rds_on_ohm_slope": (
        "baseline_rds_on_ohm_slope",
        "rds_on_slope",
    ),
}
RSF_UNAVAILABLE_STATUS = "required_but_unavailable"
RSF_PARAMETERS = {
    "n_estimators": 200,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass(frozen=True)
class WeibullResult:
    beta: float
    eta: float
    b10: float
    failures: int
    censored: int
    log_likelihood: float
    beta_ci95: tuple[float, float]
    eta_ci95: tuple[float, float]


@dataclass(frozen=True)
class ArrheniusWeibullResult:
    beta: float
    intercept: float
    activation_energy_ev: float
    use_temperature_c: float
    eta_at_use: float
    b10_at_use: float
    failures: int
    censored: int
    observed_temperature_range_c: tuple[float, float]
    extrapolation_warning: str


def _weibull_negative_log_likelihood(parameters, time, event) -> float:
    log_beta, log_eta = parameters
    beta, eta = np.exp(log_beta), np.exp(log_eta)
    scaled = np.clip(time / eta, 1e-15, 1e15)
    log_survival = -(scaled**beta)
    log_density = log_beta - log_eta + (beta - 1) * np.log(scaled) + log_survival
    return float(-np.sum(np.where(event, log_density, log_survival)))


def fit_weibull(frame: pd.DataFrame) -> WeibullResult:
    data = validate_lifetime_data(frame)
    time = data["time_to_event"].to_numpy(dtype=float)
    event = data["event_observed"].to_numpy(dtype=bool)
    initial_eta = float(np.median(time[event]))
    optimized = minimize(
        _weibull_negative_log_likelihood,
        x0=np.log([1.5, initial_eta]),
        args=(time, event),
        method="L-BFGS-B",
    )
    if not optimized.success:
        raise RuntimeError(f"Weibull fit failed: {optimized.message}")
    beta, eta = np.exp(optimized.x)
    covariance = np.asarray(optimized.hess_inv.todense())
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    lower = np.exp(optimized.x - 1.96 * standard_error)
    upper = np.exp(optimized.x + 1.96 * standard_error)
    b10 = eta * (-np.log(0.9)) ** (1 / beta)
    return WeibullResult(
        beta=float(beta),
        eta=float(eta),
        b10=float(b10),
        failures=int(event.sum()),
        censored=int((~event).sum()),
        log_likelihood=float(-optimized.fun),
        beta_ci95=(float(lower[0]), float(upper[0])),
        eta_ci95=(float(lower[1]), float(upper[1])),
    )


def _arrhenius_negative_log_likelihood(parameters, time, event, inverse_kelvin) -> float:
    log_beta, intercept, coefficient = parameters
    beta = np.exp(log_beta)
    log_eta = intercept + coefficient * inverse_kelvin
    scaled_log = np.log(time) - log_eta
    cumulative_hazard = np.exp(np.clip(beta * scaled_log, -700, 700))
    log_density = log_beta - log_eta + (beta - 1) * scaled_log - cumulative_hazard
    log_survival = -cumulative_hazard
    return float(-np.sum(np.where(event, log_density, log_survival)))


def fit_arrhenius_weibull(
    frame: pd.DataFrame, *, use_temperature_c: float = 55.0
) -> ArrheniusWeibullResult:
    beta, intercept, coefficient, data = _fit_arrhenius_parameters(frame)
    return _arrhenius_result(
        beta, intercept, coefficient, data, use_temperature_c=use_temperature_c
    )


def _fit_arrhenius_parameters(frame: pd.DataFrame):
    data = validate_lifetime_data(frame, require_temperature=True)
    time = data["time_to_event"].to_numpy(dtype=float)
    event = data["event_observed"].to_numpy(dtype=bool)
    temperatures = data["temperature_c"].to_numpy(dtype=float)
    inverse_kelvin = 1 / (temperatures + 273.15)
    initial = [np.log(1.5), np.log(np.median(time[event])), 5000.0]
    optimized = minimize(
        _arrhenius_negative_log_likelihood,
        x0=initial,
        args=(time, event, inverse_kelvin),
        method="L-BFGS-B",
    )
    if not optimized.success:
        raise RuntimeError(f"Arrhenius-Weibull fit failed: {optimized.message}")
    beta = float(np.exp(optimized.x[0]))
    intercept, coefficient = map(float, optimized.x[1:])
    return beta, intercept, coefficient, data


def _arrhenius_result(beta, intercept, coefficient, data, *, use_temperature_c: float):
    temperatures = data["temperature_c"].to_numpy(dtype=float)
    event = data["event_observed"].to_numpy(dtype=bool)
    inverse_use = 1 / (use_temperature_c + 273.15)
    eta_use = float(np.exp(intercept + coefficient * inverse_use))
    b10_use = float(eta_use * (-np.log(0.9)) ** (1 / beta))
    observed_range = (float(temperatures.min()), float(temperatures.max()))
    warning = ""
    if not observed_range[0] <= use_temperature_c <= observed_range[1]:
        warning = (
            "Use temperature is outside the observed stress range; lifetime is an "
            "extrapolation that requires mechanism validation."
        )
    return ArrheniusWeibullResult(
        beta=beta,
        intercept=intercept,
        activation_energy_ev=float(coefficient * BOLTZMANN_EV_PER_K),
        use_temperature_c=float(use_temperature_c),
        eta_at_use=eta_use,
        b10_at_use=b10_use,
        failures=int(event.sum()),
        censored=int((~event).sum()),
        observed_temperature_range_c=observed_range,
        extrapolation_warning=warning,
    )


def sweep_arrhenius_weibull(
    frame: pd.DataFrame, *, use_temperatures_c=DEFAULT_USE_TEMPERATURES_C
) -> list[dict[str, object]]:
    """Evaluate one Arrhenius-Weibull fit at several declared use temperatures."""
    temperatures = tuple(float(value) for value in use_temperatures_c)
    if not temperatures or not all(np.isfinite(temperatures)):
        raise ValueError("use_temperatures_c must contain finite temperatures")
    beta, intercept, coefficient, data = _fit_arrhenius_parameters(frame)
    rows = []
    for temperature in temperatures:
        result = _arrhenius_result(
            beta, intercept, coefficient, data, use_temperature_c=temperature
        )
        within_fitted_range = not result.extrapolation_warning
        conservative_warning = result.extrapolation_warning or (
            "Use temperature lies within the fitted stress range but is reported as a declared "
            "use-temperature extrapolation; lifetime is a model estimate, not a direct observation."
        )
        rows.append(
            {
                "use_temperature_c": result.use_temperature_c,
                "b10_s": result.b10_at_use,
                "eta_s": result.eta_at_use,
                "support_status": "extrapolated",
                "fitted_temperature_range_status": (
                    "within_fitted_range" if within_fitted_range else "outside_fitted_range"
                ),
                "extrapolation_warning": conservative_warning,
                "beta": result.beta,
                "activation_energy_ev": result.activation_energy_ev,
                "supported_temperature_min_c": result.observed_temperature_range_c[0],
                "supported_temperature_max_c": result.observed_temperature_range_c[1],
            }
        )
    return rows


def survival_probability(time, *, beta: float, eta: float) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    return np.exp(-((values / eta) ** beta))


def resolve_rsf_baseline_features(
    frame: pd.DataFrame,
) -> tuple[dict[str, str], list[str]]:
    """Resolve only documented baseline-feature aliases for the RSF contract."""
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in RSF_FEATURE_ALIASES.items():
        source = next((column for column in aliases if column in frame), None)
        if source is None:
            missing.append(canonical)
        else:
            mapping[canonical] = source
    return mapping, missing


def _rsf_unavailable(reason: str, context: dict[str, object]) -> dict[str, object]:
    return {"status": RSF_UNAVAILABLE_STATUS, "reason": reason, **context}


def _individual_prediction(survival_functions, unit_ids: pd.Series) -> dict[str, object]:
    """Keep held-out RSF survival predictions in memory for the public device plot."""
    curves = []
    for unit_id, function in zip(unit_ids.astype(str), survival_functions, strict=True):
        time = function.x
        time = np.insert(time, 0, 0.0) if time[0] > 0 else time
        curves.append(
            {
                "unit_id": unit_id,
                "_time_s": time.tolist(),
                "_survival_probability": function(time).tolist(),
            }
        )
    return {
        "status": "completed",
        "test_units": int(len(curves)),
        "curves": curves,
    }


def benchmark_survival_forest(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    data_sha256: str | None = None,
) -> dict[str, object]:
    """Attempt the required device-level RSF using the baseline-feature contract."""
    data = validate_lifetime_data(frame)
    feature_mapping, missing = resolve_rsf_baseline_features(data)
    context = {
        "required": True,
        "feature_columns": list(RSF_BASELINE_FEATURES),
        "feature_source_columns": feature_mapping,
        "model_parameters": {**RSF_PARAMETERS, "random_state": random_state},
        "data_sha256": data_sha256,
    }
    if missing:
        return _rsf_unavailable(
            "Missing baseline survival features for required RSF: " + ", ".join(missing), context
        )
    if len(data) < 30 or int(data["event_observed"].sum()) < 10:
        return _rsf_unavailable("At least 30 units and 10 observed failures are required", context)
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import (
            concordance_index_censored,
            concordance_index_ipcw,
            integrated_brier_score,
        )
    except ImportError:
        return _rsf_unavailable(
            "scikit-survival is required; run `uv sync --locked --extra survival`", context
        )
    if "split" in data:
        split_values = set(data["split"].dropna())
        if not {"train", "test"}.issubset(split_values):
            return _rsf_unavailable(
                "Input split must include both train and test devices", context
            )
        train_idx = np.flatnonzero(data["split"].isin(["train", "validation"]).to_numpy())
        test_idx = np.flatnonzero(data["split"].eq("test").to_numpy())
        split_source = "input_device_manifest"
    else:
        devices = data["unit_id"].astype(str).to_numpy()
        order = np.argsort(devices)
        boundary = int(len(order) * 0.8)
        train_idx, test_idx = order[:boundary], order[boundary:]
        split_source = "deterministic_unit_order"
    if not len(train_idx) or not len(test_idx):
        return _rsf_unavailable("Split contains no train or test units", context)
    if int(data.iloc[train_idx]["event_observed"].sum()) < 10:
        return _rsf_unavailable("Training split has fewer than 10 observed failures", context)
    try:
        x = data[[feature_mapping[column] for column in RSF_BASELINE_FEATURES]].apply(
            pd.to_numeric, errors="raise"
        )
        x.columns = RSF_BASELINE_FEATURES
    except (TypeError, ValueError) as exc:
        return _rsf_unavailable(str(exc), context)
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        return _rsf_unavailable("Baseline survival features must be finite", context)
    outcome = np.array(
        list(zip(data["event_observed"], data["time_to_event"], strict=True)),
        dtype=[("event", bool), ("time", float)],
    )
    model = RandomSurvivalForest(
        **{**RSF_PARAMETERS, "random_state": random_state}
    )
    model.fit(x.iloc[train_idx], outcome[train_idx])
    risk = model.predict(x.iloc[test_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c_index_value = concordance_index_censored(
            outcome[test_idx]["event"], outcome[test_idx]["time"], risk
        )[0]
    c_index = float(c_index_value) if np.isfinite(c_index_value) else None
    c_index_note = "" if c_index is not None else "No comparable test-set pairs"
    rng = np.random.default_rng(random_state)
    bootstrap = []
    for _ in range(1000):
        sampled = rng.integers(0, len(test_idx), len(test_idx))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                value = concordance_index_censored(
                    outcome[test_idx]["event"][sampled],
                    outcome[test_idx]["time"][sampled],
                    risk[sampled],
                )[0]
            if np.isfinite(value):
                bootstrap.append(float(value))
        except ValueError:
            continue
    c_index_ci = (
        [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
        if bootstrap
        else None
    )
    uno_c_index: float | None = None
    uno_c_index_note = ""
    try:
        uno_value = concordance_index_ipcw(outcome[train_idx], outcome[test_idx], risk)[0]
        uno_c_index = float(uno_value) if np.isfinite(uno_value) else None
        if uno_c_index is None:
            uno_c_index_note = "No comparable IPCW test-set pairs"
    except ValueError as exc:
        uno_c_index_note = str(exc)
    survival_functions = model.predict_survival_function(x.iloc[test_idx])
    individual_prediction = _individual_prediction(
        survival_functions, data.iloc[test_idx]["unit_id"]
    )
    predicted_median = []
    for function in survival_functions:
        values = function(function.x)
        positions = np.flatnonzero(values <= 0.5)
        predicted_median.append(
            float(function.x[positions[0]]) if len(positions) else float(function.x[-1])
        )
    observed_test = outcome[test_idx]["event"]
    lifetime_mae = (
        float(
            np.mean(
                np.abs(
                    np.asarray(predicted_median)[observed_test]
                    - outcome[test_idx]["time"][observed_test]
                )
            )
        )
        if observed_test.any()
        else None
    )
    ibs: float | None = None
    ibs_reason = ""
    lower = max(float(outcome[train_idx]["time"].min()), 1e-9)
    upper = min(
        float(outcome[train_idx]["time"].max()),
        float(outcome[test_idx]["time"].max()),
    )
    try:
        if upper > lower and float(outcome[test_idx]["time"].max()) < float(
            outcome[train_idx]["time"].max()
        ):
            times = np.linspace(lower, upper, 50, endpoint=False)[1:]
            probabilities = np.vstack([function(times) for function in survival_functions])
            ibs = float(
                integrated_brier_score(outcome[train_idx], outcome[test_idx], probabilities, times)
            )
        else:
            ibs_reason = "Test follow-up exceeds the training censoring range"
    except ValueError as exc:
        ibs_reason = str(exc)
    return {
        "status": "completed",
        **context,
        "c_index": c_index,
        "c_index_note": c_index_note,
        "c_index_bootstrap_ci95": c_index_ci,
        "c_index_bootstrap_valid_resamples": len(bootstrap),
        "uno_c_index": uno_c_index,
        "uno_c_index_note": uno_c_index_note,
        "integrated_brier_score": ibs,
        "integrated_brier_score_note": ibs_reason,
        "observed_failure_median_lifetime_mae": lifetime_mae,
        "train_units": len(train_idx),
        "test_units": len(test_idx),
        "split_source": split_source,
        "_individual_prediction": individual_prediction,
    }
