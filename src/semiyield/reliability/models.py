"""Reliability statistics for lifetime, censoring, and accelerated tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from semiyield.common.data import sha256_file
from semiyield.common.visuals import write_reliability_charts

BOLTZMANN_EV_PER_K = 8.617333262145e-5
RELIABILITY_COLUMNS = ("unit_id", "time_to_event", "event_observed")


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


def validate_lifetime_data(
    frame: pd.DataFrame, *, require_temperature: bool = False
) -> pd.DataFrame:
    required = list(RELIABILITY_COLUMNS)
    if require_temperature:
        required.append("temperature_c")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing reliability columns: {', '.join(missing)}")
    validated = frame.copy()
    validated["time_to_event"] = pd.to_numeric(validated["time_to_event"], errors="raise")
    if (validated["time_to_event"] <= 0).any():
        raise ValueError("time_to_event must be strictly positive")
    events = validated["event_observed"]
    if not events.isin([0, 1, False, True]).all():
        raise ValueError("event_observed must contain only 0/1 or boolean values")
    validated["event_observed"] = events.astype(bool)
    if not validated["event_observed"].any():
        raise ValueError("At least one observed failure is required")
    if validated["unit_id"].isna().any() or validated["unit_id"].astype(str).duplicated().any():
        raise ValueError("unit_id must be non-null and unique")
    if require_temperature:
        validated["temperature_c"] = pd.to_numeric(validated["temperature_c"], errors="raise")
        if validated["temperature_c"].nunique() < 2:
            raise ValueError("Arrhenius fitting requires at least two temperatures")
        if (validated["temperature_c"] <= -273.15).any():
            raise ValueError("temperature_c must be above absolute zero")
    return validated


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


def survival_probability(time, *, beta: float, eta: float) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    return np.exp(-((values / eta) ** beta))


def benchmark_survival_forest(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    random_state: int = 42,
) -> dict[str, object]:
    """Run optional device-level random survival forest, or return an auditable skip."""
    data = validate_lifetime_data(frame)
    if len(data) < 30 or int(data["event_observed"].sum()) < 10:
        return {
            "status": "skipped",
            "reason": "At least 30 units and 10 observed failures are required",
        }
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import concordance_index_censored, integrated_brier_score
    except ImportError:
        return {"status": "skipped", "reason": "Run `uv sync --locked --extra survival`"}
    missing = [column for column in feature_columns if column not in data]
    if missing:
        raise ValueError(f"Missing survival features: {', '.join(missing)}")
    if "split" in data and {"train", "test"}.issubset(set(data["split"])):
        train_idx = np.flatnonzero(data["split"].isin(["train", "validation"]).to_numpy())
        test_idx = np.flatnonzero(data["split"].eq("test").to_numpy())
        split_source = "input_manifest"
    else:
        devices = data["unit_id"].astype(str).to_numpy()
        order = np.argsort(devices)
        boundary = int(len(order) * 0.8)
        train_idx, test_idx = order[:boundary], order[boundary:]
        split_source = "deterministic_unit_order"
    if not len(train_idx) or not len(test_idx):
        return {"status": "skipped", "reason": "Split contains no train or test units"}
    x = data[feature_columns].apply(pd.to_numeric, errors="raise")
    outcome = np.array(
        list(zip(data["event_observed"], data["time_to_event"], strict=True)),
        dtype=[("event", bool), ("time", float)],
    )
    model = RandomSurvivalForest(
        n_estimators=200, min_samples_leaf=3, random_state=random_state, n_jobs=-1
    )
    model.fit(x.iloc[train_idx], outcome[train_idx])
    risk = model.predict(x.iloc[test_idx])
    c_index = concordance_index_censored(
        outcome[test_idx]["event"], outcome[test_idx]["time"], risk
    )[0]
    rng = np.random.default_rng(random_state)
    bootstrap = []
    for _ in range(1000):
        sampled = rng.integers(0, len(test_idx), len(test_idx))
        try:
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
    survival_functions = model.predict_survival_function(x.iloc[test_idx])
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
        "c_index": float(c_index),
        "c_index_bootstrap_ci95": c_index_ci,
        "integrated_brier_score": ibs,
        "integrated_brier_score_note": ibs_reason,
        "observed_failure_median_lifetime_mae": lifetime_mae,
        "train_units": len(train_idx),
        "test_units": len(test_idx),
        "split_source": split_source,
    }


def write_reliability_report(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path = "reports/reference/reliability",
    use_temperature_c: float = 55.0,
    data_sha256: str | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weibull = fit_weibull(frame)
    report: dict[str, object] = {
        "schema_version": "semiyield-reliability-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": data_sha256,
        "weibull": asdict(weibull),
    }
    if "temperature_c" in frame:
        stress_group = (pd.to_numeric(frame["temperature_c"], errors="coerce") / 10).round() * 10
        failures_by_group = (
            frame.loc[frame["event_observed"].astype(bool)].groupby(stress_group).size()
        )
        eligible_groups = failures_by_group[failures_by_group >= 3]
        if len(eligible_groups) >= 2:
            eligible = frame.loc[stress_group.isin(eligible_groups.index)]
            report["arrhenius_weibull"] = asdict(
                fit_arrhenius_weibull(eligible, use_temperature_c=use_temperature_c)
            )
        else:
            report["arrhenius_weibull"] = {
                "status": "skipped",
                "reason": "Need at least two 10 C stress groups with three failures each",
                "failures_by_rounded_stress_c": {
                    str(key): int(value) for key, value in failures_by_group.items()
                },
            }
    numeric_features = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column
        not in {
            "time_to_event",
            "event_observed",
            "threshold_ohm",
            "smoothing_windows",
            "persistence_windows",
            "baseline_feature_windows",
        }
    ]
    report["survival_forest"] = benchmark_survival_forest(
        frame, feature_columns=numeric_features[:10]
    )
    report_path = output / "reliability_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    curve_time = np.linspace(0, float(frame["time_to_event"].max()) * 1.1, 200)
    curve = pd.DataFrame(
        {
            "time": curve_time,
            "survival_probability": survival_probability(
                curve_time, beta=weibull.beta, eta=weibull.eta
            ),
        }
    )
    curve.to_csv(output / "weibull_curve.csv", index=False)
    chart_paths = write_reliability_charts(frame, curve, output)
    report["artifacts"] = {
        "weibull_curve.csv": sha256_file(output / "weibull_curve.csv"),
        **{path.name: sha256_file(path) for path in chart_paths},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
