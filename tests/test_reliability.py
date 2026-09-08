import numpy as np
import pandas as pd
import pytest

from semiyield.reliability.data import validate_lifetime_data
from semiyield.reliability.lifetime import (
    RSF_BASELINE_FEATURES,
    benchmark_survival_forest,
    fit_arrhenius_weibull,
    fit_weibull,
    survival_probability,
    sweep_arrhenius_weibull,
)
from semiyield.reliability.reporting import write_reliability_report


@pytest.fixture
def lifetime_data():
    rng = np.random.default_rng(7)
    rows = []
    for temperature, scale in [(180, 1500), (210, 900), (240, 500)]:
        times = scale * rng.weibull(2.2, 16)
        for index, value in enumerate(times):
            rows.append(
                {
                    "unit_id": f"u{temperature}_{index}",
                    "time_to_event": max(float(value), 1.0),
                    "event_observed": index % 7 != 0,
                    "temperature_c": temperature,
                    "initial_rds_on_ohm": 0.03 + rng.normal(0, 0.001),
                    "rds_on_slope": rng.normal(0, 0.0001),
                }
            )
    return pd.DataFrame(rows)


def test_weibull_handles_censoring(lifetime_data):
    result = fit_weibull(lifetime_data)
    assert result.beta > 0
    assert result.eta > result.b10 > 0
    assert result.censored > 0
    assert survival_probability([0], beta=result.beta, eta=result.eta)[0] == 1


def test_arrhenius_reports_extrapolation(lifetime_data):
    result = fit_arrhenius_weibull(lifetime_data, use_temperature_c=55)
    assert result.activation_energy_ev > 0
    assert result.eta_at_use > 0
    assert result.extrapolation_warning


def test_arrhenius_temperature_sweep(lifetime_data):
    sweep = sweep_arrhenius_weibull(lifetime_data, use_temperatures_c=(55, 85, 105, 125))
    assert [row["use_temperature_c"] for row in sweep] == [55.0, 85.0, 105.0, 125.0]
    assert all(row["eta_s"] > row["b10_s"] > 0 for row in sweep)
    assert all(row["support_status"] == "extrapolated" for row in sweep)
    within_range = sweep_arrhenius_weibull(lifetime_data, use_temperatures_c=(180,))[0]
    assert within_range["support_status"] == "extrapolated"
    assert within_range["fitted_temperature_range_status"] == "within_fitted_range"


def test_invalid_lifetime_rejected(lifetime_data):
    invalid = lifetime_data.copy()
    invalid.loc[0, "time_to_event"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_lifetime_data(invalid)


def test_survival_forest_requires_missing_baseline_features(lifetime_data):
    result = benchmark_survival_forest(lifetime_data.drop(columns=["rds_on_slope"]))
    assert result["status"] == "required_but_unavailable"
    assert result["feature_columns"] == list(RSF_BASELINE_FEATURES)
    assert "Missing baseline survival features for required RSF" in result["reason"]


def test_survival_forest_skips_without_optional_dependency(monkeypatch):
    rows = []
    for index in range(30):
        rows.append(
            {
                "unit_id": f"device_{index:03d}",
                "split": "train" if index < 24 else "test",
                "time_to_event": float(100 + index),
                "event_observed": index % 5 != 0,
                "baseline_temperature_c_mean": 105.0,
                "baseline_rds_on_ohm_mean": 0.5 + index * 0.01,
                "baseline_rds_on_ohm_slope": 0.0001,
            }
        )
    import builtins

    original_import = builtins.__import__

    def missing_sksurv(name, *args, **kwargs):
        if name.startswith("sksurv"):
            raise ImportError("optional dependency missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sksurv)
    result = benchmark_survival_forest(pd.DataFrame(rows))
    assert result["status"] == "required_but_unavailable"
    assert result["reason"] == (
        "scikit-survival is required; run `uv sync --locked --extra survival`"
    )


def test_survival_forest_uses_fixed_baseline_features_and_device_split(tmp_path):
    pytest.importorskip("sksurv")
    rng = np.random.default_rng(19)
    rows = []
    for index in range(48):
        temperature = 105.0 + 10.0 * (index % 3)
        rows.append(
            {
                "unit_id": f"device_{index:03d}",
                "split": "train" if index < 24 else "validation" if index < 32 else "test",
                "time_to_event": float(300 + 15 * index + rng.uniform(0, 10)),
                "event_observed": index % 6 != 0,
                "baseline_temperature_c_mean": temperature,
                "baseline_rds_on_ohm_mean": 0.5 + index * 0.01,
                "baseline_rds_on_ohm_slope": rng.normal(0, 0.0002),
                "future_outcome_proxy": rng.normal(),
            }
        )
    result = benchmark_survival_forest(pd.DataFrame(rows), data_sha256="test-digest")
    assert result["status"] == "completed"
    assert result["feature_columns"] == list(RSF_BASELINE_FEATURES)
    assert result["feature_source_columns"] == {
        "baseline_temperature_c_mean": "baseline_temperature_c_mean",
        "baseline_rds_on_ohm_mean": "baseline_rds_on_ohm_mean",
        "baseline_rds_on_ohm_slope": "baseline_rds_on_ohm_slope",
    }
    assert "future_outcome_proxy" not in result["feature_columns"]
    assert result["train_units"] == 32
    assert result["test_units"] == 16
    assert result["split_source"] == "input_device_manifest"
    assert result["data_sha256"] == "test-digest"
    assert result["c_index_bootstrap_valid_resamples"] > 0
    assert "uno_c_index" in result
    prediction = result["_individual_prediction"]
    assert prediction["status"] == "completed"
    assert len(prediction["curves"]) == 16
    assert all(curve["unit_id"].startswith("device_") for curve in prediction["curves"])


def test_survival_forest_uses_documented_smoke_aliases_and_deterministic_split(lifetime_data):
    pytest.importorskip("sksurv")
    result = benchmark_survival_forest(lifetime_data)
    assert result["status"] == "completed"
    assert result["feature_source_columns"] == {
        "baseline_temperature_c_mean": "temperature_c",
        "baseline_rds_on_ohm_mean": "initial_rds_on_ohm",
        "baseline_rds_on_ohm_slope": "rds_on_slope",
    }
    assert result["split_source"] == "deterministic_unit_order"
    assert result["train_units"] == 38
    assert result["test_units"] == 10


def test_survival_forest_requires_a_valid_declared_split(lifetime_data):
    invalid = lifetime_data.assign(split="validation")
    result = benchmark_survival_forest(invalid)
    assert result["status"] == "required_but_unavailable"
    assert "must include both train and test" in result["reason"]


def test_report_omits_risk_chart_when_required_rsf_is_unavailable(lifetime_data, tmp_path):
    report = write_reliability_report(
        lifetime_data.drop(columns=["rds_on_slope"]), output_dir=tmp_path
    )
    rsf = report["survival_forest"]
    assert rsf["status"] == "required_but_unavailable"
    assert rsf["individual_predicted_survival"]["status"] == "not_generated"
    assert not (tmp_path / "rsf_individual_predicted_survival.svg").exists()


def test_report_artifacts(lifetime_data, tmp_path):
    report = write_reliability_report(lifetime_data, output_dir=tmp_path)
    assert "weibull" in report
    assert (tmp_path / "reliability_report.json").exists()
    assert (tmp_path / "weibull_curve.csv").exists()
    assert (tmp_path / "arrhenius_lifetime_sweep.csv").exists()
    chart = tmp_path / "arrhenius_lifetime_sweep.svg"
    assert chart.exists()
    svg = chart.read_text(encoding="utf-8")
    assert "B10 = 10% failure time" in svg
    assert "63.2% failure time" in svg
    assert "+ = declared use-temperature extrapolation" in svg
    assert "Grouped bars show B10" in svg
    assert "Lifetime metric" in svg
    rsf_chart = tmp_path / "rsf_individual_predicted_survival.svg"
    assert rsf_chart.exists()
    rsf_svg = rsf_chart.read_text(encoding="utf-8")
    for term in ("u240_", "not Kaplan", "n=", "<title>", "<desc>"):
        assert term in rsf_svg
    prediction = report["survival_forest"]["individual_predicted_survival"]
    assert prediction["status"] == "completed"
    assert prediction["test_units"] == report["survival_forest"]["test_units"]
    assert "artifact_sha256" in prediction
    assert "_individual_prediction" not in report["survival_forest"]
    accelerated_svg = (tmp_path / "accelerated_life.svg").read_text(encoding="utf-8")
    assert "Failure-supported temperature range" not in accelerated_svg
    for color in ("#2878b5", "#3b9b8a", "#d47832", "#7252b8"):
        assert color in accelerated_svg
