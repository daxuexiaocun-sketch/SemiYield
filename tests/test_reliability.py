import numpy as np
import pandas as pd
import pytest

from semiyield.reliability import (
    fit_arrhenius_weibull,
    fit_weibull,
    survival_probability,
    validate_lifetime_data,
    write_reliability_report,
)


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
                    "initial_rds": 0.03 + rng.normal(0, 0.001),
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


def test_invalid_lifetime_rejected(lifetime_data):
    invalid = lifetime_data.copy()
    invalid.loc[0, "time_to_event"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_lifetime_data(invalid)


def test_report_artifacts(lifetime_data, tmp_path):
    report = write_reliability_report(lifetime_data, output_dir=tmp_path)
    assert "weibull" in report
    assert (tmp_path / "reliability_report.json").exists()
    assert (tmp_path / "weibull_curve.csv").exists()
