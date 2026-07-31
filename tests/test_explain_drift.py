import numpy as np

from semiyield.drift import detect_drift, population_stability_index
from semiyield.explain import explain_prediction
from semiyield.modeling import train_model


def test_local_explanation(sample_data):
    frame, target = sample_data
    artifact = train_model(frame, target, model_name="logistic", calibrate=False)
    result = explain_prediction(artifact, frame.head(1), frame, top_k=2)
    assert len(result) == 2
    assert {"feature", "risk_contribution"} <= set(result.columns)


def test_drift_detects_shift(sample_data):
    frame, _ = sample_data
    current = frame.copy()
    current["sensor_000"] += 10
    result = detect_drift(frame, current)
    shifted = result.loc[result.feature == "sensor_000"].iloc[0]
    assert shifted.severity == "red"
    assert shifted.psi >= 0.25


def test_psi_is_small_for_identical_data():
    values = np.arange(100)
    assert population_stability_index(values, values) == 0
