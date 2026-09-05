import pandas as pd
import pytest

from semiyield.manufacturing import metrics, modeling, predict

classification_metrics = metrics.classification_metrics
evaluate_model = metrics.evaluate_model
ModelArtifact = modeling.ModelArtifact
train_model = modeling.train_model
predict_risk = predict.predict_risk


def test_train_save_load_predict(sample_data, tmp_path):
    frame, target = sample_data
    artifact = train_model(frame, target, model_name="logistic", calibration_folds=2)
    path = artifact.save(tmp_path / "model.joblib")
    restored = ModelArtifact.load(path)
    first = predict_risk(artifact, frame.head())
    second = predict_risk(restored, frame.head())
    pd.testing.assert_frame_equal(first, second)
    assert first.failure_probability.between(0, 1).all()


def test_calibration_uses_single_final_estimator(sample_data):
    frame, target = sample_data
    artifact = train_model(frame, target, model_name="logistic", calibration_folds=2)
    assert artifact.metadata["calibrated"] is True
    assert artifact.estimator.ensemble is False


def test_cv_evaluation(sample_data):
    frame, target = sample_data
    artifact = train_model(frame, target, model_name="logistic", calibrate=False)
    result = evaluate_model(artifact, frame, target, protocol="cv", folds=3, repeats=1)
    assert len(result) == 3
    assert result.pr_auc.between(0, 1).all()


def test_single_class_rejected(sample_data):
    frame, target = sample_data
    with pytest.raises(ValueError, match="both pass and fail"):
        train_model(frame, target * 0)


def test_metrics_include_business_capture():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], budget=0.5)
    assert metrics["capture_at_budget"] == 1.0
