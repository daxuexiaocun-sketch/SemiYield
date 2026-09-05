from importlib import util

import pandas as pd
import pytest

from semiyield.manufacturing import modeling

ModelArtifact = modeling.ModelArtifact
train_model = modeling.train_model

pytestmark = pytest.mark.skipif(
    util.find_spec("catboost") is None,
    reason="CatBoost is an optional dependency",
)


def test_catboost_calibrated_save_load_smoke(sample_data, tmp_path):
    frame, target = sample_data
    artifact = train_model(frame, target, model_name="catboost", calibration_folds=2)
    probabilities = artifact.estimator.predict_proba(frame.head())[:, 1]
    restored = ModelArtifact.load(artifact.save(tmp_path / "catboost.joblib"))
    restored_probabilities = restored.estimator.predict_proba(frame.head())[:, 1]
    pd.testing.assert_series_equal(
        pd.Series(probabilities),
        pd.Series(restored_probabilities),
        check_names=False,
    )
