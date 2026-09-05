"""Schema-checked manufacturing inference."""

from __future__ import annotations

import pandas as pd

from .modeling import ModelArtifact
from .preprocessing import validate_features


def predict_risk(artifact: ModelArtifact, features: pd.DataFrame) -> pd.DataFrame:
    validated, warnings = validate_features(features, artifact.feature_columns)
    probability = artifact.estimator.predict_proba(validated)[:, 1]
    return pd.DataFrame(
        {
            "failure_probability": probability,
            "predicted_failure": probability >= artifact.threshold,
            "decision_threshold": artifact.threshold,
            "model_name": artifact.model_name,
            "model_created_at": artifact.created_at,
            "schema_version": artifact.schema_version,
            "input_warnings": "; ".join(warnings),
        },
        index=features.index,
    )
