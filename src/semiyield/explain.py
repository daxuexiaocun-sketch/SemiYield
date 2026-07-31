"""Model-agnostic candidate-variable explanations."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from semiyield.data import validate_features
from semiyield.modeling import ModelArtifact


def _reference_medians(reference: pd.DataFrame) -> pd.Series:
    return reference.median(numeric_only=True).reindex(reference.columns)


def explain_prediction(
    artifact: ModelArtifact,
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Rank local candidate variables by probability change under median replacement."""
    validated, _ = validate_features(sample.iloc[:1], artifact.feature_columns)
    reference_validated, _ = validate_features(reference, artifact.feature_columns)
    baseline = float(artifact.estimator.predict_proba(validated)[:, 1][0])
    medians = _reference_medians(reference_validated)
    rows = []
    for column in artifact.feature_columns:
        perturbed = validated.copy()
        perturbed[column] = medians[column]
        replacement = float(artifact.estimator.predict_proba(perturbed)[:, 1][0])
        rows.append(
            {
                "feature": column,
                "observed_value": validated.iloc[0][column],
                "reference_median": medians[column],
                "risk_contribution": baseline - replacement,
            }
        )
    result = pd.DataFrame(rows)
    result["absolute_contribution"] = result["risk_contribution"].abs()
    return (
        result.sort_values("absolute_contribution", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def global_importance(artifact, features, target, *, repeats: int = 5, top_k: int = 20):
    result = permutation_importance(
        artifact.estimator,
        features,
        target,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=42,
    )
    return (
        pd.DataFrame({"feature": features.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
