"""Manufacturing model creation, training, calibration, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from semiyield.common.catboost import BASE_PARAMS, catboost_classifier, tune
from semiyield.constants import DEFAULT_THRESHOLD, RANDOM_STATE, SCHEMA_VERSION

from .preprocessing import build_preprocessor


@dataclass
class ModelArtifact:
    estimator: BaseEstimator
    model_name: str
    feature_columns: list[str]
    threshold: float
    schema_version: str
    created_at: str
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise TypeError("File is not a SemiYield model artifact")
        return artifact


def _classifier(
    name: str, random_state: int, catboost_params: dict[str, object] | None = None
) -> tuple[BaseEstimator, bool, int | None]:
    if name == "dummy":
        return DummyClassifier(strategy="prior"), True, None
    if name == "logistic":
        return (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
            True,
            None,
        )
    if name == "catboost":
        return (
            catboost_classifier(
                seed=random_state,
                params=catboost_params,
                loss_function="Logloss",
                eval_metric="PRAUC",
                used_ram_limit="24gb",
            ),
            False,
            None,
        )
    if name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as exc:
            raise ImportError("Run `uv sync --locked --extra tabpfn`") from exc
        # The imputer can add one missingness indicator per selected feature. Selecting at
        # most 250 raw columns guarantees the final matrix stays within the v2 limit of 500.
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2), True, 250
    raise ValueError(f"Unknown model: {name}")


def build_model_pipeline(
    model_name: str,
    random_state: int = RANDOM_STATE,
    catboost_params: dict[str, object] | None = None,
) -> Pipeline:
    """Build an unfitted model pipeline for leakage-safe cross-fitting."""
    classifier, scale, max_features = _classifier(model_name, random_state, catboost_params)
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale=scale, max_features=max_features)),
            ("model", classifier),
        ]
    )


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    model_name: str = "logistic",
    calibrate: bool = True,
    calibration_folds: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
    random_state: int = RANDOM_STATE,
    catboost_params: dict[str, object] | None = None,
    catboost_params_sha256: str | None = None,
) -> ModelArtifact:
    if len(np.unique(target)) < 2:
        raise ValueError("Training requires both pass and fail samples")
    pipeline = build_model_pipeline(model_name, random_state, catboost_params)
    estimator: BaseEstimator = pipeline
    minority = int(pd.Series(target).value_counts().min())
    calibrated = calibrate and model_name != "dummy" and minority >= calibration_folds
    if calibrated:
        estimator = CalibratedClassifierCV(
            pipeline,
            method="sigmoid",
            cv=calibration_folds,
            ensemble=False,
        )
    started = perf_counter()
    estimator.fit(features, target)
    duration = perf_counter() - started
    return ModelArtifact(
        estimator=estimator,
        model_name=model_name,
        feature_columns=list(features.columns),
        threshold=float(threshold),
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata={
            "calibrated": calibrated,
            "fit_seconds": duration,
            "random_state": random_state,
            **(
                {
                    "catboost_params": {**BASE_PARAMS, **(catboost_params or {})},
                    "catboost_params_sha256": catboost_params_sha256,
                }
                if model_name == "catboost"
                else {}
            ),
        },
    )


def train_manufacturing(features, target, *, model="logistic", seed=42):
    """Train the manufacturing model used by the three-stage demonstration."""
    return train_model(features, target, model_name=model, random_state=seed, calibrate=False)


def tune_catboost(
    features, target, *, data_sha256: str, seed: int = RANDOM_STATE, trials: int = 20
):
    """Tune CatBoost without fitting preprocessing outside an inner fold."""
    return tune(
        lambda params: build_model_pipeline("catboost", seed, dict(params)),
        features,
        target,
        task="manufacturing",
        data_sha256=data_sha256,
        seed=seed,
        trials=trials,
    )
