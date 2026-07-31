"""Model creation, training, calibration, and artifact persistence."""

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

from semiyield.constants import DEFAULT_THRESHOLD, RANDOM_STATE, SCHEMA_VERSION
from semiyield.preprocessing import build_preprocessor


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


def _classifier(name: str, random_state: int) -> tuple[BaseEstimator, bool, int | None]:
    if name == "dummy":
        return DummyClassifier(strategy="prior"), True, None
    if name == "logistic":
        return (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
            True,
            None,
        )
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Install SemiYield with `pip install -e '.[catboost]'`") from exc
        return (
            CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.04,
                loss_function="Logloss",
                eval_metric="PRAUC",
                auto_class_weights="Balanced",
                random_seed=random_state,
                thread_count=4,
                used_ram_limit="24gb",
                verbose=False,
                allow_writing_files=False,
            ),
            False,
            None,
        )
    if name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as exc:
            raise ImportError("Install the optional TabPFN dependency first") from exc
        # The imputer can add one missingness indicator per selected feature. Selecting at
        # most 250 raw columns guarantees the final matrix stays within the v2 limit of 500.
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2), True, 250
    raise ValueError(f"Unknown model: {name}")


def build_model_pipeline(model_name: str, random_state: int = RANDOM_STATE) -> Pipeline:
    """Build an unfitted model pipeline for leakage-safe cross-fitting."""
    classifier, scale, max_features = _classifier(model_name, random_state)
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
) -> ModelArtifact:
    if len(np.unique(target)) < 2:
        raise ValueError("Training requires both pass and fail samples")
    pipeline = build_model_pipeline(model_name, random_state)
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
        metadata={"calibrated": calibrated, "fit_seconds": duration, "random_state": random_state},
    )
