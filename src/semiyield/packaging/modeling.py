"""Packaging models preserve mixed input types and proxy-label provenance."""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from semiyield.packaging.data import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    proxy_labels,
    resolve_threshold,
    validate_data,
)


def build_pipeline(model="logistic", seed=42):
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    if model == "dummy":
        classifier = DummyClassifier(strategy="prior")
    elif model == "logistic":
        classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    elif model == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Run `uv sync --locked --extra catboost`") from exc
        classifier = CatBoostClassifier(
            iterations=200,
            depth=6,
            random_seed=seed,
            thread_count=4,
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"Unknown packaging model: {model}")
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC),
                        ("categorical", categorical, CATEGORICAL),
                    ]
                ),
            ),
            ("model", classifier),
        ]
    )


@dataclass
class PackagingArtifact:
    estimator: Pipeline
    model_name: str
    throughput_threshold: float
    metadata: dict
    probability_threshold: float = 0.5
    schema_version: str = "semiyield-packaging-v1"
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURES))

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise TypeError("Expected a packaging model artifact")
        return artifact

    def predict(self, frame: pd.DataFrame):
        data = validate_data(frame, require_target=False)
        scores = self.estimator.predict_proba(data[self.feature_columns])[:, 1]
        result = data[[c for c in ("batch_id", "unit_id") if c in data]].copy()
        result["proxy_failure_probability"] = scores
        result["predicted_proxy_failed"] = (scores >= self.probability_threshold).astype(int)
        return result


def train_model(
    frame,
    *,
    model="logistic",
    seed=42,
    threshold=None,
    quantile=None,
    probability_threshold=0.5,
    data_sha256=None,
):
    data = validate_data(frame)
    value, provenance = resolve_threshold(data.Y, threshold=threshold, quantile=quantile)
    target = proxy_labels(data.Y, value)
    if len(np.unique(target)) < 2:
        raise ValueError("Training requires both proxy pass and proxy fail samples")
    if not np.isfinite(probability_threshold) or not 0 <= probability_threshold <= 1:
        raise ValueError("probability_threshold must be between 0 and 1")
    estimator = build_pipeline(model, seed).fit(data[FEATURES], target)
    return PackagingArtifact(
        estimator,
        model,
        value,
        {
            "threshold": provenance,
            "seed": seed,
            "data_sha256": data_sha256,
            "input_schema": {"categorical": CATEGORICAL, "numeric": NUMERIC},
            "label_rule": "Y < throughput_threshold",
            "target_role": "low_throughput_proxy_failure",
            "training_rows": len(data),
        },
        probability_threshold,
    )
