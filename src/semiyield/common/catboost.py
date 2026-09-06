"""Reproducible CatBoost defaults, tuning, and parameter artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

SCHEMA_VERSION = "semiyield-catboost-params-v1"
TRIALS = 20
BASE_PARAMS = {
    "iterations": 400,
    "learning_rate": 0.04,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
}
_RUNTIME = {
    "thread_count": 4,
    "auto_class_weights": "Balanced",
    "verbose": False,
    "allow_writing_files": False,
}
_TUNABLE = {"iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength"}


def _validate_params(chosen: Mapping[str, object]) -> None:
    if set(chosen) != _TUNABLE:
        raise ValueError("CatBoost parameter artifact has invalid best_params")
    if (
        chosen["iterations"] not in {200, 300, 400, 600}
        or not 0.01 <= float(chosen["learning_rate"]) <= 0.1
        or int(chosen["depth"]) not in range(4, 9)
        or not 1 <= float(chosen["l2_leaf_reg"]) <= 10
        or not 0.25 <= float(chosen["random_strength"]) <= 2
    ):
        raise ValueError("CatBoost parameters are outside the governed search ranges")


def catboost_classifier(*, seed: int, params: Mapping[str, object] | None = None, **task_params):
    """Create the only CatBoost classifier used by SemiYield."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Run `uv sync --locked --extra catboost`") from exc
    chosen = {**BASE_PARAMS, **(dict(params) if params else {})}
    unknown = set(chosen) - _TUNABLE
    if unknown:
        raise ValueError(f"Unsupported CatBoost parameters: {', '.join(sorted(unknown))}")
    _validate_params(chosen)
    return CatBoostClassifier(**chosen, **_RUNTIME, random_seed=seed, **task_params)


def sample_candidates(seed: int, trials: int = TRIALS) -> list[dict[str, object]]:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    rng = np.random.default_rng(seed)
    return [
        {
            "iterations": int(rng.choice([200, 300, 400, 600])),
            "learning_rate": float(10 ** rng.uniform(-2, -1)),
            "depth": int(rng.integers(4, 9)),
            "l2_leaf_reg": float(10 ** rng.uniform(0, 1)),
            "random_strength": float(rng.uniform(0.25, 2.0)),
        }
        for _ in range(trials)
    ]


def tune(
    build_pipeline: Callable[[Mapping[str, object]], object],
    features,
    target,
    *,
    task: str,
    data_sha256: str,
    seed: int = 42,
    trials: int = TRIALS,
) -> dict[str, object]:
    """Choose parameters using training-only, stratified PR-AUC cross-validation."""
    y = np.asarray(target, dtype=int)
    minority = int(np.bincount(y).min()) if len(y) else 0
    folds = min(3, minority)
    if folds < 2:
        raise ValueError("CatBoost tuning requires at least two samples in each class")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for params in sample_candidates(seed, trials):
        scores = []
        for train_idx, valid_idx in splitter.split(features, y):
            model = build_pipeline(params)
            model.fit(features.iloc[train_idx], y[train_idx])
            probability = model.predict_proba(features.iloc[valid_idx])[:, 1]
            scores.append(float(average_precision_score(y[valid_idx], probability)))
        rows.append(
            {"params": params, "mean_pr_auc": float(np.mean(scores)), "fold_pr_auc": scores}
        )
    rows.sort(
        key=lambda row: (-row["mean_pr_auc"], row["params"]["iterations"], row["params"]["depth"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "trials": trials,
        "folds": folds,
        "objective": "pr_auc",
        "data_sha256": data_sha256,
        "best_params": rows[0]["params"],
        "best_mean_pr_auc": rows[0]["mean_pr_auc"],
        "candidates": rows,
    }


def write_params(artifact: Mapping[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return destination


def load_params(
    path: str | Path, *, task: str, data_sha256: str | None = None
) -> tuple[dict[str, object], str]:
    raw = Path(path).read_bytes()
    artifact = json.loads(raw)
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("task") != task:
        raise ValueError("CatBoost parameter artifact is incompatible with this task")
    params = artifact.get("best_params")
    if not isinstance(params, dict):
        raise ValueError("CatBoost parameter artifact has invalid best_params")
    # Validate values even when loading a hand-edited JSON artifact.
    _validate_params(params)
    if data_sha256 is not None and artifact.get("data_sha256") != data_sha256:
        raise ValueError("CatBoost parameter artifact data fingerprint does not match input")
    return params, hashlib.sha256(raw).hexdigest()
