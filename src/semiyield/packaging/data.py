"""Mixed process inputs and training-only low-throughput proxy labels."""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(
    "docs/Dataset_from_semiconductor_processes/mixed_categorical_numerical_data.csv"
)
CATEGORICAL = [f"X{i}" for i in range(1, 6)]
NUMERIC = [f"X{i}" for i in range(6, 17)]
FEATURES = CATEGORICAL + NUMERIC


def validate_data(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    required = FEATURES + (["Y"] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing packaging columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Packaging input contains no samples")
    result = frame.copy()
    for column in NUMERIC + (["Y"] if require_target else []):
        result[column] = pd.to_numeric(result[column], errors="raise")
        values = result[column].to_numpy(dtype=float)
        if np.isinf(values).any() or (column == "Y" and not np.isfinite(values).all()):
            raise ValueError(f"{column} contains missing or non-finite values")
    for column in CATEGORICAL:
        result[column] = result[column].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return result


def resolve_threshold(training_y, *, threshold=None, quantile=None):
    if threshold is not None and quantile is not None:
        raise ValueError("Specify either threshold or quantile, not both")
    y = np.asarray(training_y, dtype=float)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError("Training Y must be nonempty and finite")
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        return float(threshold), {"mode": "fixed", "value": float(threshold)}
    q = 0.1 if quantile is None else quantile
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    value = float(np.quantile(y, q))
    return value, {"mode": "training_quantile", "quantile": q, "value": value}


def proxy_labels(y, threshold):
    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Y and threshold must be finite")
    return (values < threshold).astype(int)
