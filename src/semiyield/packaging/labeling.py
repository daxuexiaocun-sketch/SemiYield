"""The sole definition of the low-throughput proxy-failure label."""

import numpy as np


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
    """Return 1 exactly when throughput Y is strictly below the threshold."""
    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Y and threshold must be finite")
    return (values < threshold).astype(int)
