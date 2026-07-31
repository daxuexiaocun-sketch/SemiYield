"""Feature, missingness, and prediction drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from semiyield.data import validate_features
from semiyield.modeling import ModelArtifact


def population_stability_index(reference, current, bins: int = 10) -> float:
    ref = pd.Series(reference).dropna().to_numpy()
    cur = pd.Series(current).dropna().to_numpy()
    if not len(ref) or not len(cur):
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _severity(psi: float, adjusted_p: float, missing_delta: float) -> str:
    if psi >= 0.25 or adjusted_p < 0.01 or missing_delta >= 0.2:
        return "red"
    if psi >= 0.1 or adjusted_p < 0.05 or missing_delta >= 0.1:
        return "yellow"
    return "green"


def _benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    artifact: ModelArtifact | None = None,
) -> pd.DataFrame:
    columns = artifact.feature_columns if artifact else list(reference.columns)
    reference_validated, _ = validate_features(reference, columns)
    current_validated, _ = validate_features(current, columns)
    raw = []
    for column in columns:
        ks = ks_2samp(reference_validated[column].dropna(), current_validated[column].dropna())
        raw.append(
            {
                "feature": column,
                "psi": population_stability_index(
                    reference_validated[column], current_validated[column]
                ),
                "ks_statistic": float(ks.statistic),
                "p_value": float(ks.pvalue),
                "missing_rate_reference": float(reference_validated[column].isna().mean()),
                "missing_rate_current": float(current_validated[column].isna().mean()),
            }
        )
    result = pd.DataFrame(raw)
    result["adjusted_p_value"] = _benjamini_hochberg(result["p_value"])
    result["missing_rate_delta"] = (
        result["missing_rate_current"] - result["missing_rate_reference"]
    ).abs()
    result["severity"] = [
        _severity(psi, p, delta)
        for psi, p, delta in zip(
            result["psi"], result["adjusted_p_value"], result["missing_rate_delta"], strict=True
        )
    ]
    return result.sort_values(["severity", "psi"], ascending=[False, False]).reset_index(drop=True)
