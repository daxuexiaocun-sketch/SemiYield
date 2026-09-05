from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_json_report(payload: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return destination


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def data_quality_report(
    features: pd.DataFrame, target: pd.Series | None = None
) -> dict[str, object]:
    """Return a business-neutral tabular quality summary."""
    missing = features.isna().mean()
    report: dict[str, object] = {
        "rows": len(features),
        "columns": features.shape[1],
        "constant_columns": features.nunique(dropna=True)
        .le(1)
        .loc[lambda values: values]
        .index.tolist(),
        "duplicate_columns": int(features.T.duplicated().sum()),
        "high_missing_columns": missing[missing > 0.5].index.tolist(),
        "overall_missing_rate": float(features.isna().mean().mean()),
    }
    if target is not None:
        report.update({"failure_count": int(target.sum()), "failure_rate": float(target.mean())})
    return report
