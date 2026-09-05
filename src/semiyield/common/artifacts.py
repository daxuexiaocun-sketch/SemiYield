"""Hashes and low-level artifact utilities."""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, attempts: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SemiYield/0.1"})
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                destination.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            if destination.stat().st_size == 0:
                raise ValueError(f"Empty response from {url}")
            return
        except Exception as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_error


def validate_features(
    frame: pd.DataFrame, expected_columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Coerce an inference frame to the training schema and return non-fatal warnings."""
    missing = sorted(set(expected_columns) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected_columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing[:10])}")
    warnings: list[str] = []
    if extra:
        warnings.append(f"Ignored {len(extra)} unexpected columns")
    ordered = frame.loc[:, expected_columns].apply(pd.to_numeric, errors="coerce")
    invalid = ordered.isna() & frame.loc[:, expected_columns].notna()
    if invalid.any().any():
        warnings.append(f"Coerced {int(invalid.sum().sum())} non-numeric values to missing")
    all_missing = ordered.columns[ordered.isna().all()].tolist()
    if all_missing:
        warnings.append(f"All values are missing for {len(all_missing)} columns")
    return ordered, warnings


def data_quality_report(
    features: pd.DataFrame, target: pd.Series | None = None
) -> dict[str, object]:
    missing = features.isna().mean()
    report: dict[str, object] = {
        "rows": len(features),
        "columns": features.shape[1],
        "constant_columns": features.nunique(dropna=True).le(1).loc[lambda x: x].index.tolist(),
        "duplicate_columns": int(features.T.duplicated().sum()),
        "high_missing_columns": missing[missing > 0.5].index.tolist(),
        "overall_missing_rate": float(features.isna().mean().mean()),
    }
    if target is not None:
        report.update({"failure_count": int(target.sum()), "failure_rate": float(target.mean())})
    return report
