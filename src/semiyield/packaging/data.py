"""Mixed process inputs and training-only low-throughput proxy labels."""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LOCAL_INPUT = Path(
    "docs/Dataset_from_semiconductor_processes/mixed_categorical_numerical_data.csv"
)
CATEGORICAL = [f"X{i}" for i in range(1, 6)]
NUMERIC = [f"X{i}" for i in range(6, 17)]
FEATURES = CATEGORICAL + NUMERIC


def local_data_status(path: str | Path = DEFAULT_LOCAL_INPUT) -> dict[str, object]:
    """Describe optional local source data without treating it as a packaged asset."""
    source = Path(path)
    return {
        "path": str(source),
        "available": source.is_file(),
        "message": (
            "Local source data is available. Pass this path explicitly with --input-csv."
            if source.is_file()
            else (
                "No local packaging source is installed. Supply --input-csv; "
                "raw source data is not distributed."
            )
        ),
    }


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
