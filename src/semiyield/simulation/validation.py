"""Integrity checks for generated raw simulation tables."""

import json
from pathlib import Path

import pandas as pd

from semiyield.common.artifacts import sha256_file
from semiyield.simulation.contracts import TABLES, VERSION


def prepare_output(path: str | Path, force: bool = False) -> Path:
    output = Path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not force:
        raise ValueError(f"Output exists: {output}. Choose another directory or pass --force")
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_dataset(data_dir: str | Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION or manifest.get("dataset_role") != "synthetic":
        raise ValueError("Unsupported synthetic dataset manifest")
    tables = {}
    for name in TABLES:
        path = root / name
        if sha256_file(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {name}; regenerate the dataset")
        tables[name.removesuffix(".csv")] = pd.read_csv(path)
    manufacturing = tables["manufacturing"]
    packaging = tables["packaging_candidates"]
    lifetime = tables["lifetime_candidates"]
    if manufacturing.unit_id.duplicated().any() or not set(packaging.unit_id).issubset(set(manufacturing.unit_id)):
        raise ValueError("Simulation stage identifiers are inconsistent")
    if not set(lifetime.unit_id).issubset(set(packaging.unit_id)):
        raise ValueError("Lifetime candidates must be packaging candidates")
    if manufacturing.groupby("batch_id").split.nunique().gt(1).any():
        raise ValueError("A batch must belong to only one split")
    return manifest, tables
