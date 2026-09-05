"""SECOM acquisition, parsing, and schema validation."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from semiyield.common.data import (  # noqa: F401
    _download_file,
    data_quality_report,
    sha256_file,
    validate_features,
)
from semiyield.constants import (
    DATA_FILE_URLS,
    DATA_URL,
    DEFAULT_DATA_DIR,
    FEATURE_PREFIX,
    SCHEMA_VERSION,
)


@dataclass(frozen=True)
class SecomDataset:
    features: pd.DataFrame
    target: pd.Series
    timestamps: pd.Series
    metadata: dict[str, object]


def _download_individual_files(destination: Path) -> dict[str, str]:
    hashes = {}
    for name, source_url in DATA_FILE_URLS.items():
        target = destination / name
        _download_file(source_url, target)
        hashes[name] = sha256_file(target)
    return hashes


def download_secom(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    url: str = DATA_URL,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Download and extract SECOM, using an atomic temporary download."""
    destination = Path(data_dir)
    features_path = destination / "secom.data"
    labels_path = destination / "secom_labels.data"
    if not force and features_path.exists() and labels_path.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "secom.zip"
    temporary = destination / "secom.zip.part"
    method = "archive"
    try:
        try:
            _download_file(url, temporary)
            actual_hash = sha256_file(temporary)
            if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
                raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}")
            temporary.replace(archive)
            with zipfile.ZipFile(archive) as zipped:
                safe_names = [
                    n
                    for n in zipped.namelist()
                    if Path(n).name in {"secom.data", "secom_labels.data", "secom.names"}
                ]
                if not {Path(n).name for n in safe_names} >= {
                    "secom.data",
                    "secom_labels.data",
                }:
                    raise ValueError("Archive does not contain the expected SECOM files")
                for name in safe_names:
                    with (
                        zipped.open(name) as source,
                        (destination / Path(name).name).open("wb") as output,
                    ):
                        shutil.copyfileobj(source, output)
            file_hashes = {
                name: sha256_file(destination / name)
                for name in ("secom.data", "secom_labels.data")
            }
        except Exception:
            method = "individual-files-fallback"
            archive.unlink(missing_ok=True)
            file_hashes = _download_individual_files(destination)
            actual_hash = None
        (destination / "download.json").write_text(
            json.dumps(
                {
                    "url": url,
                    "archive_sha256": actual_hash,
                    "file_sha256": file_hashes,
                    "schema": SCHEMA_VERSION,
                    "method": method,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_secom(data_dir: str | Path = DEFAULT_DATA_DIR, *, download: bool = False) -> SecomDataset:
    root = Path(data_dir)
    if download:
        download_secom(root)
    feature_path = root / "secom.data"
    label_path = root / "secom_labels.data"
    if not feature_path.exists() or not label_path.exists():
        raise FileNotFoundError("SECOM files are missing; run `semiyield download` first")
    features = pd.read_csv(feature_path, sep=r"\s+", header=None, na_values="NaN")
    features.columns = [f"{FEATURE_PREFIX}{i:03d}" for i in range(features.shape[1])]
    labels = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None,
        names=["raw_label", "timestamp_text"],
        quotechar='"',
    )
    if len(features) != len(labels):
        raise ValueError("Feature and label row counts do not match")
    target = labels["raw_label"].map({-1: 0, 1: 1})
    if target.isna().any():
        raise ValueError("Unexpected target label; expected only -1 and 1")
    timestamps = pd.to_datetime(labels["timestamp_text"], format="%d/%m/%Y %H:%M:%S")
    return SecomDataset(
        features=features.astype(float),
        target=target.astype(int).rename("failed"),
        timestamps=timestamps.rename("timestamp"),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "rows": len(features),
            "features": features.shape[1],
            "failure_count": int(target.sum()),
            "license": "CC BY 4.0",
            "source": DATA_URL,
            "file_sha256": {
                "secom.data": sha256_file(feature_path),
                "secom_labels.data": sha256_file(label_path),
            },
        },
    )
