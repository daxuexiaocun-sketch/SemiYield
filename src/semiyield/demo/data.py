"""Reproducible, linked synthetic manufacturing, packaging and lifetime data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from semiyield.common.data import sha256_file
from semiyield.packaging.data import FEATURES, proxy_labels, resolve_threshold

VERSION = "semiyield-three-stage-v1"
SENSORS = [f"sensor_{i:03d}" for i in range(24)]
TABLES = ("manufacturing.csv", "packaging.csv", "lifetime.csv", "trace.csv")


def prepare_output(path, force=False):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())) and not force:
        raise ValueError(f"Output exists: {path}. Choose another directory or pass --force")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def generate(output_dir, *, seed=42, batches=100, units_per_batch=100, force=False):
    if batches < 2 or units_per_batch < 1:
        raise ValueError("At least 2 batches and 1 unit per batch are required")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    output = prepare_output(output_dir, force)
    rng = np.random.default_rng(seed)
    n = batches * units_per_batch
    batch = np.repeat(np.arange(batches), units_per_batch)
    test_batches = rng.permutation(batches)[: max(1, int(np.ceil(0.2 * batches)))]
    split = np.where(np.isin(batch, test_batches), "test", "train")
    latent = rng.normal(0, 0.7, batches)[batch] + rng.normal(0, 0.8, n)
    identity = pd.DataFrame(
        {
            "batch_id": [f"B{i:04d}" for i in batch],
            "unit_id": [f"U{i:07d}" for i in range(n)],
            "split": split,
            "dataset_role": "synthetic",
        }
    )
    manufacturing = identity.copy()
    sensors = rng.normal(size=(n, len(SENSORS)))
    sensors[:, :8] += latent[:, None] * np.linspace(0.5, 1.2, 8)
    manufacturing[SENSORS] = sensors
    failure_probability = 1 / (1 + np.exp(-(-2.7 + 1.1 * latent)))
    manufacturing["failed"] = (rng.random(n) < failure_probability).astype(int)
    packaging = identity.loc[manufacturing.failed.eq(0)].copy()
    idx = packaging.index.to_numpy()
    process = latent[idx] + rng.normal(0, 0.6, len(idx))
    for i, count in enumerate([2, 3, 4, 7, 22], 1):
        code = rng.integers(1, count + 1, len(idx))
        packaging[f"X{i}"] = [f"X{i}-{c}" for c in code]
    for i in range(6, 17):
        packaging[f"X{i}"] = process * (0.3 + i / 20) + rng.normal(size=len(idx))
    # Synthetic throughput is defined as units/hour; it is not a conversion of source Y.
    packaging["Y"] = np.exp(6.2 - 0.45 * process + rng.normal(0, 0.22, len(idx)))
    training_y = packaging.loc[packaging.split.eq("train"), "Y"]
    threshold = None
    threshold_info = {"status": "skipped", "reason": "No packaging training samples"}
    if len(training_y):
        threshold, threshold_info = resolve_threshold(training_y)
        packaging["proxy_failed"] = proxy_labels(packaging.Y, threshold)
    else:
        packaging["proxy_failed"] = pd.Series(pd.NA, index=packaging.index, dtype="Int64")
    lifetime = packaging.loc[packaging.proxy_failed.eq(0).fillna(False), identity.columns].copy()
    life_idx = lifetime.index.to_numpy()
    temperature = rng.choice([85.0, 105.0, 125.0], len(lifetime))
    stress = process[np.flatnonzero(packaging.proxy_failed.eq(0).fillna(False).to_numpy())]
    # Arrhenius acceleration plus correlated upstream quality and packaging damage.
    eta = 1800 * np.exp(
        4500 * (1 / (temperature + 273.15) - 1 / 378.15) - 0.3 * latent[life_idx] - 0.2 * stress
    )
    failure_time = eta * rng.weibull(1.7, len(lifetime))
    followup = rng.uniform(700, 2600, len(lifetime))
    lifetime["temperature_c"] = temperature
    lifetime["time_to_event"] = np.maximum(np.minimum(failure_time, followup), 1e-9)
    lifetime["event_observed"] = (failure_time <= followup).astype(int)
    trace = identity.copy()
    trace["manufacturing_failed"] = manufacturing.failed
    trace["entered_packaging"] = trace.unit_id.isin(packaging.unit_id)
    trace["packaging_proxy_failed"] = packaging.proxy_failed.reindex(trace.index).astype("Int64")
    trace["entered_lifetime"] = trace.unit_id.isin(lifetime.unit_id)
    trace["time_to_event"] = lifetime.time_to_event.reindex(trace.index)
    trace["event_observed"] = lifetime.event_observed.reindex(trace.index).astype("Int64")
    trace["stage_status"] = np.select(
        [
            trace.manufacturing_failed.eq(1),
            trace.packaging_proxy_failed.eq(1).fillna(False),
            trace.entered_lifetime,
        ],
        ["manufacturing_rejected", "packaging_proxy_rejected", "lifetime_observed"],
        default="packaging_label_unavailable",
    )
    tables = dict(zip(TABLES, [manufacturing, packaging, lifetime, trace], strict=True))
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    fields = {
        "batch_id": "Shared batch key; split isolation unit",
        "unit_id": "Unique device key shared across stages",
        "split": "train/test; assigned once by batch before generation",
        "dataset_role": "synthetic",
        "sensor_000..sensor_023": "Anonymous simulated measurements; arbitrary units",
        "failed": "Manufacturing observed failure: 1 fail, 0 pass",
        "X1..X5": "Synthetic categorical process settings",
        "X6..X16": "Synthetic numerical process measurements; arbitrary units",
        "Y": "Synthetic throughput; units/hour (source dataset units remain unspecified)",
        "proxy_failed": "1 when Y < training throughput threshold, otherwise 0",
        "temperature_c": "Accelerated stress temperature; Celsius",
        "time_to_event": "min(failure time, follow-up); hours",
        "event_observed": "1 observed failure, 0 right-censored; blank before lifetime stage",
        "manufacturing_failed": "Observed manufacturing result copied into trace",
        "entered_packaging": "Whether manufacturing passed",
        "packaging_proxy_failed": "Observed proxy result; blank when stage not reached/unavailable",
        "entered_lifetime": "Whether both earlier stages passed",
        "stage_status": "Final observed stage disposition, not a model prediction",
    }
    manifest = {
        "schema_version": VERSION,
        "dataset_role": "synthetic",
        "seed": seed,
        "batches": batches,
        "units_per_batch": units_per_batch,
        "threshold": threshold_info,
        "throughput_threshold": threshold,
        "split": {
            "method": "batch_holdout",
            "test_fraction": 0.2,
            "test_batch_ids": [f"B{i:04d}" for i in sorted(test_batches)],
        },
        "features": {"manufacturing": SENSORS, "packaging": FEATURES},
        "fields": fields,
        "mechanism": (
            "Shared latent quality + independent stage noise; latent quality is not exported"
        ),
        "files": {
            name: {
                "sha256": sha256_file(output / name),
                "rows": len(table),
                "columns": list(table.columns),
            }
            for name, table in tables.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_dataset(data_dir):
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION or manifest.get("dataset_role") != "synthetic":
        raise ValueError("Unsupported synthetic dataset manifest")
    for name in TABLES:
        if sha256_file(root / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {name}; regenerate the dataset")
    return manifest, {name.removesuffix(".csv"): pd.read_csv(root / name) for name in TABLES}
