"""Generate raw linked synthetic observations without business-model imports."""

import numpy as np
import pandas as pd

from semiyield.common.artifacts import sha256_file
from semiyield.common.reporting import write_json_report
from semiyield.common.validation import managed_output_dir
from semiyield.simulation.contracts import PACKAGING_FEATURES, SENSORS, TABLES, VERSION
from semiyield.simulation.scenario import Scenario


def _generate_into(output, *, seed=42, batches=100, units_per_batch=100):
    scenario = Scenario(seed=seed, batches=batches, units_per_batch=units_per_batch)
    scenario.validate()
    manufacturing_rng, packaging_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
    )
    n = batches * units_per_batch
    batch = np.repeat(np.arange(batches), units_per_batch)
    test_batches = manufacturing_rng.permutation(batches)[: max(1, int(np.ceil(0.2 * batches)))]
    split = np.where(np.isin(batch, test_batches), "test", "train")
    latent = manufacturing_rng.normal(0, 0.7, batches)[batch] + manufacturing_rng.normal(0, 0.8, n)
    identity = pd.DataFrame(
        {
            "batch_id": [f"B{i:04d}" for i in batch],
            "unit_id": [f"U{i:07d}" for i in range(n)],
            "split": split,
            "dataset_role": "synthetic",
        }
    )
    manufacturing = identity.copy()
    sensors = manufacturing_rng.normal(size=(n, len(SENSORS)))
    sensors[:, :8] += latent[:, None] * np.linspace(0.5, 1.2, 8)
    manufacturing[SENSORS] = sensors
    failure_probability = 1 / (1 + np.exp(-(-2.7 + 1.1 * latent)))
    manufacturing["failed"] = (manufacturing_rng.random(n) < failure_probability).astype(int)
    packaging = identity.loc[manufacturing.failed.eq(0)].copy()
    idx = packaging.index.to_numpy()
    process = latent[idx] + packaging_rng.normal(0, 0.6, len(idx))
    for i, count in enumerate([2, 3, 4, 7, 22], 1):
        code = packaging_rng.integers(1, count + 1, len(idx))
        packaging[f"X{i}"] = [f"X{i}-{c}" for c in code]
    for i in range(6, 17):
        packaging[f"X{i}"] = process * (0.3 + i / 20) + packaging_rng.normal(size=len(idx))
    # Synthetic throughput is defined as units/hour; it is not a conversion of source Y.
    packaging["Y"] = np.exp(6.2 - 0.45 * process + packaging_rng.normal(0, 0.22, len(idx)))
    # Stage routing belongs to demo.workflow, which applies the packaging label rule.
    tables = dict(zip(TABLES, [manufacturing, packaging], strict=True))
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
        "packaging_candidates.csv": "Raw packaging observations after manufacturing pass",
    }
    manifest = {
        "schema_version": VERSION,
        "dataset_role": "synthetic",
        "seed": seed,
        "batches": batches,
        "units_per_batch": units_per_batch,
        "split": {
            "method": "batch_holdout",
            "test_fraction": 0.2,
            "test_batch_ids": [f"B{i:04d}" for i in sorted(test_batches)],
        },
        "features": {"manufacturing": SENSORS, "packaging": PACKAGING_FEATURES},
        "fields": fields,
        "mechanism": (
            "Shared latent quality + independent stage noise; latent quality is not exported"
        ),
        "random_streams": {"manufacturing": 0, "packaging": 1, "lifetime": 2},
        "files": {
            name: {
                "sha256": sha256_file(output / name),
                "rows": len(table),
                "columns": list(table.columns),
            }
            for name, table in tables.items()
        },
    }
    write_json_report(manifest, output / "manifest.json")
    return manifest


def generate(output_dir, *, seed=42, batches=100, units_per_batch=100, force=False):
    """Generate raw manufacturing and packaging observations into a managed directory."""
    with managed_output_dir(output_dir, force=force) as output:
        return _generate_into(output, seed=seed, batches=batches, units_per_batch=units_per_batch)


def generate_lifetime_observations(packaging_passed: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Generate lifetime observations only for devices observed to pass packaging."""
    required = {"batch_id", "unit_id", "split", "dataset_role", "X6"}
    missing = sorted(required - set(packaging_passed))
    if missing:
        raise ValueError(f"Missing lifetime generation columns: {', '.join(missing)}")
    rows = packaging_passed[["batch_id", "unit_id", "split", "dataset_role"]].copy()
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
    temperature = rng.choice([85.0, 105.0, 125.0], len(rows))
    stress = pd.to_numeric(packaging_passed["X6"], errors="raise").to_numpy(dtype=float)
    eta = 1800 * np.exp(4500 * (1 / (temperature + 273.15) - 1 / 378.15) - 0.2 * stress)
    failure_time = eta * rng.weibull(1.7, len(rows))
    followup = rng.uniform(700, 2600, len(rows))
    rows["temperature_c"] = temperature
    rows["time_to_event"] = np.maximum(np.minimum(failure_time, followup), 1e-9)
    rows["event_observed"] = (failure_time <= followup).astype(int)
    return rows
