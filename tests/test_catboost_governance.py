import json

import pytest

from semiyield.common.catboost import SCHEMA_VERSION, load_params, sample_candidates, write_params


def test_candidates_are_reproducible_and_within_governed_ranges():
    first = sample_candidates(7)
    assert first == sample_candidates(7)
    assert len(first) == 20
    assert {row["iterations"] for row in first} <= {200, 300, 400, 600}
    assert all(0.01 <= row["learning_rate"] <= 0.1 for row in first)


def test_parameter_artifact_roundtrip_and_task_validation(tmp_path):
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "task": "manufacturing",
        "data_sha256": "data",
        "best_params": {
            "iterations": 400,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
        },
    }
    path = write_params(artifact, tmp_path / "params.json")
    params, digest = load_params(path, task="manufacturing", data_sha256="data")
    assert params == artifact["best_params"]
    assert len(digest) == 64
    with pytest.raises(ValueError, match="fingerprint"):
        load_params(path, task="manufacturing", data_sha256="other")
    with pytest.raises(ValueError, match="incompatible"):
        load_params(path, task="packaging")
    path.write_text(json.dumps({**artifact, "best_params": {}}))
    with pytest.raises(ValueError, match="invalid"):
        load_params(path, task="manufacturing")
