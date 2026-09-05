import json

import pandas as pd

from semiyield.common.experiment import build_experiment_manifest


def test_experiment_manifest_is_hashed_and_candid_without_nasa(tmp_path):
    output = tmp_path / "verified"
    yield_dir = output / "yield"
    yield_dir.mkdir(parents=True)
    pd.DataFrame([{"model": "catboost", "protocol": "temporal"}]).to_csv(
        yield_dir / "summary.csv", index=False
    )
    pd.DataFrame([{"model": "catboost", "status": "completed", "reason": ""}]).to_csv(
        yield_dir / "model_status.csv", index=False
    )
    report = build_experiment_manifest(output)
    persisted = json.loads((output / "experiment_manifest.json").read_text())
    assert report["nasa_analysis_status"] == "not_executed"
    assert persisted["model_status"][0]["status"] == "completed"
    assert "yield/summary.csv" in persisted["artifacts"]
