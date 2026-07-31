import json
import sys

import pandas as pd

from semiyield.resource_guard import run_guarded


def test_guard_records_completed_process(tmp_path):
    (tmp_path / "manifest.json").write_text('{"artifacts": {}}')
    result = run_guarded(
        [sys.executable, "-c", "print('guard smoke')"],
        output_dir=tmp_path,
        models=["dummy"],
        soft_limit_gb=0.5,
        hard_limit_gb=1.0,
        poll_seconds=0.01,
    )
    assert result.status == "completed"
    report = json.loads((tmp_path / "resource_usage.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert report["peak_rss_bytes"] < report["hard_limit_bytes"]
    assert manifest["resource_guard"]["status"] == "completed"


def test_guard_kills_process_tree_and_records_models(tmp_path):
    result = run_guarded(
        [
            sys.executable,
            "-c",
            "import time; payload=bytearray(160*1024*1024); time.sleep(10)",
        ],
        output_dir=tmp_path,
        models=["catboost"],
        soft_limit_gb=0.05,
        hard_limit_gb=0.1,
        poll_seconds=0.01,
    )
    assert result.status == "memory_limited"
    assert result.exit_code == 137
    status = pd.read_csv(tmp_path / "model_status.csv")
    assert status.loc[0, "model"] == "catboost"
    assert status.loc[0, "status"] == "memory_limited"
