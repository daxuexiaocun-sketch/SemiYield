"""Generate an auditable manifest from completed experiment artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from semiyield.data import sha256_file


def build_experiment_manifest(
    root: str | Path = "reports/verified",
    *,
    nasa_provenance: str | Path | None = None,
) -> dict[str, object]:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    yield_summary = output / "yield" / "summary.csv"
    files = [path for path in output.rglob("*") if path.is_file()]
    report: dict[str, object] = {
        "schema_version": "semiyield-experiment-evidence-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claims": {
            "yield": "UCI SECOM leakage-safe risk screening",
            "reliability": "right-censored lifetime and accelerated-stress analysis",
            "engineering": "hashes, fixed seeds, device isolation, tests and package build",
        },
        "nasa_analysis_status": "not_executed",
        "limitations": [
            "Anonymous SECOM variables cannot establish physical root cause.",
            "NASA results are research evidence, not production qualification.",
            "NASA row-level derivatives are not redistributed without explicit permission.",
        ],
    }
    if yield_summary.exists():
        summary = pd.read_csv(yield_summary)
        report["yield_rows"] = summary.to_dict(orient="records")
    status_rows = []
    for status_path in sorted(output.rglob("model_status.csv")):
        rows = pd.read_csv(status_path).fillna("")
        rows.insert(0, "experiment", str(status_path.parent.relative_to(output)))
        status_rows.extend(rows.to_dict(orient="records"))
    if status_rows:
        report["model_status"] = status_rows
    if nasa_provenance and Path(nasa_provenance).exists():
        provenance = json.loads(Path(nasa_provenance).read_text(encoding="utf-8"))
        report["nasa_analysis_status"] = "completed_locally"
        report["nasa_provenance"] = provenance
    report["artifacts"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in files
        if path.name != "experiment_manifest.json"
    }
    target = output / "experiment_manifest.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
