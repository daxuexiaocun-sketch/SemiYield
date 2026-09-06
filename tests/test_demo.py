import json
import re
import socket

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from semiyield.cli import app
from semiyield.demo.workflow import run
from semiyield.simulation.contracts import PACKAGING_FEATURES, SENSORS
from semiyield.simulation.generator import generate, generate_lifetime_observations
from semiyield.simulation.validation import load_dataset


def test_generation_reproducible_linked_and_batch_isolated(tmp_path):
    first = generate(tmp_path / "a", seed=42, batches=10, units_per_batch=30)
    second = generate(tmp_path / "b", seed=42, batches=10, units_per_batch=30)
    assert first == second
    manifest, tables = load_dataset(tmp_path / "a")
    manufacturing, packaging = (tables[key] for key in ["manufacturing", "packaging_candidates"])
    assert manufacturing.groupby("batch_id").split.nunique().eq(1).all()
    for table in tables.values():
        assert table.unit_id.is_unique
        assert table.dataset_role.eq("synthetic").all()
        joined = table.merge(manufacturing[["unit_id", "split"]], on="unit_id")
        assert joined.split_x.eq(joined.split_y).all()
    assert set(packaging.unit_id) == set(manufacturing.loc[manufacturing.failed.eq(0), "unit_id"])
    assert manifest["features"] == {"manufacturing": SENSORS, "packaging": PACKAGING_FEATURES}
    assert not any("latent" in column for table in tables.values() for column in table.columns)


def test_lifetime_is_generated_only_after_packaging_routing(tmp_path):
    generate(tmp_path / "data", seed=42, batches=5, units_per_batch=20)
    _, tables = load_dataset(tmp_path / "data")
    packaging = tables["packaging_candidates"].head(5)
    lifetime = generate_lifetime_observations(packaging, seed=42)
    assert set(lifetime.unit_id) == set(packaging.unit_id)
    assert lifetime.time_to_event.gt(0).all()
    assert set(lifetime.event_observed) <= {0, 1}


def test_offline_demo_reports_and_routing_counts(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")

    def forbidden(*args, **kwargs):
        raise AssertionError("Demo attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    data, output = tmp_path / "data", tmp_path / "reports"
    generate(data, batches=10, units_per_batch=30)
    report = run(data, output)
    assert report.exists()
    content = report.read_text()
    for target in re.findall(r"\]\(([^)]+)\)", content):
        assert (output / target).exists()
    stages = pd.read_csv(output / "stages.csv")
    assert stages.entered.iloc[1] == stages.entered.iloc[0] - stages.observed_failures.iloc[0]
    assert stages.entered.iloc[2] == stages.entered.iloc[1] - stages.observed_failures.iloc[1]
    assert stages.passed.isna().iloc[2]
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(metrics) == 4
    assert all(row["status"] == "completed" for row in metrics)
    assert {path.name for path in output.glob("*.svg")} == {
        "stage_flow.svg",
        "model_metrics.svg",
        "survival.svg",
    }
    assert "Events:" in (output / "survival.svg").read_text(encoding="utf-8")
    assert "Right-censored" in (output / "survival.svg").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Output exists"):
        run(data, output)
    with pytest.raises(ValueError, match="separate"):
        run(data, data, force=True)


def test_small_cohort_reports_skipped_without_fake_outcomes(tmp_path):
    pytest.importorskip("matplotlib")
    data, output = tmp_path / "data", tmp_path / "report"
    generate(data, batches=2, units_per_batch=1)
    run(data, output)
    reliability = json.loads((output / "reliability.json").read_text())
    assert reliability["status"] == "skipped"
    assert reliability["reason"]
    assert "weibull" not in reliability
    assert any(
        row["status"] == "skipped" for row in json.loads((output / "metrics.json").read_text())
    )


def test_integrity_and_output_protection(tmp_path):
    root = tmp_path / "data"
    generate(root, batches=2, units_per_batch=2)
    with pytest.raises(ValueError, match="Output exists"):
        generate(root)
    with (root / "packaging_candidates.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(root)
    with pytest.raises(ValueError):
        generate(tmp_path / "bad", batches=1)
    assert not (tmp_path / "bad").exists()


def test_force_refuses_an_unmanaged_output_directory(tmp_path):
    target = tmp_path / "manual-results"
    target.mkdir()
    (target / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="unmanaged"):
        generate(target, force=True)
    assert (target / "notes.txt").read_text() == "keep me"


def test_demo_generate_cli_and_force(tmp_path):
    runner = CliRunner()
    args = [
        "demo",
        "generate",
        "--output-dir",
        str(tmp_path / "data"),
        "--batches",
        "3",
        "--units-per-batch",
        "10",
    ]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, args).exit_code != 0
    assert runner.invoke(app, [*args, "--force"]).exit_code == 0


def test_generated_probability_labels_are_not_features(tmp_path):
    generate(tmp_path / "data", batches=5, units_per_batch=20)
    _, tables = load_dataset(tmp_path / "data")

    from semiyield.manufacturing.modeling import train_manufacturing

    frame = tables["manufacturing"]
    artifact = train_manufacturing(frame[SENSORS], frame.failed)
    assert artifact.feature_columns == SENSORS
    assert np.isfinite(artifact.estimator.predict_proba(frame[SENSORS])).all()


def test_force_removes_old_successful_artifacts_when_new_fit_is_skipped(tmp_path):
    pytest.importorskip("matplotlib")
    data, output = tmp_path / "data", tmp_path / "report"
    generate(data, batches=10, units_per_batch=30)
    run(data, output)
    assert (output / "weibull_curve.csv").exists()
    assert (output / "packaging/logistic.joblib").exists()
    generate(data, batches=2, units_per_batch=1, force=True)
    run(data, output, force=True)
    assert not (output / "weibull_curve.csv").exists()
    assert not (output / "packaging/logistic.joblib").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert "weibull_curve.csv" not in manifest["artifacts"]
