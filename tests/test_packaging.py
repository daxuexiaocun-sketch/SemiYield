import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, train_test_split
from typer.testing import CliRunner

from semiyield.cli import app
from semiyield.common.metrics import classification_report
from semiyield.packaging.data import FEATURES, validate_data
from semiyield.packaging.labeling import proxy_labels, resolve_threshold
from semiyield.packaging.modeling import PackagingArtifact, train_model
from semiyield.packaging.evaluate import run_benchmark, train_holdout


@pytest.fixture
def process_frame():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({f"X{i}": rng.choice(["a", "b"], 150) for i in range(1, 6)})
    for i in range(6, 17):
        frame[f"X{i}"] = rng.normal(size=150)
    frame["Y"] = np.exp(5 + frame.X6)
    frame["unit_id"] = [f"u{i}" for i in range(len(frame))]
    return frame


def test_strict_boundary_and_invalid_thresholds():
    np.testing.assert_array_equal(proxy_labels([9, 10, 11], 10), [1, 0, 0])
    for kwargs in [
        {"threshold": 2, "quantile": 0.1},
        {"quantile": 0},
        {"quantile": 1},
        {"threshold": np.inf},
        {"quantile": np.nan},
    ]:
        with pytest.raises(ValueError):
            resolve_threshold([1, 2, 3], **kwargs)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "bad"])
def test_invalid_target_rejected(process_frame, value):
    process_frame["Y"] = process_frame.Y.astype(object)
    process_frame.loc[0, "Y"] = value
    with pytest.raises(ValueError):
        validate_data(process_frame)


def test_holdout_labels_never_fit_threshold(process_frame, tmp_path):
    source = tmp_path / "source.csv"
    process_frame.to_csv(source, index=False)
    first = train_holdout(source, tmp_path / "one.joblib")
    train, test = train_test_split(process_frame, test_size=0.2, random_state=42)
    assert first.throughput_threshold == pytest.approx(train.Y.quantile(0.1))
    process_frame.loc[test.index, "Y"] = 1e12
    process_frame.to_csv(source, index=False)
    second = train_holdout(source, tmp_path / "two.joblib")
    assert second.throughput_threshold == first.throughput_threshold
    np.testing.assert_allclose(
        first.predict(test).proxy_failure_probability,
        second.predict(test).proxy_failure_probability,
    )
    assert second.metadata["holdout_metrics"]["roc_auc"] is None
    assert second.metadata["holdout_metrics"]["undefined_reasons"]


def test_mixed_schema_unknown_categories_and_roundtrip(process_frame, tmp_path):
    process_frame.loc[0, "X1"] = np.nan
    process_frame.loc[0, "X6"] = np.nan
    artifact = train_model(process_frame)
    assert artifact.feature_columns == FEATURES
    inference = process_frame.iloc[:5].drop(columns="Y").copy()
    inference["X1"] = "never_seen"
    before = artifact.predict(inference)
    assert np.isfinite(before.proxy_failure_probability).all()
    assert before.unit_id.tolist() == inference.unit_id.tolist()
    path = tmp_path / "model.joblib"
    artifact.save(path)
    pd.testing.assert_frame_equal(PackagingArtifact.load(path).predict(inference), before)
    with pytest.raises(ValueError, match="Missing packaging columns"):
        artifact.predict(inference.drop(columns="X6"))
    with pytest.raises(ValueError, match="both proxy"):
        train_model(process_frame, threshold=0)


def test_benchmark_matched_training_thresholds(process_frame, tmp_path):
    source = tmp_path / "source.csv"
    process_frame.to_csv(source, index=False)
    output = tmp_path / "benchmark"
    run_benchmark(source, output)
    manifest = json.loads((output / "manifest.json").read_text())
    for fold, (train, _) in enumerate(
        KFold(3, shuffle=True, random_state=42).split(process_frame), 1
    ):
        rows = [r for r in manifest["folds"] if r["fold"] == fold]
        assert {r["model"] for r in rows} == {"dummy", "logistic"}
        for row in rows:
            assert row["threshold"]["value"] == pytest.approx(
                process_frame.iloc[train].Y.quantile(0.1)
            )
    assert len(pd.read_csv(output / "fold_metrics.csv")) == 6


def test_cli_rejects_conflicting_thresholds(process_frame, tmp_path):
    source = tmp_path / "input.csv"
    process_frame.to_csv(source, index=False)
    result = CliRunner().invoke(
        app,
        [
            "packaging",
            "train",
            "--input-csv",
            str(source),
            "--threshold",
            "10",
            "--quantile",
            "0.1",
        ],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_undefined_metrics_are_null_with_reasons():
    report = classification_report([0, 0], [0.1, 0.2])
    assert report["roc_auc"] is None
    assert report["recall"] is None
    assert report["tn"] == 2
    assert "recall" in report["undefined_reasons"]
    json.dumps(report, allow_nan=False)
