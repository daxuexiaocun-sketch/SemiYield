from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from semiyield.common.reporting import data_quality_report, write_pr_auc_benchmark_chart
from semiyield.manufacturing.data import download_secom, load_secom
from semiyield.manufacturing.preprocessing import validate_features


def test_quality_report_detects_problem_columns(sample_data):
    frame, target = sample_data
    report = data_quality_report(frame, target)
    assert "sensor_002" in report["constant_columns"]
    assert "sensor_003" in report["high_missing_columns"]
    assert report["failure_count"] == int(target.sum())


def test_pr_auc_benchmark_chart_uses_shared_order_and_metadata(tmp_path):
    path = tmp_path / "benchmark.svg"
    chart = write_pr_auc_benchmark_chart(
        pd.DataFrame(
            {
                "model": ["catboost", "dummy", "logistic"],
                "pr_auc_mean": [0.7, 0.1, 0.5],
                "pr_auc_std": [0.02, 0.01, 0.03],
            }
        ),
        path,
        title="Shared PR-AUC benchmark",
        description="Test benchmark chart.",
        y_max=0.5,
    )
    assert chart == path
    content = path.read_text(encoding="utf-8")
    assert "<title>Shared PR-AUC benchmark</title>" in content
    assert "<desc>Test benchmark chart.</desc>" in content
    assert "<!-- 0.5 -->" in content
    assert content.index("<!-- Dummy -->") < content.index("<!-- Logistic -->")
    assert content.index("<!-- Logistic -->") < content.index("<!-- Catboost -->")
    for color in ("#9aa9b8", "#2878b5", "#d47832"):
        assert color in content


def test_validate_reorders_and_warns_on_extra(sample_data):
    frame, _ = sample_data
    incoming = frame[["sensor_001", "sensor_000"]].assign(extra=1)
    validated, warnings = validate_features(incoming, ["sensor_000", "sensor_001"])
    assert list(validated) == ["sensor_000", "sensor_001"]
    assert warnings


def test_validate_rejects_missing_column(sample_data):
    frame, _ = sample_data
    with pytest.raises(ValueError, match="Missing required"):
        validate_features(frame, ["not_present"])


def test_download_rejects_hash_mismatch(tmp_path, monkeypatch):
    archive = BytesIO()
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("secom.data", "1 2\n")
        zipped.writestr("secom_labels.data", "-1 01/01/2008 00:00:00\n")

    def fake_download(url, destination, attempts=3):
        destination.write_bytes(archive.getvalue())

    monkeypatch.setattr("semiyield.manufacturing.data._download_file", fake_download)
    monkeypatch.setattr(
        "semiyield.manufacturing.data._download_individual_files",
        lambda destination: (_ for _ in ()).throw(RuntimeError("fallback failed")),
    )
    with pytest.raises(RuntimeError, match="fallback failed"):
        download_secom(tmp_path, expected_sha256="not-the-real-hash")


def test_load_secom_parses_quoted_timestamp(tmp_path):
    (tmp_path / "secom.data").write_text("1.0 NaN\n2.0 3.0\n", encoding="utf-8")
    (tmp_path / "secom_labels.data").write_text(
        '-1 "19/07/2008 11:55:00"\n1 "20/07/2008 12:32:00"\n', encoding="utf-8"
    )
    dataset = load_secom(tmp_path)
    assert list(dataset.target) == [0, 1]
    assert str(dataset.timestamps.iloc[0]) == "2008-07-19 11:55:00"
