import json
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from semiyield.reliability.nasa import (
    _transient_rows,
    aggregate_signals,
    convert_matlab_directory,
    convert_nasa_matlab_archive,
    derive_lifetime_table,
    inspect_matlab_file,
    prepare_nasa_features,
    split_devices,
    verify_archive,
)


def _signals(device: str, rows: int = 24):
    time = np.arange(rows, dtype=float)
    current = np.full(rows, 2.0)
    resistance = 0.03 + time * 0.0001
    return pd.DataFrame(
        {
            "device_id": device,
            "time_s": time,
            "temperature_c": 210 + np.sin(time),
            "vds_v": resistance * current,
            "id_a": current,
            "event_observed": [False] * (rows - 1) + [True],
        }
    )


def test_aggregate_signals_builds_rds_features():
    result = aggregate_signals(_signals("device_1"), window_size=8)
    assert len(result) == 3
    assert result.iloc[-1].event_observed
    assert result.rds_on_delta_ohm.iloc[-1] > 0


def test_device_split_has_no_overlap():
    split = split_devices([f"d{i}" for i in range(10)])
    groups = [set(split[name]) for name in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == {f"d{i}" for i in range(10)}


def test_lifetime_threshold_and_censoring():
    features = pd.DataFrame(
        {
            "device_id": ["a", "a", "b", "b", "c", "c"],
            "time_end_s": [10, 20, 10, 20, 10, 20],
            "rds_on_delta_ohm": [0.0, 0.06, 0.0, 0.03, 0.0, 0.07],
        }
    )
    splits = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    result = derive_lifetime_table(
        features,
        threshold_ohm=0.05,
        splits=splits,
        smoothing_windows=1,
        persistence_windows=1,
    )
    indexed = result.set_index("unit_id")
    assert indexed.loc["a", "event_observed"]
    assert not indexed.loc["b", "event_observed"]
    assert indexed.loc["c", "split"] == "test"


def test_transient_reduction_uses_on_state_and_nearest_temperature():
    measurement = {
        "transient": [
            {
                "timeEpoch": 10.0,
                "timeDomain": {
                    "gateSourceVoltage": [0] * 5 + [10] * 5,
                    "drainSourceVoltage": [5] * 5 + [0.2] * 5,
                    "drainCurrent": [0.01] * 5 + [2.0] * 5,
                },
            }
        ],
        "steadyState": [{"timeEpoch": 10.0, "timeDomain": {"packageTemperature": 210.0}}],
    }
    result = _transient_rows(measurement, "device_001")
    assert result.loc[0, "vds_v"] == pytest.approx(0.2)
    assert result.loc[0, "id_a"] == pytest.approx(2.0)
    assert result.loc[0, "temperature_c"] == pytest.approx(210.0)


def test_prepare_emits_provenance_and_splits(tmp_path):
    source = tmp_path / "normalized"
    source.mkdir()
    for index in range(5):
        _signals(f"device_{index}").to_csv(source / f"device_{index}.csv", index=False)
    try:
        result = prepare_nasa_features(source, output_dir=tmp_path / "out", window_size=8)
    except ImportError:
        pytest.skip("pyarrow is optional")
    assert result["features"].device_id.nunique() == 5
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert provenance["license_status"] == "redistribution-not-cleared"


def test_verify_archive(tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Test_1_run_1.mat", b"fixture")
    inventory = verify_archive(archive)
    assert inventory["file_types"] == {".mat": 1}


def test_download_keeps_partial_file_when_content_is_truncated(tmp_path):
    from io import BytesIO

    from semiyield.reliability.nasa import download_archive

    class Response(BytesIO):
        headers = {"Content-Length": "10"}

        def getcode(self):
            return 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    archive = tmp_path / "archive.zip"
    with patch("urllib.request.urlopen", return_value=Response(b"short")):
        with pytest.raises(OSError, match="incomplete"):
            download_archive(archive, url="https://example.invalid/archive.zip")
    assert not archive.exists()
    assert archive.with_suffix(".zip.part").read_bytes() == b"short"


def test_matlab_mapping_conversion(tmp_path):
    source = tmp_path / "matlab"
    source.mkdir()
    mat_path = source / "Test_7_run_1.mat"
    time = np.arange(12, dtype=float)
    savemat(
        mat_path,
        {
            "experiment": {
                "slow": {
                    "time": time,
                    "temperature": np.full(12, 220.0),
                    "voltage": 2 * (0.03 + time * 0.0001),
                    "current": np.full(12, 2.0),
                }
            }
        },
    )
    inspected = inspect_matlab_file(mat_path)
    assert inspected["variables"][0]["name"] == "experiment"
    mapping = {
        "device": None,
        "time": "experiment.slow.time",
        "temperature": "experiment.slow.temperature",
        "voltage": "experiment.slow.voltage",
        "current": "experiment.slow.current",
        "event": None,
    }
    manifest = convert_matlab_directory(source, output_dir=tmp_path / "normalized", mapping=mapping)
    converted = pd.read_csv(tmp_path / "normalized" / "Test_7_run_1.csv")
    assert manifest["files"][0]["device_id"] == "device_007"
    assert len(converted) == 12
    assert not converted.event_observed.any()


def test_official_conversion_reads_nested_experiment_zip(tmp_path, monkeypatch):
    mat_path = tmp_path / "Test_7_run_1.mat"
    savemat(
        mat_path,
        {
            "measurement": {
                "transient": [
                    {
                        "timeEpoch": 10.0,
                        "timeDomain": {
                            "gateSourceVoltage": [10.0],
                            "drainSourceVoltage": [0.2],
                            "drainCurrent": [2.0],
                        },
                    }
                ],
                "steadyState": [{"timeEpoch": 10.0, "timeDomain": {"packageTemperature": 210.0}}],
            }
        },
    )
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as zipped:
        zipped.write(mat_path, "dataset/Test_7_run_1.mat")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zipped:
        zipped.write(inner, "dataset/experiments.zip")
    monkeypatch.setattr(
        "semiyield.reliability.nasa._transient_rows",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "device_id": ["device_007"],
                "run_id": [1],
                "time_s": [0.0],
                "temperature_c": [210.0],
                "vds_v": [0.2],
                "id_a": [2.0],
                "event_observed": [False],
            }
        ),
    )
    manifest = convert_nasa_matlab_archive(outer, output_dir=tmp_path / "normalized")
    assert manifest["inner_archive"] == "dataset/experiments.zip"
    assert manifest["files"][0]["device_id"] == "device_007"
