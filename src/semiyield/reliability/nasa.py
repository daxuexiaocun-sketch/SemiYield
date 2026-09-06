"""Auditable preparation of NASA MOSFET thermal-overstress data.

The upstream archive is large and its MATLAB structure is not bundled here.  Preparation
accepts either a directory of normalized CSV/Parquet files or a ZIP containing such files.
MATLAB files can be inventoried; conversion requires a mapping after inspecting the archive.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from semiyield.common.artifacts import sha256_file
from semiyield.constants import (
    MOSFET_FEATURE_SCHEMA_VERSION,
    NASA_DATASET_CITATION,
    NASA_MOSFET_URL,
    RANDOM_STATE,
)
from semiyield.reliability.reporting import write_degradation_chart

LICENSE_NAMES = {"license", "license.txt", "copying", "notice", "readme", "readme.txt"}


def download_archive(
    output: str | Path = "data/external/nasa/mosfet_thermal_overstress.zip",
    *,
    url: str = NASA_MOSFET_URL,
    force: bool = False,
) -> dict[str, object]:
    """Download the official archive to an ignored local cache and hash it."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return verify_archive(destination)
    partial = destination.with_suffix(destination.suffix + ".part")
    if force:
        partial.unlink(missing_ok=True)

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "SemiYield/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.getcode()
        if offset and status != 206:
            offset = 0
        content_length = response.headers.get("Content-Length")
        expected_bytes = offset + int(content_length) if content_length else None
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)

    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise OSError(
            f"NASA archive download was incomplete; rerun the command to resume from {partial}"
        )
    verify_archive(partial)
    partial.replace(destination)
    return verify_archive(destination)


def inspect_archive_notices(archive: str | Path) -> dict[str, object]:
    """Return small notice/readme files; absence never implies redistribution rights."""
    path = Path(archive)
    notices = []
    with zipfile.ZipFile(path) as zipped:
        for member in zipped.infolist():
            name = Path(member.filename).name.lower()
            if member.is_dir() or member.file_size > 2_000_000:
                continue
            if name in LICENSE_NAMES or "license" in name or "readme" in name:
                raw = zipped.read(member)
                notices.append(
                    {
                        "name": member.filename,
                        "sha256": __import__("hashlib").sha256(raw).hexdigest(),
                        "text_preview": raw.decode("utf-8", errors="replace")[:20_000],
                    }
                )
    return {
        "archive_sha256": sha256_file(path),
        "notices": notices,
        "redistribution_cleared": False,
        "interpretation": (
            "No automated license inference is made. Dataset-specific permission or an "
            "explicit archive notice is required before publishing row-level derivatives."
        ),
    }


@dataclass(frozen=True)
class SignalMapping:
    device: str = "device_id"
    time: str = "time_s"
    temperature: str = "temperature_c"
    voltage: str = "vds_v"
    current: str = "id_a"
    event: str | None = "event_observed"


def verify_archive(archive: str | Path, expected_sha256: str | None = None) -> dict[str, object]:
    """Validate a local NASA archive without extracting it."""
    path = Path(archive)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}")
    if not zipfile.is_zipfile(path):
        raise ValueError("NASA archive is not a valid ZIP file")
    with zipfile.ZipFile(path) as zipped:
        members = [item for item in zipped.infolist() if not item.is_dir()]
        suffix_counts: dict[str, int] = {}
        for member in members:
            suffix = Path(member.filename).suffix.lower() or "<none>"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        return {
            "archive": str(path.resolve()),
            "sha256": actual_hash,
            "compressed_bytes": path.stat().st_size,
            "uncompressed_bytes": sum(item.file_size for item in members),
            "file_count": len(members),
            "file_types": suffix_counts,
            "sample_members": [item.filename for item in members[:20]],
            "source": NASA_MOSFET_URL,
            "citation": NASA_DATASET_CITATION,
        }


def _device_from_name(path: Path) -> str:
    match = re.search(r"(?:test|device|mosfet)[_-]?(\d+)", path.stem, re.IGNORECASE)
    return f"device_{int(match.group(1)):03d}" if match else path.stem


def inspect_matlab_file(path: str | Path) -> dict[str, object]:
    """List top-level MATLAB variables without loading large experiment arrays."""
    from scipy.io import whosmat

    source = Path(path)
    if source.suffix.lower() != ".mat":
        raise ValueError("Expected a .mat file")
    try:
        variables = [
            {"name": name, "shape": list(shape), "matlab_class": matlab_class}
            for name, shape, matlab_class in whosmat(source)
        ]
    except (NotImplementedError, ValueError) as exc:
        raise ValueError(
            "Unable to inspect this MATLAB release with scipy. MATLAB v7.3/HDF5 files "
            "require an explicit h5py mapping."
        ) from exc
    return {"file": str(source.resolve()), "bytes": source.stat().st_size, "variables": variables}


def _nested_value(container, dotted_path: str):
    current = container
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(dotted_path)
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        elif (
            isinstance(current, np.ndarray) and current.dtype.names and part in current.dtype.names
        ):
            current = current[part]
        else:
            raise KeyError(dotted_path)
    return np.asarray(current).squeeze()


def convert_matlab_directory(
    source_dir: str | Path,
    *,
    output_dir: str | Path,
    mapping: dict[str, str | None],
) -> dict[str, object]:
    """Convert explicitly mapped MATLAB slow channels to normalized CSV files."""
    from scipy.io import loadmat

    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    required = {"time", "temperature", "voltage", "current"}
    missing_mapping = sorted(key for key in required if not mapping.get(key))
    if missing_mapping:
        raise ValueError(f"MATLAB mapping is missing: {', '.join(missing_mapping)}")
    converted = []
    for mat_path in sorted(source.rglob("*.mat")):
        try:
            content = loadmat(mat_path, simplify_cells=True)
        except NotImplementedError as exc:
            raise ValueError(
                f"{mat_path.name} is MATLAB v7.3/HDF5; export its mapped slow channels "
                "with MATLAB or h5py before using SemiYield"
            ) from exc
        arrays = {
            key: _nested_value(content, path)
            for key, path in mapping.items()
            if path and key in {"time", "temperature", "voltage", "current", "event"}
        }
        vector_lengths = [value.size for value in arrays.values() if value.size > 1]
        if not vector_lengths:
            raise ValueError(f"Mapped channels in {mat_path.name} contain no vectors")
        length = min(vector_lengths)

        def aligned(name: str, source_arrays=arrays, target_length=length):
            value = source_arrays[name]
            if value.ndim == 0 or value.size == 1:
                return np.repeat(float(value), target_length)
            return np.ravel(value)[:target_length]

        device_path = mapping.get("device")
        device = (
            str(_nested_value(content, device_path).item())
            if device_path
            else _device_from_name(mat_path)
        )
        normalized = pd.DataFrame(
            {
                "device_id": device,
                "time_s": aligned("time"),
                "temperature_c": aligned("temperature"),
                "vds_v": aligned("voltage"),
                "id_a": aligned("current"),
            }
        )
        if "event" in arrays:
            normalized["event_observed"] = aligned("event").astype(bool)
        else:
            normalized["event_observed"] = False
        destination = output / f"{mat_path.stem}.csv"
        normalized.to_csv(destination, index=False)
        converted.append(
            {
                "source": str(mat_path.relative_to(source)),
                "source_sha256": sha256_file(mat_path),
                "output": destination.name,
                "output_sha256": sha256_file(destination),
                "rows": len(normalized),
                "device_id": device,
            }
        )
    if not converted:
        raise ValueError("No MATLAB files found")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mapping": mapping,
        "files": converted,
        "license_status": "redistribution-not-cleared",
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _transient_rows(measurement: dict, device: str, run_id: int | None = None) -> pd.DataFrame:
    """Reduce every switching transient to physically meaningful on-state measurements."""
    transient = measurement.get("transient", [])
    steady = measurement.get("steadyState", [])
    if isinstance(transient, dict):
        transient = [transient]
    if isinstance(steady, dict):
        steady = [steady]
    steady_time = np.asarray([row.get("timeEpoch", np.nan) for row in steady], dtype=float)
    steady_temp = np.asarray(
        [row.get("timeDomain", {}).get("packageTemperature", np.nan) for row in steady],
        dtype=float,
    )
    valid_steady = np.isfinite(steady_time) & np.isfinite(steady_temp)
    steady_time, steady_temp = steady_time[valid_steady], steady_temp[valid_steady]
    order = np.argsort(steady_time)
    steady_time, steady_temp = steady_time[order], steady_temp[order]
    epochs = np.asarray([row.get("timeEpoch", np.nan) for row in transient], dtype=float)
    finite_epoch = epochs[np.isfinite(epochs)]
    if not len(finite_epoch):
        raise ValueError("Transient records contain no finite timeEpoch")
    start = float(finite_epoch.min())
    rows = []
    for record, epoch in zip(transient, epochs, strict=True):
        domain = record.get("timeDomain", {})
        gate = np.ravel(np.asarray(domain.get("gateSourceVoltage", []), dtype=float))
        voltage = np.ravel(np.asarray(domain.get("drainSourceVoltage", []), dtype=float))
        current = np.ravel(np.asarray(domain.get("drainCurrent", []), dtype=float))
        length = min(len(gate), len(voltage), len(current))
        if not np.isfinite(epoch) or length < 10:
            continue
        gate, voltage, current = gate[:length], voltage[:length], current[:length]
        gate_threshold = (np.nanpercentile(gate, 10) + np.nanpercentile(gate, 90)) / 2
        on = (gate >= gate_threshold) & np.isfinite(voltage) & np.isfinite(current)
        # The archive contains setup/open-load runs around 0.3 A. They are retained in the
        # conversion audit but cannot support an on-resistance degradation claim.
        on &= np.abs(current) >= 1.0
        if on.sum() < 3:
            continue
        median_voltage = float(np.nanmedian(voltage[on]))
        median_current = float(np.nanmedian(current[on]))
        temperature = np.nan
        if len(steady_time):
            nearest = int(np.argmin(np.abs(steady_time - epoch)))
            temperature = float(steady_temp[nearest])
        rows.append(
            {
                "device_id": device,
                "run_id": run_id,
                "time_s": float((epoch - start) * 86400.0),
                "temperature_c": temperature,
                "vds_v": median_voltage,
                "id_a": median_current,
                "event_observed": False,
            }
        )
    if not rows:
        raise ValueError("No valid on-state transient measurements")
    return pd.DataFrame(rows)


def convert_nasa_matlab_archive(
    archive: str | Path,
    *,
    output_dir: str | Path = "data/interim/nasa_mosfet_normalized",
) -> dict[str, object]:
    """Stream all identifiable NASA run files to compact normalized transient tables."""
    from scipy.io import loadmat

    source = Path(archive)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files, skipped = [], []
    pattern = re.compile(r"Test_(\d+)_run_(\d+)\.mat$", re.IGNORECASE)
    with zipfile.ZipFile(source) as outer:
        inner_members = [
            member
            for member in outer.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".zip")
        ]
        if not inner_members:
            raise ValueError("Official NASA archive contains no inner experiment ZIP")
        inner_member = max(inner_members, key=lambda member: member.file_size)
        with (
            outer.open(inner_member) as inner_handle,
            zipfile.ZipFile(inner_handle) as zipped,
            tempfile.TemporaryDirectory(prefix="semiyield-nasa-mat-") as temporary,
        ):
            for member in sorted(zipped.infolist(), key=lambda item: item.filename):
                match = pattern.search(Path(member.filename).name)
                if member.is_dir() or not match:
                    continue
                device = f"device_{int(match.group(1)):03d}"
                run = int(match.group(2))
                temporary_path = Path(temporary) / Path(member.filename).name
                digest = hashlib.sha256()
                with zipped.open(member) as input_handle, temporary_path.open("wb") as handle:
                    while block := input_handle.read(8 * 1024 * 1024):
                        digest.update(block)
                        handle.write(block)
                try:
                    content = loadmat(temporary_path, simplify_cells=True)
                    frame = _transient_rows(content["measurement"], device, run_id=run)
                    destination = output / f"{device}_run_{run:03d}.csv"
                    frame.to_csv(destination, index=False)
                    files.append(
                        {
                            "source": member.filename,
                            "source_sha256": digest.hexdigest(),
                            "output": destination.name,
                            "output_sha256": sha256_file(destination),
                            "rows": len(frame),
                            "device_id": device,
                            "run": run,
                        }
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    skipped.append({"source": member.filename, "reason": str(exc)})
                temporary_path.unlink(missing_ok=True)
    if not files:
        raise ValueError("No identifiable NASA Test_<device>_run_<n>.mat files were converted")
    manifest = {
        "schema_version": "nasa-mosfet-matlab-conversion-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha256_file(source),
        "inner_archive": inner_member.filename,
        "method": "median on-state VDS/ID per switching transient; nearest package temperature",
        "files": files,
        "skipped": skipped,
        "devices": sorted({item["device_id"] for item in files}),
        "license_status": "redistribution-not-cleared",
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported normalized input: {path}")


def _iter_normalized_tables(source: Path) -> Iterable[tuple[Path, pd.DataFrame]]:
    supported = {".csv", ".parquet", ".pq"}
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() in supported:
                yield path, _read_table(path)
        return
    if not zipfile.is_zipfile(source):
        raise ValueError("Input must be a directory or ZIP archive")
    with tempfile.TemporaryDirectory(prefix="semiyield-nasa-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(source) as zipped:
            safe_members = [
                member
                for member in zipped.infolist()
                if not member.is_dir()
                and Path(member.filename).suffix.lower() in supported
                and ".." not in Path(member.filename).parts
            ]
            if not safe_members:
                raise ValueError(
                    "Archive contains no normalized CSV/Parquet tables. "
                    "Run `semiyield reliability nasa inspect` and convert the MATLAB "
                    "slow-measurement "
                    "structures using an explicit mapping before preparation."
                )
            for member in safe_members:
                target = root / Path(member.filename).name
                with zipped.open(member) as source_handle, target.open("wb") as output:
                    while block := source_handle.read(1024 * 1024):
                        output.write(block)
                yield target, _read_table(target)


def _slope(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(numeric), dtype=float), numeric, 1)[0])


def aggregate_signals(
    frame: pd.DataFrame,
    *,
    mapping: SignalMapping | None = None,
    window_size: int = 1000,
    fallback_device: str = "device_unknown",
) -> pd.DataFrame:
    """Aggregate normalized slow measurements into auditable aging-window features."""
    mapping = mapping or SignalMapping()
    required = [mapping.time, mapping.temperature, mapping.voltage, mapping.current]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing normalized signal columns: {', '.join(missing)}")
    working = frame.copy()
    if mapping.device not in working:
        working[mapping.device] = fallback_device
    working = working.sort_values([mapping.device, mapping.time]).reset_index(drop=True)
    current = pd.to_numeric(working[mapping.current], errors="coerce")
    voltage = pd.to_numeric(working[mapping.voltage], errors="coerce")
    working["rds_on_raw_ohm"] = np.where(current.abs() > 1e-12, voltage / current, np.nan)
    output = []
    signals = {
        "temperature_c": mapping.temperature,
        "vds_v": mapping.voltage,
        "id_a": mapping.current,
        "rds_on_raw_ohm": "rds_on_raw_ohm",
        "rds_on_ohm": "rds_on_ohm",
    }
    for device_id, device_frame in working.groupby(mapping.device, sort=True):
        device_frame = device_frame.reset_index(drop=True).copy()
        temperatures = pd.to_numeric(device_frame[mapping.temperature], errors="coerce")
        reference_temperature = float(temperatures.median())
        device_frame["rds_temperature_coefficient_ohm_per_c"] = 0.0
        device_frame["rds_on_ohm"] = device_frame["rds_on_raw_ohm"]
        correction_groups = (
            device_frame.groupby("run_id", dropna=False).groups
            if "run_id" in device_frame
            else {"device": device_frame.index}
        )
        for indices in correction_groups.values():
            run_temp = temperatures.loc[indices].to_numpy(dtype=float)
            run_rds = pd.to_numeric(
                device_frame.loc[indices, "rds_on_raw_ohm"], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(run_temp) & np.isfinite(run_rds)
            coefficient = 0.0
            if valid.sum() >= 20 and np.ptp(run_temp[valid]) >= 10.0:
                coefficient = float(np.polyfit(run_temp[valid], run_rds[valid], 1)[0])
            device_frame.loc[indices, "rds_temperature_coefficient_ohm_per_c"] = coefficient
            device_frame.loc[indices, "rds_on_ohm"] = run_rds - coefficient * (
                run_temp - reference_temperature
            )
        if "run_id" in device_frame:
            first_run = device_frame["run_id"].dropna().min()
            baseline_values = device_frame.loc[device_frame["run_id"].eq(first_run), "rds_on_ohm"]
        else:
            baseline_values = device_frame["rds_on_ohm"].head(window_size)
        baseline = float(pd.to_numeric(baseline_values, errors="coerce").median())
        window_count = int(np.ceil(len(device_frame) / window_size))
        for window_index in range(window_count):
            window = device_frame.iloc[
                window_index * window_size : (window_index + 1) * window_size
            ]
            row: dict[str, object] = {
                "device_id": str(device_id),
                "window_index": window_index,
                "sample_count": len(window),
                "time_start_s": float(pd.to_numeric(window[mapping.time], errors="coerce").min()),
                "time_end_s": float(pd.to_numeric(window[mapping.time], errors="coerce").max()),
                "event_observed": bool(
                    window_index == window_count - 1
                    and (
                        True
                        if not mapping.event or mapping.event not in device_frame
                        else bool(device_frame[mapping.event].iloc[-1])
                    )
                ),
                "rds_temperature_coefficient_ohm_per_c": float(
                    window["rds_temperature_coefficient_ohm_per_c"].median()
                ),
                "rds_reference_temperature_c": reference_temperature,
            }
            for feature_name, column in signals.items():
                values = pd.to_numeric(window[column], errors="coerce")
                row.update(
                    {
                        f"{feature_name}_mean": float(values.mean()),
                        f"{feature_name}_std": float(values.std(ddof=0)),
                        f"{feature_name}_min": float(values.min()),
                        f"{feature_name}_max": float(values.max()),
                        f"{feature_name}_p95": float(values.quantile(0.95)),
                        f"{feature_name}_slope": _slope(values),
                    }
                )
            row["rds_on_delta_ohm"] = float(row["rds_on_ohm_mean"] - baseline)
            output.append(row)
    return pd.DataFrame(output)


def split_devices(
    device_ids: Iterable[str],
    *,
    random_state: int = RANDOM_STATE,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, list[str]]:
    devices = np.asarray(sorted(set(map(str, device_ids))), dtype=object)
    if len(devices) < 3:
        raise ValueError("At least three devices are required for device-level splitting")
    shuffled = np.random.default_rng(random_state).permutation(devices)
    train_end = max(1, int(len(shuffled) * train_fraction))
    validation_end = max(train_end + 1, int(len(shuffled) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(shuffled) - 1)
    return {
        "train": sorted(shuffled[:train_end].tolist()),
        "validation": sorted(shuffled[train_end:validation_end].tolist()),
        "test": sorted(shuffled[validation_end:].tolist()),
    }


def derive_lifetime_table(
    features: pd.DataFrame,
    *,
    threshold_ohm: float = 0.05,
    smoothing_windows: int = 5,
    persistence_windows: int = 3,
    baseline_feature_windows: int = 5,
    splits: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Create one right-censored lifetime row per device from RDS(on) degradation."""
    required = {"device_id", "time_end_s", "rds_on_delta_ohm"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Missing derived feature columns: {', '.join(missing)}")
    if threshold_ohm <= 0:
        raise ValueError("threshold_ohm must be positive")
    if smoothing_windows < 1 or persistence_windows < 1 or baseline_feature_windows < 1:
        raise ValueError("window counts must be positive")
    split_lookup = {
        str(device): split
        for split, devices in (splits or split_devices(features["device_id"])).items()
        for device in devices
    }
    rows = []
    for device, group in features.sort_values("time_end_s").groupby("device_id", sort=True):
        group = group.copy()
        delta = pd.to_numeric(group["rds_on_delta_ohm"], errors="coerce")
        filtered = delta.rolling(smoothing_windows, min_periods=1, center=True).median()
        above = filtered >= threshold_ohm
        persistent = (
            above.rolling(persistence_windows, min_periods=persistence_windows)
            .sum()
            .eq(persistence_windows)
        )
        reached_positions = np.flatnonzero(persistent.to_numpy())
        event = bool(len(reached_positions))
        endpoint_position = (
            max(0, int(reached_positions[0]) - persistence_windows + 1) if event else len(group) - 1
        )
        endpoint = group.iloc[endpoint_position]
        row = {
            "unit_id": str(device),
            "time_to_event": float(endpoint["time_end_s"]),
            "event_observed": event,
            "threshold_ohm": float(threshold_ohm),
            "split": split_lookup[str(device)],
            "smoothing_windows": smoothing_windows,
            "persistence_windows": persistence_windows,
            "baseline_feature_windows": baseline_feature_windows,
        }
        baseline_features = group.head(baseline_feature_windows)
        for column in ("temperature_c_mean", "rds_on_ohm_mean", "rds_on_ohm_slope"):
            if column in group:
                row[f"baseline_{column}"] = float(
                    pd.to_numeric(baseline_features[column], errors="coerce").median()
                )
        if "baseline_temperature_c_mean" in row:
            row["temperature_c"] = row["baseline_temperature_c_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_nasa_features(
    source: str | Path,
    *,
    output_dir: str | Path = "data/processed/nasa_mosfet",
    mapping: SignalMapping | None = None,
    window_size: int = 1000,
) -> dict[str, object]:
    """Prepare all normalized tables and emit features, splits, dictionary, and provenance."""
    source_path = Path(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized_frames = []
    input_files = []
    device_offsets: dict[str, float] = {}
    for path, frame in _iter_normalized_tables(source_path):
        active_mapping = mapping or SignalMapping()
        table = frame.copy()
        if active_mapping.device not in table:
            table[active_mapping.device] = _device_from_name(path)
        if "run_id" not in table:
            run_match = re.search(r"run[_-]?(\d+)", path.stem, re.IGNORECASE)
            if run_match:
                table["run_id"] = int(run_match.group(1))
        for device_id, indices in table.groupby(active_mapping.device).groups.items():
            times = pd.to_numeric(table.loc[indices, active_mapping.time], errors="raise")
            positive_steps = times.sort_values().diff().loc[lambda values: values > 0]
            step = float(positive_steps.median()) if len(positive_steps) else 1.0
            offset = device_offsets.get(str(device_id), 0.0)
            shifted = times - float(times.min()) + offset
            table.loc[indices, active_mapping.time] = shifted
            device_offsets[str(device_id)] = float(shifted.max()) + step
        normalized_frames.append(table)
        input_files.append({"name": path.name, "rows": len(frame), "sha256": sha256_file(path)})
    if not normalized_frames:
        raise ValueError("No normalized tables were found")
    features = aggregate_signals(
        pd.concat(normalized_frames, ignore_index=True),
        mapping=mapping,
        window_size=window_size,
    ).sort_values(["device_id", "window_index"])
    feature_path = output / "mosfet_features.parquet"
    try:
        features.to_parquet(feature_path, index=False)
    except ImportError as exc:
        raise ImportError("Install `semiyield[parquet]` to write derived features") from exc
    splits = split_devices(features["device_id"])
    split_path = output / "split_manifest.json"
    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    dictionary_path = output / "feature_dictionary.md"
    dictionary_path.write_text(_feature_dictionary(), encoding="utf-8")
    source_hash = sha256_file(source_path) if source_path.is_file() else None
    conversion_manifest_path = source_path / "conversion_manifest.json"
    conversion_provenance = (
        json.loads(conversion_manifest_path.read_text(encoding="utf-8"))
        if conversion_manifest_path.exists()
        else None
    )
    provenance = {
        "schema_version": MOSFET_FEATURE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": NASA_MOSFET_URL,
        "citation": NASA_DATASET_CITATION,
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "source_files": input_files,
        "conversion_manifest": conversion_provenance,
        "mapping": asdict(mapping or SignalMapping()),
        "window_size": window_size,
        "derived_sha256": sha256_file(feature_path),
        "license_status": "redistribution-not-cleared",
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    lifetime_files = {}
    for threshold in (0.045, 0.05):
        lifetime = derive_lifetime_table(features, threshold_ohm=threshold, splits=splits)
        lifetime_path = output / f"mosfet_lifetime_{threshold:.3f}ohm.csv"
        lifetime.to_csv(lifetime_path, index=False)
        lifetime_files[lifetime_path.name] = sha256_file(lifetime_path)
    provenance["lifetime_sensitivity"] = lifetime_files
    chart_path = write_degradation_chart(features, output / "degradation_trends.svg")
    if chart_path:
        provenance["degradation_chart_sha256"] = sha256_file(chart_path)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"features": features, "splits": splits, "provenance": provenance}


def _feature_dictionary() -> str:
    return """# NASA MOSFET derived feature dictionary

This schema is produced from normalized slow measurements. It is not raw NASA data.

| Field | Meaning |
|---|---|
| `device_id` | Stable device identifier used for leakage-safe splitting |
| `window_index` | Sequential aging window within a device |
| `time_start_s`, `time_end_s` | Window bounds in seconds |
| `temperature_c_*` | Temperature distribution and trend statistics |
| `vds_v_*` | Drain-source voltage statistics |
| `id_a_*` | Drain current statistics |
| `rds_on_raw_ohm_*` | Raw on-state resistance computed as VDS/ID |
| `rds_on_ohm_*` | RDS(on) corrected to the device's initial median temperature |
| `rds_temperature_coefficient_ohm_per_c` | Initial-window linear temperature coefficient |
| `rds_on_delta_ohm` | Corrected change from the device's initial median RDS(on) |
| `event_observed` | Source event marker; absence never turns a file ending into failure |
"""
