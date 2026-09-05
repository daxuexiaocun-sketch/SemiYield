"""Process-level resource guard for memory-intensive benchmark commands."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep

import pandas as pd
import psutil

from semiyield.common.artifacts import sha256_file


@dataclass(frozen=True)
class ResourceResult:
    schema_version: str
    created_at: str
    status: str
    exit_code: int
    peak_rss_bytes: int
    soft_limit_bytes: int
    hard_limit_bytes: int
    soft_limit_exceeded: bool
    elapsed_seconds: float
    models: list[str]


def _process_tree_rss(process: psutil.Process) -> int:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for item in processes:
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:  # pragma: no cover - exercised on Windows CI only
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on Windows CI only
            process.kill()
        process.wait(timeout=5)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _record_failed_models(output_dir: Path, models: list[str], status: str, reason: str) -> None:
    status_path = output_dir / "model_status.csv"
    if status_path.exists():
        rows = pd.read_csv(status_path).fillna("")
        rows = rows.loc[~rows["model"].isin(models)]
    else:
        rows = pd.DataFrame(columns=["model", "status", "reason"])
    failed = pd.DataFrame(
        [{"model": model, "status": status, "reason": reason} for model in models]
    )
    pd.concat([rows, failed], ignore_index=True).to_csv(status_path, index=False)


def _attach_guard_to_manifest(output_dir: Path, report: dict[str, object]) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resource_guard"] = report
    resource_path = output_dir / "resource_usage.json"
    manifest.setdefault("artifacts", {})[resource_path.name] = sha256_file(resource_path)
    _write_json_atomic(manifest_path, manifest)


def run_guarded(
    command: list[str],
    *,
    output_dir: str | Path,
    models: list[str],
    soft_limit_gb: float = 32.0,
    hard_limit_gb: float = 48.0,
    poll_seconds: float = 0.25,
) -> ResourceResult:
    """Run a command in a process group and stop its tree at the RSS hard limit."""
    if not 0 < soft_limit_gb < hard_limit_gb:
        raise ValueError("memory limits must satisfy 0 < soft < hard")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    soft_bytes = int(soft_limit_gb * 1024**3)
    hard_bytes = int(hard_limit_gb * 1024**3)
    started = monotonic()
    process = subprocess.Popen(command, start_new_session=os.name == "posix")
    tracked = psutil.Process(process.pid)
    peak = 0
    warned = False
    hard_exceeded = False
    while process.poll() is None:
        rss = _process_tree_rss(tracked)
        peak = max(peak, rss)
        if rss >= soft_bytes and not warned:
            warned = True
            print(
                f"WARNING: benchmark RSS reached {rss / 1024**3:.2f} GB "
                f"(soft limit {soft_limit_gb:g} GB)",
                file=sys.stderr,
                flush=True,
            )
        if rss >= hard_bytes:
            hard_exceeded = True
            print(
                f"ERROR: benchmark RSS reached {rss / 1024**3:.2f} GB; "
                f"terminating at the {hard_limit_gb:g} GB hard limit",
                file=sys.stderr,
                flush=True,
            )
            _terminate_process_group(process)
            break
        sleep(poll_seconds)
    exit_code = process.wait()
    if hard_exceeded:
        status = "memory_limited"
        exit_code = 137
    elif exit_code == 0:
        status = "completed"
    else:
        status = "failed"
    result = ResourceResult(
        schema_version="semiyield-resource-usage-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        exit_code=exit_code,
        peak_rss_bytes=peak,
        soft_limit_bytes=soft_bytes,
        hard_limit_bytes=hard_bytes,
        soft_limit_exceeded=warned,
        elapsed_seconds=monotonic() - started,
        models=models,
    )
    report = asdict(result)
    resource_path = output / "resource_usage.json"
    _write_json_atomic(resource_path, report)
    if status != "completed":
        reason = (
            f"RSS exceeded {hard_limit_gb:g} GB hard limit"
            if status == "memory_limited"
            else f"guarded worker exited with code {exit_code}"
        )
        _record_failed_models(output, models, status, reason)
    _attach_guard_to_manifest(output, report)
    return result
