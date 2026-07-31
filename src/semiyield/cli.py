"""SemiYield command-line interface."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from semiyield.benchmark import BenchmarkConfig, run_benchmark
from semiyield.constants import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH
from semiyield.data import (
    _download_file,
    data_quality_report,
    download_secom,
    load_secom,
    sha256_file,
)
from semiyield.drift import detect_drift
from semiyield.evaluate import evaluate_model
from semiyield.evidence import build_experiment_manifest
from semiyield.explain import explain_prediction
from semiyield.modeling import ModelArtifact, train_model
from semiyield.nasa import (
    SignalMapping,
    convert_matlab_directory,
    convert_nasa_matlab_archive,
    download_archive,
    inspect_archive_notices,
    inspect_matlab_file,
    prepare_nasa_features,
    verify_archive,
)
from semiyield.predict import predict_risk
from semiyield.reliability import validate_lifetime_data, write_reliability_report
from semiyield.resource_guard import run_guarded

app = typer.Typer(help="Semiconductor yield-risk and reliability analysis.")
data_app = typer.Typer(help="Manage example and licensed derived data.")
nasa_app = typer.Typer(help="Audit and prepare the external NASA MOSFET archive.")
reliability_app = typer.Typer(help="Fit lifetime and accelerated reliability models.")
app.add_typer(data_app, name="data")
app.add_typer(nasa_app, name="nasa")
app.add_typer(reliability_app, name="reliability")


def _install_example_data(output: Path) -> None:
    """Install the bundled synthetic reliability example at ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    bundled = files("semiyield").joinpath("assets/mosfet_lifetime_smoke.csv")
    with bundled.open("rb") as source, output.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def _quickstart_python_error(version: tuple[int, int] | None = None) -> str | None:
    """Return a compatibility message when CatBoost is unavailable by Python version."""
    major, minor = version or sys.version_info[:2]
    if (major, minor) >= (3, 14):
        return "quickstart requires Python 3.10–3.13 because CatBoost has no supported build here"
    return None


@app.command()
def download(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False):
    """Download the CC BY 4.0 UCI SECOM dataset."""
    typer.echo(f"Dataset ready at {download_secom(data_dir, force=force)}")


@data_app.command("download-demo")
def download_demo(
    output: Path = Path("data/demo/mosfet_lifetime_smoke.csv"),
    url: str | None = typer.Option(None, help="Public URL for a license-cleared derived table"),
    expected_sha256: str | None = None,
):
    """Fetch a cleared derived table, or install the synthetic example dataset."""
    if url:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".part")
        _download_file(url, temporary)
        actual = sha256_file(temporary)
        if expected_sha256 and actual.lower() != expected_sha256.lower():
            temporary.unlink(missing_ok=True)
            raise typer.BadParameter(f"SHA-256 mismatch: got {actual}")
        temporary.replace(output)
        typer.echo(f"Downloaded license-cleared derived data to {output} (sha256={actual})")
        return
    _install_example_data(output)
    typer.echo(
        f"Installed synthetic example data at {output}. "
        "It is not derived from NASA and is intended for local workflow validation."
    )


@app.command()
def quickstart(
    data_dir: Path = DEFAULT_DATA_DIR,
    demo_output: Path = Path("data/demo/mosfet_lifetime_smoke.csv"),
    benchmark_output: Path = Path("reports/reference/yield"),
    reliability_output: Path = Path("reports/reference/reliability"),
    skip_download: bool = False,
    skip_benchmark: bool = False,
    soft_memory_gb: float = typer.Option(32.0, min=0.1),
    memory_limit_gb: float = typer.Option(48.0, min=0.2),
):
    """Run the local SECOM and reliability evaluation workflow."""
    compatibility_error = _quickstart_python_error()
    if compatibility_error:
        raise typer.BadParameter(compatibility_error)
    if not skip_download:
        download(data_dir=data_dir)
    _install_example_data(demo_output)
    typer.echo(f"Installed synthetic reliability example at {demo_output}")
    if not skip_benchmark:
        benchmark(
            data_dir=data_dir,
            output_dir=benchmark_output,
            profile="quick",
            models="dummy,logistic,catboost",
            soft_memory_gb=soft_memory_gb,
            memory_limit_gb=memory_limit_gb,
        )
    reliability_report(
        input_csv=demo_output,
        output_dir=reliability_output,
        profile="quick",
    )
    typer.echo("Quick evaluation complete.")
    typer.echo(f"Yield results: {benchmark_output}")
    typer.echo(f"Reliability results: {reliability_output}")
    typer.echo("Start the local interface with: semiyield app")


@app.command()
def validate(data_dir: Path = DEFAULT_DATA_DIR):
    """Validate SECOM and print a data-quality report."""
    dataset = load_secom(data_dir)
    typer.echo(json.dumps(data_quality_report(dataset.features, dataset.target), indent=2))


@app.command()
def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    output: Path = DEFAULT_MODEL_PATH,
    model: str = typer.Option("logistic", help="dummy, logistic, catboost, or tabpfn"),
    no_calibration: bool = False,
):
    """Train and persist a leakage-safe model pipeline."""
    dataset = load_secom(data_dir)
    artifact = train_model(
        dataset.features, dataset.target, model_name=model, calibrate=not no_calibration
    )
    artifact.save(output)
    typer.echo(f"Saved {model} model to {output}")


@app.command()
def benchmark(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = Path("reports/reference/yield"),
    profile: str = typer.Option("quick", help="quick or full"),
    models: str | None = typer.Option(None, help="Comma-separated model names"),
    soft_memory_gb: float = typer.Option(32.0, min=0.1),
    memory_limit_gb: float = typer.Option(48.0, min=0.2),
    worker: bool = typer.Option(False, "--worker", hidden=True),
):
    """Run a memory-guarded benchmark with training-only threshold selection."""
    if profile not in {"quick", "full"}:
        raise typer.BadParameter("profile must be quick or full")
    selected = models or (
        "dummy,logistic,catboost" if profile == "quick" else "dummy,logistic,catboost,tabpfn"
    )
    model_names = tuple(item.strip() for item in selected.split(",") if item.strip())
    if not model_names:
        raise typer.BadParameter("at least one model is required")
    if not worker:
        command = [
            sys.executable,
            "-m",
            "semiyield.cli",
            "benchmark",
            "--data-dir",
            str(data_dir.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--profile",
            profile,
            "--models",
            ",".join(model_names),
            "--soft-memory-gb",
            str(soft_memory_gb),
            "--memory-limit-gb",
            str(memory_limit_gb),
            "--worker",
        ]
        result = run_guarded(
            command,
            output_dir=output_dir,
            models=list(model_names),
            soft_limit_gb=soft_memory_gb,
            hard_limit_gb=memory_limit_gb,
        )
        typer.echo(
            f"Resource status: {result.status}; peak RSS "
            f"{result.peak_rss_bytes / 1024**3:.2f} GB"
        )
        if result.exit_code:
            raise typer.Exit(result.exit_code)
        return
    dataset = load_secom(data_dir)
    config = BenchmarkConfig(
        models=model_names,
        folds=3 if profile == "quick" else 5,
        repeats=1 if profile == "quick" else 3,
    )
    result = run_benchmark(dataset, config=config, output_dir=output_dir)
    typer.echo(result["summary"].to_string(index=False))
    typer.echo(f"Experiment artifacts: {output_dir}")


@app.command()
def evaluate(
    model_path: Path = DEFAULT_MODEL_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    protocol: str = "temporal",
    output: Path | None = None,
):
    """Evaluate a saved model; benchmark is preferred for published results."""
    dataset = load_secom(data_dir)
    artifact = ModelArtifact.load(model_path)
    metrics = evaluate_model(
        artifact, dataset.features, dataset.target, protocol=protocol, timestamps=dataset.timestamps
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(output, index=False)
    typer.echo(metrics.to_string(index=False))


@app.command()
def predict(model_path: Path, input_csv: Path, output: Path = Path("predictions.csv")):
    """Predict failure risk for a schema-compatible CSV."""
    result = predict_risk(ModelArtifact.load(model_path), pd.read_csv(input_csv))
    result.to_csv(output, index=False)
    typer.echo(f"Wrote {len(result)} predictions to {output}")


@app.command()
def explain(
    model_path: Path,
    input_csv: Path,
    reference_csv: Path,
    output: Path = Path("explanation.csv"),
    top_k: int = 10,
):
    """Rank candidate anomalous variables for the first input row."""
    result = explain_prediction(
        ModelArtifact.load(model_path),
        pd.read_csv(input_csv),
        pd.read_csv(reference_csv),
        top_k=top_k,
    )
    result.to_csv(output, index=False)
    typer.echo(f"Wrote candidate-variable explanation to {output}")


@app.command()
def drift(
    reference_csv: Path,
    current_csv: Path,
    model_path: Path | None = None,
    output: Path = Path("drift.csv"),
):
    """Compare reference and current feature distributions."""
    artifact = ModelArtifact.load(model_path) if model_path else None
    result = detect_drift(pd.read_csv(reference_csv), pd.read_csv(current_csv), artifact=artifact)
    result.to_csv(output, index=False)
    typer.echo(f"Wrote drift report to {output}")


@nasa_app.command("verify")
def nasa_verify(archive: Path, expected_sha256: str | None = None):
    """Verify the large external archive and print its inventory."""
    typer.echo(json.dumps(verify_archive(archive, expected_sha256), indent=2))


@nasa_app.command("download")
def nasa_download(
    output: Path = Path("data/external/nasa/mosfet_thermal_overstress.zip"),
    force: bool = False,
):
    """Download and hash the official NASA archive into the ignored local cache."""
    typer.echo(json.dumps(download_archive(output, force=force), indent=2))


@nasa_app.command("notices")
def nasa_notices(archive: Path):
    """Inspect embedded license/readme notices without inferring permission."""
    typer.echo(json.dumps(inspect_archive_notices(archive), indent=2))


@nasa_app.command("inspect")
def nasa_inspect(archive: Path):
    """Inspect archive types before defining a MATLAB-to-normalized mapping."""
    typer.echo(json.dumps(verify_archive(archive), indent=2))


@nasa_app.command("inspect-mat")
def nasa_inspect_mat(input_mat: Path):
    """List top-level variables in one extracted MATLAB experiment file."""
    typer.echo(json.dumps(inspect_matlab_file(input_mat), indent=2))


@nasa_app.command("convert-matlab")
def nasa_convert_matlab(
    source_dir: Path,
    mapping_json: Path,
    output_dir: Path = Path("data/interim/nasa_mosfet_normalized"),
):
    """Convert explicitly mapped MATLAB slow channels to normalized CSV files."""
    mapping = json.loads(mapping_json.read_text(encoding="utf-8"))
    manifest = convert_matlab_directory(source_dir, output_dir=output_dir, mapping=mapping)
    typer.echo(f"Converted {len(manifest['files'])} MATLAB files to {output_dir}")


@nasa_app.command("convert-official")
def nasa_convert_official(
    archive: Path,
    output_dir: Path = Path("data/interim/nasa_mosfet_normalized"),
):
    """Stream the official inner ZIP into compact on-state transient tables."""
    manifest = convert_nasa_matlab_archive(archive, output_dir=output_dir)
    typer.echo(
        f"Converted {len(manifest['files'])} runs from {len(manifest['devices'])} devices; "
        f"skipped {len(manifest['skipped'])} runs"
    )


@nasa_app.command("prepare")
def nasa_prepare(
    source: Annotated[Path, typer.Option("--archive", "--source")],
    output_dir: Path = Path("data/processed/nasa_mosfet"),
    window_size: int = 1000,
    device_column: str = "device_id",
    time_column: str = "time_s",
    temperature_column: str = "temperature_c",
    voltage_column: str = "vds_v",
    current_column: str = "id_a",
):
    """Aggregate all normalized slow-measurement tables into compact device features."""
    mapping = SignalMapping(
        device=device_column,
        time=time_column,
        temperature=temperature_column,
        voltage=voltage_column,
        current=current_column,
    )
    result = prepare_nasa_features(
        source, output_dir=output_dir, mapping=mapping, window_size=window_size
    )
    typer.echo(
        f"Prepared {len(result['features'])} windows from "
        f"{result['features']['device_id'].nunique()} devices at {output_dir}"
    )


@reliability_app.command("validate")
def reliability_validate(input_csv: Path):
    """Validate lifetime, censoring, and optional temperature columns."""
    data = validate_lifetime_data(pd.read_csv(input_csv))
    typer.echo(f"Valid lifetime table: {len(data)} units, {data.event_observed.sum()} failures")


@reliability_app.command("report")
def reliability_report(
    input_csv: Path = Path("data/demo/mosfet_lifetime_smoke.csv"),
    output_dir: Path = Path("reports/reference/reliability"),
    profile: str = "quick",
    use_temperature_c: float = 55.0,
):
    """Generate Weibull, Arrhenius, and optional survival-model reports."""
    if profile not in {"quick", "full"}:
        raise typer.BadParameter("profile must be quick or full")
    if not input_csv.exists():
        raise typer.BadParameter("Input is missing; run `semiyield data download-demo` first")
    frame = pd.read_csv(input_csv)
    report = write_reliability_report(
        frame,
        output_dir=output_dir,
        use_temperature_c=use_temperature_c,
        data_sha256=sha256_file(input_csv),
    )
    if "dataset_role" in frame and frame["dataset_role"].eq("ci_smoke_only").any():
        typer.echo("NOTE: synthetic example data is not a NASA experimental result")
    typer.echo(json.dumps(report, indent=2))


@app.command("report")
def report(
    output_dir: Path = Path("reports/verified"),
    nasa_provenance: Path | None = None,
):
    """Build a hashed manifest for completed experiment reports."""
    report = build_experiment_manifest(output_dir, nasa_provenance=nasa_provenance)
    typer.echo(json.dumps(report, indent=2))


@app.command("evidence", hidden=True)
def evidence(
    output_dir: Path = Path("reports/verified"),
    nasa_provenance: Path | None = None,
):
    """Compatibility alias for ``semiyield report``."""
    report(output_dir=output_dir, nasa_provenance=nasa_provenance)


@app.command(name="app")
def launch_app():
    """Launch the local bilingual Streamlit application."""
    app_file = Path(__file__).with_name("streamlit_app.py")
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_file)]))


if __name__ == "__main__":
    app()
