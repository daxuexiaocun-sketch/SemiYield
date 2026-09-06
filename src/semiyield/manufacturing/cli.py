from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import typer

from semiyield.common.reporting import data_quality_report
from semiyield.common.resources import resolve_memory_limits, run_guarded
from semiyield.constants import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH

from .data import download_secom, load_secom
from .drift import detect_drift
from .evaluate import BenchmarkConfig, run_benchmark
from .explain import explain_prediction
from .metrics import evaluate_model
from .modeling import ModelArtifact, train_model
from .predict import predict_risk

app = typer.Typer()


@app.command("download")
def download(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False):
    """Download the CC BY 4.0 UCI SECOM dataset."""
    typer.echo(f"Dataset ready at {download_secom(data_dir, force=force)}")


@app.command("validate")
def validate(data_dir: Path = DEFAULT_DATA_DIR):
    """Validate SECOM and print a data-quality report."""
    dataset = load_secom(data_dir)
    typer.echo(json.dumps(data_quality_report(dataset.features, dataset.target), indent=2))


@app.command("train")
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


@app.command("benchmark")
def benchmark(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = Path("reports/reference/yield"),
    profile: str = typer.Option("quick", help="quick or full"),
    models: str | None = typer.Option(None, help="Comma-separated model names"),
    soft_memory_gb: float | None = typer.Option(
        None, min=0.1, help="RSS warning limit in GiB; defaults to 70% of physical memory"
    ),
    memory_limit_gb: float | None = typer.Option(
        None, min=0.2, help="RSS termination limit in GiB; defaults to 80% of physical memory"
    ),
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
        soft_memory_gb, memory_limit_gb = resolve_memory_limits(soft_memory_gb, memory_limit_gb)
        command = [
            sys.executable,
            "-m",
            "semiyield.cli",
            "yield",
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
            "Resource status: "
            f"{result.status}; peak RSS {result.peak_rss_bytes / 1024**3:.2f} GB; "
            f"guard {soft_memory_gb:.2f}/{memory_limit_gb:.2f} GiB"
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


@app.command("evaluate")
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


@app.command("predict")
def predict(model_path: Path, input_csv: Path, output: Path = Path("predictions.csv")):
    """Predict failure risk for a schema-compatible CSV."""
    result = predict_risk(ModelArtifact.load(model_path), pd.read_csv(input_csv))
    result.to_csv(output, index=False)
    typer.echo(f"Wrote {len(result)} predictions to {output}")


@app.command("explain")
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


@app.command("drift")
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
