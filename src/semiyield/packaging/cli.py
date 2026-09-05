"""Packaging process commands."""

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from semiyield.common.cli import friendly_errors
from semiyield.common.reporting import data_quality_report
from semiyield.packaging.data import FEATURES, local_data_status, validate_data
from semiyield.packaging.evaluate import run_benchmark, train_holdout
from semiyield.packaging.modeling import PackagingArtifact

app = typer.Typer(help="Low-throughput proxy failure analysis (not physical device failure).")


@app.command()
@friendly_errors
def validate(input_csv: Annotated[Path, typer.Option("--input-csv")] = ...):
    """Validate raw mixed categorical and numerical process data."""
    frame = validate_data(pd.read_csv(input_csv))
    typer.echo(json.dumps(data_quality_report(frame[FEATURES]), indent=2))
    typer.echo("Y is throughput; its source unit is unspecified. Labels are proxy failures.")


@app.command()
@friendly_errors
def train(
    input_csv: Annotated[Path, typer.Option("--input-csv")] = ...,
    output: Path = Path("artifacts/packaging/model.joblib"),
    model: str = "logistic",
    threshold: float | None = None,
    quantile: float | None = None,
    seed: int = 42,
    probability_threshold: float = 0.5,
):
    """Train on 80% of rows; derive the proxy threshold only from training data."""
    artifact = train_holdout(
        input_csv,
        output,
        model=model,
        seed=seed,
        threshold=threshold,
        quantile=quantile,
        probability_threshold=probability_threshold,
    )
    typer.echo(json.dumps(artifact.metadata, indent=2))


@app.command()
@friendly_errors
def benchmark(
    input_csv: Annotated[Path, typer.Option("--input-csv")] = ...,
    output_dir: Path = Path("reports/packaging"),
    models: str = "dummy,logistic",
    threshold: float | None = None,
    quantile: float | None = None,
    seed: int = 42,
):
    """Compare models on shared three-fold splits with training-only proxy thresholds."""
    result = run_benchmark(
        input_csv,
        output_dir,
        models=tuple(m.strip() for m in models.split(",")),
        threshold=threshold,
        quantile=quantile,
        seed=seed,
    )
    typer.echo(result.to_string())


@app.command()
@friendly_errors
def predict(model_path: Path, input_csv: Path, output: Path = Path("packaging_predictions.csv")):
    """Predict proxy failure probabilities without requiring Y."""
    result = PackagingArtifact.load(model_path).predict(pd.read_csv(input_csv))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    typer.echo(f"Wrote {len(result)} packaging predictions to {output}")


@app.command("data-status")
def data_status(path: Path | None = None):
    """Show whether the optional local packaging source is available."""
    typer.echo(json.dumps(local_data_status(path) if path else local_data_status(), indent=2))
