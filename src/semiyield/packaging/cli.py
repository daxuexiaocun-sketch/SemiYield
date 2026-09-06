"""Packaging process commands."""

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from semiyield.common.artifacts import sha256_file
from semiyield.common.catboost import load_params, write_params
from semiyield.common.cli import friendly_errors
from semiyield.common.reporting import data_quality_report
from semiyield.packaging.data import FEATURES, local_data_status, validate_data
from semiyield.packaging.evaluate import run_benchmark, train_holdout
from semiyield.packaging.modeling import PackagingArtifact, tune_catboost

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
    catboost_params: Annotated[Path | None, typer.Option("--catboost-params")] = None,
):
    """Train on 80% of rows; derive the proxy threshold only from training data."""
    params = params_hash = None
    if catboost_params:
        if model != "catboost":
            raise typer.BadParameter("--catboost-params requires --model catboost")
        params, params_hash = load_params(
            catboost_params, task="packaging", data_sha256=sha256_file(input_csv)
        )
    artifact = train_holdout(
        input_csv,
        output,
        model=model,
        seed=seed,
        threshold=threshold,
        quantile=quantile,
        probability_threshold=probability_threshold,
        catboost_params=params,
        catboost_params_sha256=params_hash,
    )
    typer.echo(json.dumps(artifact.metadata, indent=2))


@app.command("tune")
@friendly_errors
def tune_command(
    input_csv: Annotated[Path, typer.Option("--input-csv")] = ...,
    output: Path = Path("artifacts/packaging/catboost_params.json"),
    seed: int = 42,
    threshold: float | None = None,
    quantile: float | None = None,
):
    """Run a reproducible CatBoost PR-AUC search and save its parameter artifact."""
    artifact = tune_catboost(
        pd.read_csv(input_csv),
        data_sha256=sha256_file(input_csv),
        seed=seed,
        threshold=threshold,
        quantile=quantile,
    )
    write_params(artifact, output)
    typer.echo(f"Saved CatBoost parameters to {output}")


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
