"""CLI entry points for offline synthetic demonstrations."""

from pathlib import Path

import typer

from semiyield.common.cli import friendly_errors
from semiyield.demo.workflow import run as run_demo
from semiyield.simulation.generator import generate as generate_data

app = typer.Typer(help="Reproducible synthetic three-stage data and static reports.")
DATA = Path("data/demo/three_stage")
REPORTS = Path("reports/demo/three_stage")


@app.command()
@friendly_errors
def generate(
    output_dir: Path = DATA,
    seed: int = 42,
    batches: int = 100,
    units_per_batch: int = 100,
    force: bool = False,
):
    """Generate linked synthetic tables, never downloading real datasets."""
    generate_data(
        output_dir, seed=seed, batches=batches, units_per_batch=units_per_batch, force=force
    )
    typer.echo(f"Synthetic dataset: {output_dir}")


@app.command()
@friendly_errors
def run(data_dir: Path = DATA, output_dir: Path = REPORTS, force: bool = False):
    """Evaluate all three stages and produce static reports without network access."""
    typer.echo(f"Synthetic report: {run_demo(data_dir, output_dir, force=force)}")


@app.command()
@friendly_errors
def quickstart(
    data_dir: Path = DATA,
    output_dir: Path = REPORTS,
    seed: int = 42,
    batches: int = 100,
    units_per_batch: int = 100,
    force: bool = False,
):
    """Generate and run the complete offline three-stage demonstration."""
    source, destination = data_dir.resolve(), output_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Data and report directories must be separate and non-nested")
    for path in (data_dir, output_dir):
        if path.exists() and (not path.is_dir() or any(path.iterdir())) and not force:
            raise ValueError(f"Output exists: {path}. Choose another directory or pass --force")
    generate_data(
        data_dir, seed=seed, batches=batches, units_per_batch=units_per_batch, force=force
    )
    typer.echo(f"Synthetic report: {run_demo(data_dir, output_dir, force=force)}")
