from __future__ import annotations

import json
import shutil
import sys
from importlib.resources import files
from pathlib import Path

import typer

from semiyield.common.data import (
    _download_file,
    sha256_file,
)
from semiyield.constants import DEFAULT_DATA_DIR
from semiyield.evidence import build_experiment_manifest
from semiyield.reliability.cli import reliability_report
from semiyield.yield_risk.cli import benchmark, download

app = typer.Typer()
data_app = typer.Typer()


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


@app.command("quickstart")
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
