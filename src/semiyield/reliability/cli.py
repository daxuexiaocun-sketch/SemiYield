from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from semiyield.common.artifacts import sha256_file
from semiyield.reliability.data import validate_lifetime_data
from semiyield.reliability.nasa import (
    SignalMapping,
    convert_matlab_directory,
    convert_nasa_matlab_archive,
    download_archive,
    inspect_archive_notices,
    inspect_matlab_file,
    prepare_nasa_features,
    verify_archive,
)
from semiyield.reliability.reporting import write_reliability_report

app = typer.Typer()
nasa_app = typer.Typer(help="Prepare NASA MOSFET data.")
example_app = typer.Typer(help="Install the bundled synthetic lifetime example.")


@example_app.command("install")
def example_install(output: Path = Path("data/demo/mosfet_lifetime_smoke.csv")):
    """Install the bundled synthetic lifetime table for local smoke testing."""
    output.parent.mkdir(parents=True, exist_ok=True)
    bundled = files("semiyield").joinpath("assets/mosfet_lifetime_smoke.csv")
    with bundled.open("rb") as source, output.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    typer.echo(
        f"Installed synthetic example data at {output}; it is not a NASA experimental result."
    )


app.add_typer(nasa_app, name="nasa")
app.add_typer(example_app, name="example")


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


@app.command("validate")
def reliability_validate(input_csv: Path):
    """Validate lifetime, censoring, and optional temperature columns."""
    data = validate_lifetime_data(pd.read_csv(input_csv))
    typer.echo(f"Valid lifetime table: {len(data)} units, {data.event_observed.sum()} failures")


@app.command("report")
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
        raise typer.BadParameter(
            "Input is missing; run `semiyield reliability example install` first"
        )
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
