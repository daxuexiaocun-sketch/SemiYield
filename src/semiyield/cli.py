"""Root registration for the three public SemiYield business commands."""

import subprocess
import sys
from pathlib import Path

import typer

from semiyield.manufacturing import cli as manufacturing_cli
from semiyield.packaging.cli import app as packaging_app
from semiyield.reliability import cli as reliability_cli

app = typer.Typer(
    help="Manufacturing yield, packaging process, and device lifetime analysis."
)
app.add_typer(manufacturing_cli.app, name="yield")
app.add_typer(packaging_app, name="packaging")
app.add_typer(reliability_cli.app, name="reliability")


@app.command("app")
def launch_app():
    """Launch the bilingual local interface."""
    entry = Path(__file__).with_name("streamlit_app.py")
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(entry)]))


if __name__ == "__main__":
    app()
