"""Root registration for the four public SemiYield business commands."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path

import typer

from semiyield.demo.cli import app as demo_app
from semiyield.packaging.cli import app as packaging_app
from semiyield.reliability import cli as reliability_cli

yield_cli = import_module("semiyield.yield.cli")

app = typer.Typer(help="Manufacturing yield, packaging process, device lifetime, and demo analysis.")
app.add_typer(yield_cli.app, name="yield")
app.add_typer(packaging_app, name="packaging")
app.add_typer(reliability_cli.app, name="reliability")
app.add_typer(demo_app, name="demo")


@app.command("app")
def launch_app():
    """Launch the bilingual local interface."""
    entry = Path(__file__).with_name("streamlit_app.py")
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(entry)]))


if __name__ == "__main__":
    app()
