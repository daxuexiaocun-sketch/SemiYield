"""Root command registration and compatibility exports."""

import subprocess
import sys
from pathlib import Path

import typer

from semiyield import workflows
from semiyield.demo.cli import app as demo_app
from semiyield.packaging.cli import app as packaging_app
from semiyield.reliability import cli as reliability_cli
from semiyield.yield_risk import cli as yield_cli

app = workflows.app
app.info.help = "Manufacturing yield, packaging proxy failure, and device lifetime analysis."
app.add_typer(yield_cli.app, name="yield")
app.add_typer(packaging_app, name="packaging")
app.add_typer(reliability_cli.app, name="reliability")
app.add_typer(reliability_cli.nasa_app, name="nasa")
app.add_typer(workflows.data_app, name="data")
app.add_typer(demo_app, name="demo")
# Legacy top-level SECOM commands and Python function imports.
for command in yield_cli.app.registered_commands:
    app.command(command.name)(command.callback)


def __getattr__(name):
    for module in (yield_cli, reliability_cli, workflows):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


@app.command("app")
def launch_app():
    """Launch the bilingual local interface."""
    entry = Path(__file__).with_name("streamlit_app.py")
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(entry)]))


if __name__ == "__main__":
    app()
