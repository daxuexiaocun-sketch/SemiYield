import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from semiyield.cli import app


def test_legacy_modules_are_not_importable():
    importlib.import_module("semiyield.manufacturing")
    for name in (
        "data",
        "benchmark",
        "modeling",
        "preprocessing",
        "workflows",
        "nasa",
        "yield",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"semiyield.{name}")


def test_business_command_help_has_no_legacy_routes():
    for args in [
        ["yield", "train"],
        ["reliability", "nasa", "prepare"],
        ["reliability", "example", "install"],
        ["reliability", "report"],
        ["packaging", "train"],
        ["demo", "run"],
    ]:
        result = CliRunner().invoke(app, [*args, "--help"])
        assert result.exit_code == 0, result.output
    root = CliRunner().invoke(app, ["--help"])
    assert "quickstart" not in root.stdout
    assert "nasa" not in root.stdout


def test_star_script_in_installed_isolated_interpreter():
    script = Path(__file__).parents[1] / "scripts/update_star_history.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--repository" in result.stdout
