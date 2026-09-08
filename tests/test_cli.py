from typer.testing import CliRunner

from semiyield.cli import app

runner = CliRunner()


def test_cli_help_lists_business_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "yield" in result.stdout
    assert "packaging" in result.stdout
    assert "reliability" in result.stdout
    assert "demo" not in result.stdout
    assert "quickstart" not in result.stdout


def test_reliability_smoke_cli(tmp_path):
    smoke = tmp_path / "smoke.csv"
    report = tmp_path / "report"
    downloaded = runner.invoke(app, ["reliability", "example", "install", "--output", str(smoke)])
    assert downloaded.exit_code == 0
    assert "synthetic example" in downloaded.stdout
    generated = runner.invoke(
        app,
        [
            "reliability",
            "report",
            "--input-csv",
            str(smoke),
            "--output-dir",
            str(report),
        ],
    )
    assert generated.exit_code == 0
    assert "synthetic example data" in generated.stdout
    assert (report / "reliability_report.json").exists()


def test_old_top_level_quickstart_is_absent():
    assert runner.invoke(app, ["quickstart"]).exit_code != 0
