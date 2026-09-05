from typer.testing import CliRunner

import semiyield.workflows as cli
from semiyield.cli import app

runner = CliRunner()


def test_cli_help_lists_research_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "benchmark" in result.stdout
    assert "quickstart" in result.stdout
    assert "reliability" in result.stdout
    assert "nasa" in result.stdout


def test_demo_and_reliability_cli(tmp_path):
    demo = tmp_path / "smoke.csv"
    report = tmp_path / "report"
    downloaded = runner.invoke(app, ["data", "download-demo", "--output", str(demo)])
    assert downloaded.exit_code == 0
    assert "synthetic example" in downloaded.stdout
    generated = runner.invoke(
        app,
        [
            "reliability",
            "report",
            "--input-csv",
            str(demo),
            "--output-dir",
            str(report),
        ],
    )
    assert generated.exit_code == 0
    assert "synthetic example data" in generated.stdout
    assert (report / "reliability_report.json").exists()


def test_quickstart_runs_shared_workflow(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "_quickstart_python_error", lambda: None)
    monkeypatch.setattr(cli, "download", lambda **kwargs: calls.append(("download", kwargs)))
    monkeypatch.setattr(
        cli, "_install_example_data", lambda output: calls.append(("example", {"output": output}))
    )
    monkeypatch.setattr(cli, "benchmark", lambda **kwargs: calls.append(("benchmark", kwargs)))
    monkeypatch.setattr(
        cli, "reliability_report", lambda **kwargs: calls.append(("reliability", kwargs))
    )

    result = runner.invoke(
        app,
        [
            "quickstart",
            "--data-dir",
            str(tmp_path / "secom"),
            "--demo-output",
            str(tmp_path / "demo.csv"),
            "--benchmark-output",
            str(tmp_path / "yield"),
            "--reliability-output",
            str(tmp_path / "reliability"),
        ],
    )
    assert result.exit_code == 0
    assert [name for name, _ in calls] == ["download", "example", "benchmark", "reliability"]
    assert "semiyield app" in result.stdout


def test_quickstart_python_compatibility_message():
    assert "Python 3.10–3.13" in cli._quickstart_python_error((3, 14))
    assert cli._quickstart_python_error((3, 13)) is None
