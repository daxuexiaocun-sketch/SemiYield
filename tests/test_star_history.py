import io
import json
from urllib.error import HTTPError

import pytest

from semiyield import star_history as update_star_history


def test_zero_stars_produces_creation_point_and_accessible_svg():
    history = update_star_history.build_history("owner/repo", "2026-01-02T00:00:00Z", [])
    assert history["points"] == [{"date": "2026-01-02", "total": 0}]
    svg = update_star_history.render_svg(history)
    assert "<title" in svg
    assert "current total 0" in svg


def test_stars_are_aggregated_by_day():
    history = update_star_history.build_history(
        "owner/repo",
        "2026-01-01T00:00:00Z",
        ["2026-01-03T01:00:00Z", "2026-01-03T22:00:00Z", "2026-01-05T00:00:00Z"],
    )
    assert history["points"][-2:] == [
        {"date": "2026-01-03", "total": 2},
        {"date": "2026-01-05", "total": 3},
    ]


def test_fetch_follows_full_pages(monkeypatch):
    calls = []

    class Response(io.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/repos/owner/repo"):
            payload = {"created_at": "2026-01-01T00:00:00Z"}
        elif request.full_url.endswith("page=1"):
            payload = [{"starred_at": "2026-01-02T00:00:00Z"}] * 100
        else:
            payload = [{"starred_at": "2026-01-03T00:00:00Z"}]
        return Response(json.dumps(payload))

    monkeypatch.setattr(update_star_history, "urlopen", fake_urlopen)
    history = update_star_history.fetch_star_history("owner/repo")
    assert history["points"][-1]["total"] == 101
    assert any("page=2" in call for call in calls)
    assert len(calls) == 3


def test_api_failure_preserves_existing_outputs(tmp_path, monkeypatch):
    data_path = tmp_path / "history.json"
    svg_path = tmp_path / "history.svg"
    data_path.write_text("old data")
    svg_path.write_text("old svg")

    def fail(*_args, **_kwargs):
        raise HTTPError("https://api.github.com", 403, "rate limited", {}, None)

    monkeypatch.setattr(update_star_history, "fetch_star_history", fail)
    with pytest.raises(HTTPError):
        update_star_history.update("owner/repo", None, data_path, svg_path)
    assert data_path.read_text() == "old data"
    assert svg_path.read_text() == "old svg"


def test_irregular_dates_use_elapsed_days():
    import re

    history = update_star_history.build_history(
        "owner/repo",
        "2026-01-01T00:00:00Z",
        ["2026-01-02T00:00:00Z", "2026-01-11T00:00:00Z"],
    )
    svg = update_star_history.render_svg(history)
    points = re.search(r'<polyline class="line" points="([^"]+)"', svg).group(1)
    xs = [float(point.split(",")[0]) for point in points.split()]
    assert (xs[1] - xs[0]) / (xs[2] - xs[0]) == pytest.approx(0.1)


def test_pending_history_does_not_claim_zero_stars():
    svg = update_star_history.render_svg({"repository": "owner/repo", "points": []})
    assert "awaiting" in svg
    assert "current total 0" not in svg


def test_render_failure_preserves_both_outputs(tmp_path, monkeypatch):
    data, svg = tmp_path / "history.json", tmp_path / "history.svg"
    data.write_text("old data")
    svg.write_text("old svg")
    monkeypatch.setattr(
        update_star_history,
        "fetch_star_history",
        lambda *_: {
            "repository": "owner/repo",
            "points": [{"date": "invalid", "total": 1}],
        },
    )
    with pytest.raises(ValueError):
        update_star_history.update("owner/repo", None, data, svg)
    assert data.read_text() == "old data"
    assert svg.read_text() == "old svg"


def test_cli_permission_failure_reports_status_without_token(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["star-history", "--repository", "owner/repo"])
    monkeypatch.setenv("GITHUB_TOKEN", "do-not-print-this")

    def forbidden(*args):
        raise HTTPError("https://api.github.com", 403, "Forbidden", {}, None)

    monkeypatch.setattr(update_star_history, "update", forbidden)
    with pytest.raises(SystemExit) as exc:
        update_star_history.main()
    assert exc.value.code == 1
    output = capsys.readouterr().err
    assert "HTTP 403" in output and "retained" in output
    assert "do-not-print-this" not in output


def test_workflow_detects_new_assets_and_unchanged_assets(tmp_path):
    import subprocess
    from pathlib import Path

    workflow = (Path(__file__).parents[1] / ".github/workflows/star-history.yml").read_text()
    # Execute the actual staging/change-detection portion without a remote mutation.
    detection = workflow.split("        run: |\n", 1)[1].split("          git config", 1)[0]
    import textwrap

    detection = textwrap.dedent(detection) + '\necho "changed"\n'
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    assets = tmp_path / "docs/assets"
    assets.mkdir(parents=True)
    (assets / "star-history.json").write_text("{}")
    (assets / "star-history.svg").write_text("<svg/>")
    result = subprocess.run(
        ["bash", "-c", detection], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert "changed" in result.stdout
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    result = subprocess.run(
        ["bash", "-c", detection], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "Star history is unchanged."
