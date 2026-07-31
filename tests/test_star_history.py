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
