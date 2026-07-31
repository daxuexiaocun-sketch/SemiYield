# ruff: noqa: E501
"""Generate a deterministic GitHub star-history dataset and SVG."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

API_ROOT = "https://api.github.com"
STAR_ACCEPT = "application/vnd.github.star+json"


def _request_json(url: str, token: str | None, accept: str = "application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "semiyield-star-history"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=30) as response:  # noqa: S310
        return json.load(response)


def fetch_star_history(repository: str, token: str | None = None) -> dict[str, object]:
    """Fetch repository creation time and every timestamped stargazer."""
    metadata = _request_json(f"{API_ROOT}/repos/{repository}", token)
    created_at = str(metadata["created_at"])
    starred_at: list[str] = []
    page = 1
    while True:
        rows = _request_json(
            f"{API_ROOT}/repos/{repository}/stargazers?per_page=100&page={page}",
            token,
            STAR_ACCEPT,
        )
        if not isinstance(rows, list):
            raise ValueError("GitHub stargazer response was not a list")
        starred_at.extend(str(row["starred_at"]) for row in rows)
        if len(rows) < 100:
            break
        page += 1
        if page > 1000:
            raise RuntimeError("GitHub pagination exceeded the 100,000-star safety limit")
    return build_history(repository, created_at, starred_at)


def build_history(
    repository: str, created_at: str, starred_at: list[str]
) -> dict[str, object]:
    """Aggregate timestamped stars into a daily cumulative series."""
    created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date().isoformat()
    daily = Counter(
        datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        for value in starred_at
    )
    dates = sorted(set(daily) | {created_date})
    total = 0
    points = []
    for date in dates:
        total += daily[date]
        points.append({"date": date, "total": total})
    return {
        "schema_version": "semiyield-star-history-v1",
        "repository": repository,
        "source": "github-api",
        "points": points,
    }


def render_svg(history: dict[str, object]) -> str:  # noqa: C901
    """Render a dependency-free SVG that remains legible in light and dark themes."""
    points = list(history["points"])
    width, height = 900, 360
    left, right, top, bottom = 74, 28, 52, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = [int(point["total"]) for point in points]
    maximum = max(max(totals, default=0), 1)
    denominator = max(len(points) - 1, 1)
    coordinates = [
        (
            left + index * plot_w / denominator,
            top + plot_h - total * plot_h / maximum,
        )
        for index, total in enumerate(totals)
    ]
    if len(coordinates) == 1:
        coordinates.append((left + plot_w, coordinates[0][1]))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
    first_date = escape(str(points[0]["date"]))
    last_date = escape(str(points[-1]["date"]))
    repository = escape(str(history["repository"]))
    current = totals[-1] if totals else 0
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">GitHub stars over time for {repository}</title>
<desc id="desc">Cumulative GitHub stars from {first_date} to {last_date}; current total {current}.</desc>
<style>
  .bg {{ fill: #ffffff; }} .fg {{ fill: #172033; }} .muted {{ fill: #667085; }}
  .grid {{ stroke: #d8dee9; }} .line {{ stroke: #2563eb; }} .area {{ fill: #dbeafe; }}
  @media (prefers-color-scheme: dark) {{
    .bg {{ fill: #0d1117; }} .fg {{ fill: #e6edf3; }} .muted {{ fill: #9da7b3; }}
    .grid {{ stroke: #30363d; }} .line {{ stroke: #58a6ff; }} .area {{ fill: #16345f; }}
  }}
</style>
<rect class="bg" width="100%" height="100%" rx="12"/>
<text class="fg" x="{left}" y="30" font-family="system-ui,sans-serif" font-size="18" font-weight="600">GitHub stars over time</text>
<text class="muted" x="{width-right}" y="30" text-anchor="end" font-family="system-ui,sans-serif" font-size="14">{current} stars</text>
<line class="grid" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>
<line class="grid" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>
<line class="grid" x1="{left}" y1="{top}" x2="{left+plot_w}" y2="{top}" stroke-dasharray="4 6"/>
<text class="muted" x="{left-12}" y="{top+5}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12">{maximum}</text>
<text class="muted" x="{left-12}" y="{top+plot_h+5}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12">0</text>
<polygon class="area" points="{left},{top+plot_h} {polyline} {left+plot_w},{top+plot_h}" opacity="0.7"/>
<polyline class="line" points="{polyline}" fill="none" stroke-width="3" stroke-linejoin="round"/>
<text class="muted" x="{left}" y="{height-24}" font-family="system-ui,sans-serif" font-size="12">{first_date}</text>
<text class="muted" x="{left+plot_w}" y="{height-24}" text-anchor="end" font-family="system-ui,sans-serif" font-size="12">{last_date}</text>
</svg>
'''


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def update(repository: str, token: str | None, data_path: Path, svg_path: Path) -> None:
    """Fetch first, then atomically replace both outputs after a successful response."""
    history = fetch_star_history(repository, token)
    _atomic_write(data_path, json.dumps(history, indent=2) + "\n")
    _atomic_write(svg_path, render_svg(history))


def main() -> None:
    """Update the repository's star-history assets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--data", type=Path, default=Path("docs/assets/star-history.json"))
    parser.add_argument("--svg", type=Path, default=Path("docs/assets/star-history.svg"))
    args = parser.parse_args()
    if not args.repository:
        parser.error("--repository or GITHUB_REPOSITORY is required")
    update(args.repository, os.getenv("GITHUB_TOKEN"), args.data, args.svg)
