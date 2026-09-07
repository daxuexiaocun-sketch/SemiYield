import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PUBLIC_TEXT = [
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "CHANGELOG.md",
    *sorted((ROOT / "docs").glob("*.md")),
    ROOT / "reports" / "verified" / "README.md",
]


@pytest.mark.parametrize("readme", [ROOT / "README.md", ROOT / "README.zh-CN.md"])
def test_readme_local_links_resolve(readme):
    targets = re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", readme.read_text(encoding="utf-8"))
    local_targets = [target.split("#", 1)[0] for target in targets if "://" not in target]
    missing = [target for target in local_targets if not (readme.parent / target).exists()]
    assert not missing


def test_readmes_have_matching_content_structure():
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert english.count("\n## ") == chinese.count("\n## ") == 6
    for command in (
        "uv sync --locked --extra demo --extra dev",
        "semiyield demo quickstart",
        "semiyield packaging benchmark",
        "--input-csv",
        "semiyield reliability example install",
    ):
        assert command in english
        assert command in chinese


def test_public_docs_do_not_contain_personal_job_search_language():
    english = ["recruit" + "ment", "inter" + "view"]
    chinese = ["\u62db\u8058", "\u9762\u8bd5", "\u6c42\u804c", "\u7b80\u5386"]
    blocked = re.compile("|".join(english + chinese), re.IGNORECASE)
    findings = []
    for path in PUBLIC_TEXT:
        if blocked.search(path.read_text(encoding="utf-8")):
            findings.append(str(path.relative_to(ROOT)))
    assert not findings


def test_documentation_svgs_are_portable():
    for path in (ROOT / "docs" / "assets").glob("*.svg"):
        content = path.read_text(encoding="utf-8")
        assert "/Users/" not in content
        assert "file://" not in content
        assert "<title" in content
        assert "<desc" in content


def test_four_route_reference_assets_are_linked_and_aggregate_only():
    for readme in (ROOT / "README.md", ROOT / "README.zh-CN.md"):
        content = readme.read_text(encoding="utf-8")
        assert "docs/assets/results-overview.svg" in content
    for asset in (
        "docs/assets/results-overview.svg",
        "reports/verified/yield/benchmark_pr_auc.svg",
        "reports/verified/packaging/benchmark_summary.svg",
        "reports/verified/packaging/benchmark_metrics_table.svg",
        "reports/verified/nasa/degradation_trends.svg",
    ):
        assert (ROOT / asset).is_file()
    public_demo = ROOT / "reports" / "verified" / "demo" / "README.md"
    assert "example_trace.csv" not in public_demo.read_text(encoding="utf-8")
