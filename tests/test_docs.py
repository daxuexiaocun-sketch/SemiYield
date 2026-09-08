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
        "uv sync --locked --extra charts --extra app --extra dev",
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


def test_three_route_reference_assets_are_linked_and_aggregate_only():
    for readme in (ROOT / "README.md", ROOT / "README.zh-CN.md"):
        content = readme.read_text(encoding="utf-8")
        assert "docs/assets/results-overview.svg" in content
    for asset in (
        "docs/assets/results-overview.svg",
        "reports/verified/yield/benchmark_pr_auc.svg",
        "reports/verified/packaging/benchmark_summary.svg",
        "reports/verified/packaging/benchmark_metrics_table.svg",
        "reports/verified/nasa/degradation_trends.svg",
        "reports/verified/reliability_0045/rsf_individual_predicted_survival.svg",
    ):
        assert (ROOT / asset).is_file()
    report = (ROOT / "reports" / "verified" / "README.md").read_text(encoding="utf-8")
    assert "benchmark_metrics_table.svg" not in report
    assert "rsf_performance.svg" not in report
    assert "rsf_individual_predicted_survival.svg" in report
    for text in (
        "NASA MOSFET reliability combines observed high-temperature data",
        "reported conservatively as",
        "baseline-feature RSF achieved Harrell C-index **0.905**",
        "bootstrap 95% CI",
        "Uno C-index **0.862**",
        "integrated Brier score **0.149**",
        "MAE **5,810 s**",
        "protected 9-device test set",
    ):
        assert text in report


def test_no_synthetic_demo_route_remains():
    for path in (
        ROOT / "src" / "semiyield" / "demo",
        ROOT / "src" / "semiyield" / "simulation",
        ROOT / "tests" / "test_demo.py",
        ROOT / "docs" / "DEMO.md",
        ROOT / "data" / "demo" / "three_stage",
        ROOT / "reports" / "demo" / "three_stage",
        ROOT / "reports" / "verified" / "demo",
    ):
        assert not path.exists()
    assert (ROOT / "data" / "demo" / "mosfet_lifetime_smoke.csv").is_file()
