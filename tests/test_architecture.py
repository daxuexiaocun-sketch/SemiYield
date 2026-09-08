"""Prevent accidental cross-business imports."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "semiyield"
BUSINESS = {"manufacturing", "packaging", "reliability"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_business_packages_do_not_import_each_other():
    for owner in BUSINESS:
        for path in (ROOT / owner).glob("*.py"):
            imports = _imports(path)
            for dependency in BUSINESS - {owner}:
                assert not any(name.startswith(f"semiyield.{dependency}") for name in imports), path
