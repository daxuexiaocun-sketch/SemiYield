"""Business-neutral validation helpers for paths and output locations."""

from pathlib import Path


def require_distinct_directories(source: str | Path, destination: str | Path) -> tuple[Path, Path]:
    """Resolve two directories and reject equal or nested source/output locations."""
    source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
    if source_path == destination_path or source_path in destination_path.parents or destination_path in source_path.parents:
        raise ValueError("Source and output directories must be separate and non-nested")
    return source_path, destination_path
