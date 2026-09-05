"""Business-neutral validation helpers for paths and output locations."""

import json
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


def require_distinct_directories(source: str | Path, destination: str | Path) -> tuple[Path, Path]:
    """Resolve two directories and reject equal or nested source/output locations."""
    source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
    if (
        source_path == destination_path
        or source_path in destination_path.parents
        or destination_path in source_path.parents
    ):
        raise ValueError("Source and output directories must be separate and non-nested")
    return source_path, destination_path


@contextmanager
def managed_output_dir(path: str | Path, *, force: bool = False):
    """Stage a complete managed output directory and atomically replace it on success."""
    target = Path(path)
    if target.exists() and any(target.iterdir()):
        manifest = target / "manifest.json"
        if not force:
            raise ValueError(f"Output exists: {target}. Choose another directory or pass --force")
        if not manifest.exists():
            raise ValueError(f"Refusing to overwrite unmanaged output directory: {target}")
        try:
            schema = json.loads(manifest.read_text(encoding="utf-8")).get("schema_version", "")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Refusing to overwrite unreadable output manifest: {target}") from exc
        if not str(schema).startswith("semiyield-"):
            raise ValueError(f"Refusing to overwrite unmanaged output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        yield temporary
        backup = None
        if target.exists():
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
            target.replace(backup)
        try:
            temporary.replace(target)
        except Exception:
            if backup is not None and backup.exists():
                backup.replace(target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
