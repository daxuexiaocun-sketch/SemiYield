# Contributing

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). Python 3.10, 3.12 and 3.13 are covered by CI.

```bash
uv sync --locked --extra demo --extra dev
uv run ruff check src tests scripts
uv run pytest --cov=semiyield
uv run python -m build
uv run semiyield demo quickstart
```

Keep business logic in its owning package and reusable infrastructure in `common`.
See [architecture](docs/ARCHITECTURE.md) and [the demo guide](docs/DEMO.md).
Test threshold isolation, schema compatibility and stage routing when changing data workflows.
Preserve existing import aliases and CLI commands. Keep real and synthetic results separate.

`uv.lock` is the source of truth. After intentional dependency edits run `uv lock`, review its diff,
and verify with `uv sync --locked --extra demo --extra dev`. Do not hand-maintain a second lockfile.
For consumers requiring requirements format, run `make export-requirements`; its output is derived
from `uv.lock` and is not an independent Python 3.13 dependency definition.
