# Business architecture / 业务架构

SemiYield separates manufacturing yield, packaging process screening and device lifetime analysis
inside one installable Python package. It remains a single uv project, not three deployed services.

| Package | Owns | Public commands |
|---|---|---|
| `yield_risk` | SECOM acquisition, yield benchmark, manufacturing model service, review pages | `semiyield yield …` |
| `packaging` | Mixed-type process schema, throughput proxy labels, training, evaluation, review page | `semiyield packaging …` |
| `reliability` | NASA ingestion, lifetime/Arrhenius/optional survival analysis, review page | `semiyield reliability …`, `semiyield nasa …` |
| `common` | Numeric model primitives, metrics, preprocessing, I/O, explanations, drift, plotting, resource protection | Shared Python interfaces |
| `demo` | Synthetic generation, cross-stage orchestration, static reports | `semiyield demo …` |

`cli.py` registers commands and compatibility exports; `streamlit_app.py` routes business pages.
`workflows.py` retains the legacy SECOM plus reliability quickstart and experiment-manifest commands.
Business packages depend on shared utilities, not on one another. Only the demo/workflow layer
coordinates business services. Optional plotting/app/model imports stay out of basic CLI startup.

## Compatibility

Existing top-level SECOM commands (`download`, `train`, `benchmark`, etc.) remain aliases.
`quickstart` still downloads SECOM and installs the small lifetime example; use `demo quickstart`
for the new entirely synthetic three-stage workflow. Existing `data download-demo` and NASA
commands remain available.

Former Python modules are compatibility aliases to their implementation modules. Old imports such
as `semiyield.data.SecomDataset`, `semiyield.modeling.ModelArtifact` and
`semiyield.preprocessing.ColumnCleaner` still resolve, including when loading trusted old joblib
artifacts. `semiyield.reliability` is now a package exporting the existing lifetime API.
Packaging uses its own `PackagingArtifact` and does not pass categorical values through SECOM's
numeric coercion. Artifact loaders only accept their respective artifact types.

## Reproducible environments

```bash
uv sync --locked --extra demo --extra dev
uv run semiyield --help
uv run pytest
uv run python -m build
```

Python 3.13 is the default demonstration environment. The supported CI matrix remains
3.10/3.12/3.13. Existing extras are retained; `demo` adds plotting, app and CatBoost dependencies,
while the demo's default models require neither CatBoost nor survival forests at runtime.
Use `--extra survival`, `--extra explain` or `--extra tabpfn` only when needed.
`uv.lock` is authoritative; exported requirements are generated via `make export-requirements`.
