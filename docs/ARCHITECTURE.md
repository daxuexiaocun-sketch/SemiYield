# Business architecture / 业务架构

SemiYield separates manufacturing yield, packaging process screening and device lifetime analysis
inside one installable Python package. It remains a single uv project, not three deployed services.

| Package | Owns | Public commands |
|---|---|---|
| `manufacturing` | SECOM acquisition, preprocessing, models, evaluation, review pages | `semiyield yield …` |
| `packaging` | Mixed-type process schema, throughput proxy labels, training, evaluation, review page | `semiyield packaging …` |
| `reliability` | NASA ingestion, Weibull/Arrhenius/required-RSF analysis, review page | `semiyield reliability …` |
| `common` | Hashes, generic reporting, metrics, validation and resource protection | Shared infrastructure |

`cli.py` only registers `yield`, `packaging`, `reliability`, and `app`; `streamlit_app.py` only
routes these business pages. Optional plotting, app, and model imports stay out of basic CLI startup.

```text
src/semiyield/
├── common/       # artifacts, generic validation, metrics, reporting, resources
├── manufacturing/# SECOM data, preprocessing, modeling, evaluation, explanations, drift
├── packaging/    # input contract, labeling, mixed preprocessing, models, evaluation
├── reliability/  # lifetime contract/models/reports and NASA preparation
├── cli.py        # public command registration
└── streamlit_app.py
```

## 1.1 migration

This is a breaking API release. The old top-level commands (`download`, `train`, `benchmark`,
`quickstart`, `data`, and `nasa`), the old Python aliases, `workflows.py`, and old joblib loading
paths were removed. Use `semiyield yield …` for the manufacturing CLI,
`semiyield.manufacturing` for Python imports, `semiyield reliability nasa …`, and
`semiyield reliability example install`. Re-train old model files with 1.1, or keep the result
exported by the version that created them. Packaging keeps its independent mixed-type schema and
does not apply SECOM numeric coercion.

## Reproducible environments

```bash
uv sync --locked --extra charts --extra dev
uv run semiyield --help
uv run pytest
uv run python -m build
```

Python 3.13 is the default development environment. The supported CI matrix remains 3.10/3.12/3.13.
Use `--extra charts` for static SVG reports, `--extra app` for Streamlit, and `--extra survival` for
reliability reports (or install `--extra all`); use `--extra explain` or `--extra tabpfn` only when
those optional routes are needed.
`uv.lock` is authoritative; exported requirements are generated via `make export-requirements`.
