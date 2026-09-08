SEMIYIELD ?= uv run --locked semiyield

.PHONY: install test lint build app report export-requirements
install:
	uv sync --locked --extra charts --extra dev
test:
	uv run --locked pytest --cov=semiyield
lint:
	uv run --locked ruff check src tests scripts
build:
	uv run --locked python -m build
app:
	$(SEMIYIELD) app
report:
	$(SEMIYIELD) yield download
	$(SEMIYIELD) yield benchmark --profile quick --models dummy,logistic,catboost --output-dir reports/verified/yield
export-requirements:
	uv export --locked --extra charts --no-dev --no-emit-project --format requirements-txt --output-file requirements-charts-py313.lock
