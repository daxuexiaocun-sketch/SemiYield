SEMIYIELD ?= uv run --locked semiyield

.PHONY: install test lint build app quick demo report evidence export-requirements
install:
	uv sync --locked --extra demo --extra dev
test:
	uv run --locked pytest --cov=semiyield
lint:
	uv run --locked ruff check src tests scripts
build:
	uv run --locked python -m build
app:
	$(SEMIYIELD) app
quick:
	$(SEMIYIELD) quickstart
demo:
	$(SEMIYIELD) demo quickstart
report:
	$(SEMIYIELD) yield download
	$(SEMIYIELD) yield benchmark --profile quick --models dummy,logistic,catboost --output-dir reports/verified/yield
	$(SEMIYIELD) report --output-dir reports/verified
evidence: report
export-requirements:
	uv export --locked --extra demo --no-dev --no-emit-project --format requirements-txt --output-file requirements-demo-py313.lock
