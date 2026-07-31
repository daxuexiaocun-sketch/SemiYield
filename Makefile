SEMIYIELD ?= semiyield

.PHONY: install test lint build app quick report evidence
install:
	python -m pip install -e ".[all,dev]"
test:
	pytest --cov=semiyield
lint:
	ruff check src tests
build:
	python -m build
app:
	$(SEMIYIELD) app
quick:
	$(SEMIYIELD) quickstart
report:
	$(SEMIYIELD) download
	$(SEMIYIELD) benchmark --profile quick --models dummy,logistic,catboost --output-dir reports/verified/yield
	$(SEMIYIELD) report --output-dir reports/verified
evidence: report
