.PHONY: install test lint benchmark serve
install:
	pip install -r requirements-dev.txt && pip install -e .
test:
	pytest -q
lint:
	ruff check src api tests
benchmark:
	python -m ragbench.experiment
serve:
	uvicorn api.main:app --reload
