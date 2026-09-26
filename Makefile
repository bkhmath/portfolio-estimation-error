.PHONY: help install test test-fast lint format data study study-offline clean

help:
	@echo "install       install the package and development dependencies"
	@echo "test          run the full test suite (no network needed)"
	@echo "test-fast     skip the end-to-end reproducibility run"
	@echo "lint          ruff check and format check"
	@echo "format        apply ruff formatting and safe fixes"
	@echo "data          download the Kenneth French files into data/cache (~20 MB, once)"
	@echo "study         run the full study, simulated and real"
	@echo "study-offline run only the simulated layer; needs no download"
	@echo "clean         remove caches and build artefacts"

install:
	python -m pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest -m "not slow"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

data:
	python scripts/download_data.py

study:
	python scripts/run_study.py

study-offline:
	python scripts/run_study.py --simulated

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
