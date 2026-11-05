PYTHON ?= python3

.PHONY: install demo test lint typecheck serve-dashboard

install:
	$(PYTHON) -m pip install -e ".[dev]"

demo:
	$(PYTHON) -m benchlab run --config configs/example.yaml

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy

test: lint typecheck
	$(PYTHON) -m pytest

serve-dashboard:
	$(PYTHON) -m http.server 8741 --directory space
