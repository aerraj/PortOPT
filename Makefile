.PHONY: install test lint backtest clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

backtest:
	portopt --config configs/default.yaml

clean:
	rm -rf .pytest_cache .coverage reports/*.csv reports/*.json reports/*.png

