# .PHONY: check style install install-dev test clean all
.PHONY: check style install install-dev clean all

check:
	ruff check
	ruff format --diff

style:
	ruff format
	ruff check --fix

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

# Add this back in once we have some unit tests
# test:
# 	pytest -v -s

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache/

all: style
# all: style test