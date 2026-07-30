PYTHON ?= python3
NPM ?= npm
UV ?= uv

.PHONY: setup setup-python setup-frontend check check-python check-frontend \
	check-release build clean

setup: setup-python setup-frontend

setup-python:
	$(UV) sync --locked --extra dev --extra online --extra offline-index

setup-frontend:
	$(NPM) --prefix frontend ci

check: check-python check-frontend check-release build

check-python:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked pytest
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked python -m offline.run \
		--config offline/fixtures/config.toml \
		--output "$$(mktemp -d)/bettercallagent-smoke"

check-frontend:
	$(NPM) --prefix frontend run check
	$(NPM) --prefix frontend run build
	$(NPM) --prefix frontend audit --omit=dev --audit-level=high

check-release:
	$(UV) run --locked python scripts/check_release.py

build:
	$(UV) run --locked python -m build
	$(UV) run --locked python scripts/check_distribution.py dist

clean:
	$(PYTHON) scripts/clean_generated.py
