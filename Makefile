# Reproducible pipeline for the statistical arbitrage backtester.
#
# `make backtest` regenerates every number and chart the README reports, from
# fixed seeds, on a clean checkout. Nothing in the README is hand-written.

SHELL := /bin/bash
ENGINE_DIR := engine
PYTHON := python3
PYTEST := $(PYTHON) -m pytest

# opam is not on PATH in a non-login shell, so every dune invocation is wrapped.
DUNE := eval $$(opam env) && dune

.DEFAULT_GOAL := help
.PHONY: help build test test-ocaml test-python backtest report clean deps fmt check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

deps: ## Install OCaml and Python dependencies
	opam install -y dune alcotest qcheck
	$(PYTHON) -m pip install -r requirements.txt

build: ## Build the OCaml engine
	cd $(ENGINE_DIR) && $(DUNE) build

test-ocaml: build ## Run the OCaml test suite
	cd $(ENGINE_DIR) && $(DUNE) test

test-python: build ## Run the Python test suite (integration tests need the engine)
	$(PYTEST) python/tests -v

test: test-ocaml test-python ## Run both test suites

backtest: build ## Run the full pipeline and regenerate every reported number
	$(PYTHON) scripts/run_pipeline.py

backtest-offline: build ## Run the pipeline without the network fetch
	$(PYTHON) scripts/run_pipeline.py --skip-real-data

report: ## Regenerate the README results section from reports/summary.json
	$(PYTHON) scripts/generate_readme.py

all: test backtest report ## Test, backtest, and regenerate the README

check: test ## Alias for `test`, for CI readability

clean: ## Remove build artifacts and generated outputs
	cd $(ENGINE_DIR) && $(DUNE) clean
	rm -rf reports data/raw data/processed
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
