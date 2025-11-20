# Makefile for B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis Pipeline
#
# Quick reference:
#   make help          - Show all available targets
#   make venv          - Activate virtual environment
#   make test          - Run all tests with coverage
#   make pipeline      - Run full analysis pipeline (All years: 2016, 2017, 2018)
#   make pipeline-2016 - Run with 2016 data only
#   make pipeline-2017 - Run with 2017 data only
#   make pipeline-2018 - Run with 2018 data only
#   make clean         - Remove cache and output files

.PHONY: help clean clean-cache clean-outputs pipeline venv test test-unit test-integration test-validation test-quick test-coverage test-watch test-failed test-verbose

# Virtual environment detection
VENV_DIR := $(shell if [ -d ".venv" ]; then echo ".venv"; elif [ -d "../.venv" ]; then echo "../.venv"; else echo ""; fi)
VENV_ACTIVATE := $(if $(VENV_DIR),$(VENV_DIR)/bin/activate,)

# Python interpreter (use venv if available)
PYTHON := $(if $(VENV_ACTIVATE),$(VENV_DIR)/bin/python,python)

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Show this help message
	@printf "\033[0;34mB⁺ → Λ̄pK⁻K⁺ Charmonium Analysis Pipeline\033[0m\n\n"
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[1;33m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[0;32m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[0;34m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

venv: ## Activate virtual environment (checks .venv in current or parent dir)
	@if [ -n "$(VENV_DIR)" ]; then \
		printf "\033[0;32mVirtual environment found: $(VENV_DIR)\033[0m\n"; \
		printf "\033[1;33mRun: source $(VENV_ACTIVATE)\033[0m\n"; \
	else \
		printf "\033[0;31mNo virtual environment found in .venv or ../.venv\033[0m\n"; \
		printf "\033[1;33mCreate one with: python -m venv .venv\033[0m\n"; \
		exit 1; \
	fi

##@ Pipeline Execution

pipeline: ## Run full analysis pipeline (all years: 2016, 2017, 2018)
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════════════════════\033[0m\n"
	@printf "\033[0;34mComplete Analysis Pipeline - All Years (2016, 2017, 2018)\033[0m\n"
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════════════════════\033[0m\n"
	@printf "\033[1;33m📊 Phases: 1 (Config Validation) → 2 → 3 → 4 → 5 → 6 → 7\033[0m\n"
	@printf "\033[1;33m📊 Includes: Per-year + Combined fits\033[0m\n\n"
	@$(PYTHON) run_pipeline.py --years 2016,2017,2018

pipeline-2016: ## Run pipeline with 2016 data only
	@printf "\033[0;34m═══ Pipeline: 2016 Only ═══\033[0m\n"
	@$(PYTHON) run_pipeline.py --years 2016

pipeline-2017: ## Run pipeline with 2017 data only
	@printf "\033[0;34m═══ Pipeline: 2017 Only ═══\033[0m\n"
	@$(PYTHON) run_pipeline.py --years 2017

pipeline-2018: ## Run pipeline with 2018 data only
	@printf "\033[0;34m═══ Pipeline: 2018 Only ═══\033[0m\n"
	@$(PYTHON) run_pipeline.py --years 2018

pipeline-no-cache: ## Run pipeline forcing reprocessing (no cache)
	@printf "\033[0;34m═══ Pipeline (No Cache) ═══\033[0m\n"
	@printf "\033[0;31m⚠️  Forcing full reprocessing\033[0m\n"
	@$(PYTHON) run_pipeline.py --years 2016,2017,2018 --no-cache

pipeline-manual: ## Run pipeline with manual cuts (skips grid scan optimization)
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════════════════════\033[0m\n"
	@printf "\033[0;34mPipeline with Manual Cuts - All Years (2016, 2017, 2018)\033[0m\n"
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════════════════════\033[0m\n"
	@printf "\033[1;33m⚡ Skipping Phase 3 optimization (using manual cuts from config)\033[0m\n"
	@printf "\033[1;33m📝 Edit config/selection.toml [manual_cuts] section to define cuts\033[0m\n\n"
	@$(PYTHON) run_pipeline.py --use-manual-cuts --years 2016,2017,2018

pipeline-manual-2016: ## Run pipeline with manual cuts (2016 only, fast!)
	@printf "\033[0;34m═══ Pipeline (Manual Cuts): 2016 Only ═══\033[0m\n"
	@printf "\033[1;33m⚡ Skipping optimization (saves 5-10 minutes)\033[0m\n"
	@$(PYTHON) run_pipeline.py --use-manual-cuts --years 2016

pipeline-manual-2017: ## Run pipeline with manual cuts (2017 only)
	@printf "\033[0;34m═══ Pipeline (Manual Cuts): 2017 Only ═══\033[0m\n"
	@$(PYTHON) run_pipeline.py --use-manual-cuts --years 2017

pipeline-manual-2018: ## Run pipeline with manual cuts (2018 only)
	@printf "\033[0;34m═══ Pipeline (Manual Cuts): 2018 Only ═══\033[0m\n"
	@$(PYTHON) run_pipeline.py --use-manual-cuts --years 2018

pipeline-manual-test: ## Quick test with manual cuts (2016, no cache)
	@printf "\033[0;34m═══ Quick Test (Manual Cuts + No Cache) ═══\033[0m\n"
	@printf "\033[1;33m⚡ Fast iteration for testing cut values\033[0m\n"
	@$(PYTHON) run_pipeline.py --use-manual-cuts --no-cache --years 2016


##@ Output Management

show-results: ## Show final results table
	@printf "\033[0;34mFinal Branching Fraction Ratios:\033[0m\n"
	@if [ -f analysis_output/tables/branching_fraction_ratios.csv ]; then \
		column -s, -t < analysis_output/tables/branching_fraction_ratios.csv; \
	else \
		printf "\033[0;31mNo results yet. Run pipeline first.\033[0m\n"; \
	fi

show-yields: ## Show fitted yields
	@printf "\033[0;34mFitted Yields:\033[0m\n"
	@if [ -f analysis_output/tables/phase5_yields.csv ]; then \
		column -s, -t < analysis_output/tables/phase5_yields.csv; \
	else \
		printf "\033[0;31mNo yields yet. Run phase 5 first.\033[0m\n"; \
	fi

show-efficiencies: ## Show selection efficiencies
	@printf "\033[0;34mSelection Efficiencies:\033[0m\n"
	@if [ -f analysis_output/tables/efficiencies.csv ]; then \
		column -s, -t < analysis_output/tables/efficiencies.csv; \
	else \
		printf "\033[0;31mNo efficiencies yet. Run phase 6 first.\033[0m\n"; \
	fi

list-outputs: ## List all output files
	@printf "\033[0;34mOutput Files:\033[0m\n"
	@printf "\033[1;33mTables:\033[0m\n"
	@ls -lh analysis_output/tables/*.csv 2>/dev/null || printf "  None\n"
	@printf "\n"
	@printf "\033[1;33mPlots:\033[0m\n"
	@ls -lh analysis_output/plots/*.pdf 2>/dev/null || printf "  None\n"
	@printf "\n"
	@printf "\033[1;33mResults:\033[0m\n"
	@ls -lh analysis_output/results/*.md 2>/dev/null || printf "  None\n"

##@ Cleanup

clean-cache: ## Remove cached intermediate results
	@printf "\033[0;34mRemoving cache files...\033[0m\n"
	@rm -rfv cache/data/*.pkl cache/metadata/*.json
	@printf "\033[0;32mCache cleaned\033[0m\n"

clean-outputs: ## Remove output files (tables, plots, results)
	@printf "\033[0;34mRemoving output files...\033[0m\n"
	@rm -rfv analysis_output
	@printf "\033[0;32mOutputs cleaned\033[0m\n"

clean: clean-cache clean-outputs ## Remove everything (cache + outputs)
	@printf "\033[0;32mAll intermediate files removed\033[0m\n"

##@ Development

validate-config: ## Validate TOML configuration files
	@printf "\033[0;34mValidating configuration files...\033[0m\n"
	@$(PYTHON) utils/validate_config.py

check-dependencies: ## Check if all required packages are installed
	@printf "\033[0;34mChecking dependencies...\033[0m\n"
	@$(PYTHON) -c "import uproot; print('✓ uproot')"
	@$(PYTHON) -c "import awkward; print('✓ awkward')"
	@$(PYTHON) -c "import numpy; print('✓ numpy')"
	@$(PYTHON) -c "import pandas; print('✓ pandas')"
	@$(PYTHON) -c "import matplotlib; print('✓ matplotlib')"
	@$(PYTHON) -c "import tomli; print('✓ tomli')"
	@$(PYTHON) -c "import vector; print('✓ vector')"
	@$(PYTHON) -c "import ROOT; print('✓ ROOT (PyROOT)')"
	@printf "\033[0;32mAll dependencies installed\033[0m\n"

setup-dirs: ## Create necessary directories
	@printf "\033[0;34mCreating directories...\033[0m\n"
	@mkdir -p cache analysis_output/tables analysis_output/plots analysis_output/results
	@mkdir -p analysis_output/plots/fits analysis_output/plots/optimization
	@printf "\033[0;32mDirectories created\033[0m\n"

##@ Testing

test: ## Run all tests with coverage
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════\033[0m\n"
	@printf "\033[0;34m  Running Test Suite\033[0m\n"
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════\033[0m\n"
	@cd .. && PYTHONPATH=. .venv/bin/python -m pytest analysis/tests/ -v --cov=analysis/modules --cov-report=term-missing --cov-report=html

test-unit: ## Run only unit tests
	@printf "\033[0;34mRunning unit tests...\033[0m\n"
	@cd .. && PYTHONPATH=. .venv/bin/python -m pytest analysis/tests/unit/ -v

test-integration: ## Run only integration tests
	@printf "\033[0;34mRunning integration tests...\033[0m\n"
	@cd .. && PYTHONPATH=. .venv/bin/python -m pytest analysis/tests/integration/ -v -m integration

test-validation: ## Run only validation tests
	@printf "\033[0;34mRunning validation tests...\033[0m\n"
	@cd .. && PYTHONPATH=. .venv/bin/python -m pytest analysis/tests/validation/ -v -m validation

test-quick: ## Run tests without coverage (faster)
	@printf "\033[0;34mRunning tests (quick mode)...\033[0m\n"
	@cd .. && PYTHONPATH=. .venv/bin/python -m pytest analysis/tests/ -v

test-coverage: ## Open HTML coverage report in browser (run 'make test' first)
	@if [ -f ../htmlcov/index.html ]; then \
		printf "\033[0;32mOpening coverage report...\033[0m\n"; \
		xdg-open ../htmlcov/index.html 2>/dev/null || open ../htmlcov/index.html 2>/dev/null || printf "\033[1;33mReport location: htmlcov/index.html\033[0m\n"; \
	else \
		printf "\033[0;31mNo coverage report found. Run 'make test' first.\033[0m\n"; \
	fi

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@printf "\033[0;34mWatching for changes...\033[0m\n"
	@cd .. && PYTHONPATH=. .venv/bin/python -m pytest_watch analysis/tests/ -- -v

test-failed: ## Re-run only failed tests
	@printf "\033[0;34mRe-running failed tests...\033[0m\n"
	@cd .. && PYTHONPATH=. .venv/bin/python -m pytest analysis/tests/ --lf -v

test-verbose: ## Run tests with extra verbosity
	@printf "\033[0;34mRunning tests (verbose)...\033[0m\n"
	@cd .. && PYTHONPATH=. .venv/bin/python -m pytest analysis/tests/ -vv -s

##@ Code Quality

pre-commit-install: ## Install pre-commit hooks
	@printf "\033[0;34mInstalling pre-commit hooks...\033[0m\n"
	@cd .. && pre-commit install
	@cd .. && pre-commit install --hook-type pre-push
	@printf "\033[0;32m✓ Pre-commit hooks installed\033[0m\n"
	@printf "\033[1;33mHooks will run automatically on git commit\033[0m\n"

pre-commit-run: ## Run all pre-commit hooks manually
	@printf "\033[0;34mRunning pre-commit hooks on all files...\033[0m\n"
	@cd .. && pre-commit run --all-files

pre-commit-update: ## Update pre-commit hook versions
	@printf "\033[0;34mUpdating pre-commit hooks...\033[0m\n"
	@cd .. && pre-commit autoupdate
	@printf "\033[0;32m✓ Pre-commit hooks updated\033[0m\n"

format: ## Format code with black and isort
	@printf "\033[0;34mFormatting code...\033[0m\n"
	@cd .. && black analysis/ --line-length=100
	@cd .. && isort analysis/ --profile=black --line-length=100
	@printf "\033[0;32m✓ Code formatted\033[0m\n"

lint: ## Lint code with ruff
	@printf "\033[0;34mLinting code...\033[0m\n"
	@cd .. && ruff check analysis/ --fix
	@printf "\033[0;32m✓ Linting complete\033[0m\n"

typecheck: ## Run type checking with mypy
	@printf "\033[0;34mType checking...\033[0m\n"
	@cd .. && mypy analysis/modules/ --ignore-missing-imports --no-strict-optional
	@printf "\033[0;32m✓ Type checking complete\033[0m\n"

quality: format lint typecheck ## Run all code quality checks
	@printf "\033[0;32m✓ All quality checks complete\033[0m\n"

##@ Information

info: ## Show pipeline information
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════\033[0m\n"
	@printf "\033[0;34m  B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis Pipeline\033[0m\n"
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════\033[0m\n\n"
	@printf "\033[0;32mAnalysis Goal:\033[0m\n"
	@printf "  Measure branching fraction ratios of charmonium states\n"
	@printf "  (J/ψ, ηc, χc0, χc1) in B⁺ → Λ̄pK⁻K⁺ decays\n\n"
	@printf "\033[0;32mKey Strategy:\033[0m\n"
	@printf "  Self-normalization to J/ψ - measure RATIOS\n"
	@printf "  No absolute branching fractions needed!\n\n"
	@printf "\033[0;32mPipeline Phases:\033[0m\n"
	@printf "  0. Branch configuration (done)\n"
	@printf "  1. Configuration validation (automatic)\n"
	@printf "  2. Data/MC loading + Lambda cuts\n"
	@printf "  3. Selection optimization (optional)\n"
	@printf "  4. Apply optimized cuts\n"
	@printf "  5. Mass fitting\n"
	@printf "  6. Efficiency calculation\n"
	@printf "  7. Branching fraction ratios\n\n"
	@printf "\033[0;32mQuick Start:\033[0m\n"
	@printf "  make venv              # Check/activate virtual environment\n"
	@printf "  make pipeline-2016     # Test with 2016 data\n"
	@printf "  make pipeline          # Full analysis (all years)\n"
	@printf "  make show-results      # View final results\n\n"
	@printf "\033[0;32mConfiguration:\033[0m\n"
	@printf "  Verbosity settings: config/data.toml [verbosity] section\n"
	@printf "  Override with: ANALYSIS_DATA_LOADING=on/off\n\n"
	@printf "\033[0;32mDocumentation:\033[0m\n"
	@printf "\033[0;34m═══════════════════════════════════════════════════════════════\033[0m\n"

version: ## Show pipeline version info
	@printf "\033[0;34mPipeline Version Information:\033[0m\n"
	@printf "Analysis: B⁺ → Λ̄pK⁻K⁺ Charmonium\n"
	@printf "Draft Analysis - Statistical Uncertainties Only\n"
	@printf "Last Updated: November 2, 2025\n\n"
	@printf "Python: $$($(PYTHON) --version)\n"
	@printf "ROOT: $$($(PYTHON) -c 'import ROOT; print(ROOT.__version__)')\n"
	@printf "Uproot: $$($(PYTHON) -c 'import uproot; print(uproot.__version__)')\n"
	@printf "Awkward: $$($(PYTHON) -c 'import awkward as ak; print(ak.__version__)')\n"
