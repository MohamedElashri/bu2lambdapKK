# Makefile for B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis Pipeline
#
# Quick reference:
#   make help          - Show all available targets
#   make venv          - Activate virtual environment
#   make pipeline      - Run full analysis pipeline
#   make quick-test    - Quick test with 2016 LL only
#   make clean         - Remove cache and output files

.PHONY: help clean pipeline phase2 phase3 phase5 phase6 phase7 venv

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
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[1;33m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[0;32m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[0;34m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

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

pipeline: ## Run full analysis pipeline (all years, skip optimization)
	@printf "\033[0;34mRunning complete analysis pipeline\033[0m\n"
	@printf "\033[1;33mThis will process all data and may take 20-30 minutes\033[0m\n"
	@$(PYTHON) run_pipeline.py --skip-optimization

pipeline-full: ## Run full pipeline including optimization (60+ minutes)
	@printf "\033[0;34mRunning complete pipeline with optimization\033[0m\n"
	@printf "\033[0;31mWarning: This will take 60-90 minutes\033[0m\n"
	@$(PYTHON) run_pipeline.py

pipeline-no-cache: ## Run pipeline forcing reprocessing (no cache)
	@printf "\033[0;34mRunning pipeline without cache\033[0m\n"
	@$(PYTHON) run_pipeline.py --skip-optimization --no-cache

quick-test: ## Quick pipeline test (2016 LL only, ~5 min)
	@printf "\033[0;34mQuick pipeline test with 2016 LL only\033[0m\n"
	@$(PYTHON) run_phase.py 2 --years 2016 --track-types LL
	@$(PYTHON) run_phase.py 5 --use-cached
	@$(PYTHON) run_phase.py 6 --use-cached
	@$(PYTHON) run_phase.py 7
	@printf "\033[0;32mQuick test complete! Check tables/ for results\033[0m\n"

##@ Individual Phases

phase2: ## Phase 2: Load data/MC with Lambda cuts (5-15 min)
	@printf "\033[0;34mPhase 2: Loading data and MC with Lambda cuts\033[0m\n"
	@$(PYTHON) run_phase.py 2

phase2-quick: ## Phase 2: Load 2016 only (for testing)
	@printf "\033[0;34mPhase 2: Loading 2016 data only\033[0m\n"
	@$(PYTHON) run_phase.py 2 --years 2016

phase3: ## Phase 3: Selection optimization (30-60 min, optional)
	@printf "\033[0;34mPhase 3: Selection optimization\033[0m\n"
	@printf "\033[1;33mThis takes 30-60 minutes. Use 'make pipeline' to skip.\033[0m\n"
	@printf "\033[1;33mNot yet implemented - use optimized cuts from Phase 4 tests\033[0m\n"

phase5: ## Phase 5: Mass fitting (requires Phase 2 cache)
	@printf "\033[0;34mPhase 5: Mass fitting\033[0m\n"
	@$(PYTHON) run_phase.py 5 --use-cached

phase6: ## Phase 6: Efficiency calculation (requires Phase 2 cache)
	@printf "\033[0;34mPhase 6: Efficiency calculation\033[0m\n"
	@$(PYTHON) run_phase.py 6 --use-cached

phase7: ## Phase 7: Branching fraction ratios (requires Phase 5 & 6 cache)
	@printf "\033[0;34mPhase 7: Branching fraction ratios\033[0m\n"
	@$(PYTHON) run_phase.py 7

##@ Workflow Examples

workflow-production: ## Production workflow: full pipeline with all data
	@printf "\033[0;34mProduction workflow\033[0m\n"
	@printf "\033[1;33mThis will run the complete pipeline on all data\033[0m\n"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) pipeline; \
	else \
		printf "\033[1;33mCancelled\033[0m\n"; \
	fi

##@ Output Management

show-results: ## Show final results table
	@printf "\033[0;34mFinal Branching Fraction Ratios:\033[0m\n"
	@if [ -f tables/branching_fraction_ratios.csv ]; then \
		column -s, -t < tables/branching_fraction_ratios.csv; \
	else \
		printf "\033[0;31mNo results yet. Run pipeline first.\033[0m\n"; \
	fi

show-yields: ## Show fitted yields
	@printf "\033[0;34mFitted Yields:\033[0m\n"
	@if [ -f tables/phase5_yields.csv ]; then \
		column -s, -t < tables/phase5_yields.csv; \
	else \
		printf "\033[0;31mNo yields yet. Run phase 5 first.\033[0m\n"; \
	fi

show-efficiencies: ## Show selection efficiencies
	@printf "\033[0;34mSelection Efficiencies:\033[0m\n"
	@if [ -f tables/efficiencies.csv ]; then \
		column -s, -t < tables/efficiencies.csv; \
	else \
		printf "\033[0;31mNo efficiencies yet. Run phase 6 first.\033[0m\n"; \
	fi

list-outputs: ## List all output files
	@printf "\033[0;34mOutput Files:\033[0m\n"
	@printf "\033[1;33mTables:\033[0m\n"
	@ls -lh tables/*.csv 2>/dev/null || printf "  None\n"
	@printf "\n"
	@printf "\033[1;33mPlots:\033[0m\n"
	@ls -lh plots/*.png 2>/dev/null || printf "  None\n"
	@printf "\n"
	@printf "\033[1;33mResults:\033[0m\n"
	@ls -lh results/*.md 2>/dev/null || printf "  None\n"

##@ Cleanup

clean-cache: ## Remove cached intermediate results
	@printf "\033[0;34mRemoving cache files...\033[0m\n"
	@rm -rf cache/*.pkl
	@printf "\033[0;32mCache cleaned\033[0m\n"

clean-outputs: ## Remove output files (tables, plots, results)
	@printf "\033[0;34mRemoving output files...\033[0m\n"
	@rm -rf tables/*.csv tables/*.md
	@rm -rf plots/*.png
	@rm -rf results/*.md
	@printf "\033[0;32mOutputs cleaned\033[0m\n"

clean-all: clean-cache clean-outputs ## Remove everything (cache + outputs)
	@printf "\033[0;32mAll intermediate files removed\033[0m\n"

clean: clean-cache ## Alias for clean-cache

##@ Development

validate-config: ## Validate TOML configuration files
	@printf "\033[0;34mValidating configuration files...\033[0m\n"
	@$(PYTHON) -c "from modules.data_handler import TOMLConfig; config = TOMLConfig(); print('✓ All config files valid')"

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
	@mkdir -p cache tables plots results
	@mkdir -p plots/fits plots/optimization
	@printf "\033[0;32mDirectories created\033[0m\n"

##@ Documentation

docs: ## Open README.md
	@if command -v xdg-open > /dev/null; then \
		xdg-open README.md; \
	elif command -v open > /dev/null; then \
		open README.md; \
	else \
		cat README.md; \
	fi

show-plan: ## Show plan.md
	@if command -v less > /dev/null; then \
		less plan.md; \
	else \
		cat plan.md; \
	fi

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
	@printf "  2. Data/MC loading + Lambda cuts (5-15 min)\n"
	@printf "  3. Selection optimization (optional, 30-60 min)\n"
	@printf "  4. Apply optimized cuts (< 1 min)\n"
	@printf "  5. Mass fitting (5-10 min)\n"
	@printf "  6. Efficiency calculation (2-5 min)\n"
	@printf "  7. Branching fraction ratios (< 1 min)\n\n"
	@printf "\033[0;32mQuick Start:\033[0m\n"
	@printf "  make venv              # Check/activate virtual environment\n"
	@printf "  make quick-test        # Test with 2016 LL (~5 min)\n"
	@printf "  make pipeline          # Full analysis (~20-30 min)\n"
	@printf "  make show-results      # View final results\n\n"
	@printf "\033[0;32mDocumentation:\033[0m\n"
	@printf "  README.md   - Complete usage guide\n"
	@printf "  plan.md     - Detailed phase specifications\n\n"
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
