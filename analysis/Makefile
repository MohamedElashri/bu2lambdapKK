# =============================================================================
# B+ -> Lambdabar p K- K+ Charmonium Analysis -- Makefile
#
# Public entrypoint for the analysis/ directory.
# The Makefile is intentionally thin: all reproducible workflow logic lives in
# the top-level Snakefile, and these targets are wrappers around named
# Snakemake workflow families or rules.
# =============================================================================

OPT_METHOD ?= mva
CORES      ?= 4
OUTPUT_ROOT ?= generated/output
CACHE_ROOT  ?= generated/cache

SNAKEMAKE = uv run snakemake
SNAKEJOBS = -j$(CORES)
SNAKECFG  = --config opt_method=$(OPT_METHOD) output_root=$(OUTPUT_ROOT) cache_root=$(CACHE_ROOT)
VALIDATION_OUTPUT_ROOT ?= $(OUTPUT_ROOT)/validation
VALIDATION_CACHE_ROOT  ?= $(CACHE_ROOT)/validation
SMOKE_YEAR           ?= 2016
SMOKE_TRACK          ?= LL
SMOKE_MAGNET         ?= MD
SMOKE_STATE          ?= Jpsi
VALIDATION_CFG       = --config opt_method=$(OPT_METHOD) output_root=$(VALIDATION_OUTPUT_ROOT) cache_root=$(VALIDATION_CACHE_ROOT)
SMOKE_CFG            = $(VALIDATION_CFG) years='["$(SMOKE_YEAR)"]' track_types='["$(SMOKE_TRACK)"]' magnets='["$(SMOKE_MAGNET)"]' states='["$(SMOKE_STATE)"]' no_cache=true

# =============================================================================
# Full workflow families
# =============================================================================
.PHONY: all full-analysis studies main-pipeline systematics plots reports \
        collect-results report-hep

all: full-analysis

full-analysis:
	@echo ">>> [Workflow] Full analysis (studies -> main pipeline -> systematics -> reports)"
	$(SNAKEMAKE) $(SNAKEJOBS) full_analysis $(SNAKECFG)

studies:
	@echo ">>> [Workflow] Prerequisite studies"
	$(SNAKEMAKE) $(SNAKEJOBS) prerequisite_studies $(SNAKECFG)

main-pipeline:
	@echo ">>> [Workflow] Main pipeline"
	$(SNAKEMAKE) $(SNAKEJOBS) main_pipeline $(SNAKECFG)

systematics:
	@echo ">>> [Workflow] Systematics aggregation"
	$(SNAKEMAKE) $(SNAKEJOBS) systematics_workflow $(SNAKECFG)

plots:
	@echo ">>> [Workflow] Presentation plots"
	$(SNAKEMAKE) $(SNAKEJOBS) presentation_plots $(SNAKECFG)

reports:
	@echo ">>> [Workflow] Collected reports"
	$(SNAKEMAKE) $(SNAKEJOBS) reports $(SNAKECFG)

collect-results:
	@echo ">>> [Collect] Gathering outputs into $(OUTPUT_ROOT)/reports/collected/"
	$(SNAKEMAKE) $(SNAKEJOBS) collect_results $(SNAKECFG)

report-hep:
	@echo ">>> [Report] Generating HEP-style report"
	$(SNAKEMAKE) $(SNAKEJOBS) report_hep $(SNAKECFG)

# =============================================================================
# Prerequisite studies
# =============================================================================
.PHONY: study-mva retrain-mva study-trigger study-reweighting study-efficiency

study-mva:
	@echo ">>> [Study] MVA optimization"
	$(SNAKEMAKE) $(SNAKEJOBS) study_mva $(SNAKECFG)

retrain-mva:
	@echo ">>> [Study] Force MVA retraining"
	$(SNAKEMAKE) $(SNAKEJOBS) study_mva $(SNAKECFG) --forcerun study_mva

study-trigger:
	@echo ">>> [Study] TIS/TOS trigger"
	$(SNAKEMAKE) $(SNAKEJOBS) study_trigger $(SNAKECFG)

study-reweighting:
	@echo ">>> [Study] Kinematic reweighting"
	$(SNAKEMAKE) $(SNAKEJOBS) study_reweighting $(SNAKECFG)

study-efficiency:
	@echo ">>> [Study] Efficiency steps"
	$(SNAKEMAKE) $(SNAKEJOBS) study_efficiency $(SNAKECFG)

# =============================================================================
# Main pipeline rules
# =============================================================================
.PHONY: load-data optimize apply-cuts mass-fitting efficiency branching-ratios \
        compare export-latex

load-data:
	@echo ">>> [Pipeline] Validate config + load data"
	$(SNAKEMAKE) $(SNAKEJOBS) load_data $(SNAKECFG) --forcerun load_data

optimize:
	@echo ">>> [Pipeline] Optimize selection"
	$(SNAKEMAKE) $(SNAKEJOBS) optimize_selection $(SNAKECFG)

apply-cuts:
	@echo ">>> [Pipeline] Apply optimized cuts"
	$(SNAKEMAKE) $(SNAKEJOBS) apply_cuts $(SNAKECFG)

mass-fitting:
	@echo ">>> [Pipeline] Mass fitting"
	$(SNAKEMAKE) $(SNAKEJOBS) mass_fitting $(SNAKECFG)

efficiency:
	@echo ">>> [Pipeline] Efficiency calculation"
	$(SNAKEMAKE) $(SNAKEJOBS) efficiency_calculation $(SNAKECFG)

branching-ratios:
	@echo ">>> [Pipeline] Branching ratios"
	$(SNAKEMAKE) $(SNAKEJOBS) branching_ratios $(SNAKECFG)

compare:
	@echo ">>> [Pipeline] Branch comparison"
	$(SNAKEMAKE) $(SNAKEJOBS) compare_branches $(SNAKECFG)

export-latex:
	@echo ">>> [Pipeline] Export LaTeX results"
	$(SNAKEMAKE) $(SNAKEJOBS) export_latex_results $(SNAKECFG)

# =============================================================================
# Presentation workflows
# =============================================================================
.PHONY: plot-backgrounds plot-reweighting plot-note

plot-backgrounds:
	@echo ">>> [Plots] Background studies"
	$(SNAKEMAKE) $(SNAKEJOBS) plot_backgrounds $(SNAKECFG)

plot-reweighting:
	@echo ">>> [Plots] Reweighting validation"
	$(SNAKEMAKE) $(SNAKEJOBS) plot_reweighting $(SNAKECFG)

plot-note:
	@echo ">>> [Plots] Analysis note plots"
	$(SNAKEMAKE) $(SNAKEJOBS) plot_note $(SNAKECFG)

# =============================================================================
# Systematic studies
# =============================================================================
.PHONY: study-fit-syst study-sel-syst study-pid-bootstrap

study-fit-syst:
	@echo ">>> [Systematics] Fit model variations"
	$(SNAKEMAKE) $(SNAKEJOBS) fit_systematics $(SNAKECFG)

study-sel-syst:
	@echo ">>> [Systematics] Selection threshold variations"
	$(SNAKEMAKE) $(SNAKEJOBS) selection_systematic $(SNAKECFG)

study-pid-bootstrap:
	@echo ">>> [Systematics] PID bootstrap"
	$(SNAKEMAKE) $(SNAKEJOBS) study_pid_bootstrap $(SNAKECFG)

# =============================================================================
# Clean
# =============================================================================
.PHONY: clean clean-generated clean-main clean-intermediate clean-studies clean-snakemake clean-results

clean-main:
	@echo "Removing $(OUTPUT_ROOT) and $(CACHE_ROOT) ..."
	rm -rf $(OUTPUT_ROOT) $(CACHE_ROOT)

clean-intermediate:
	@echo "Removing generated pipeline, study, presentation outputs and cache ..."
	rm -rf $(OUTPUT_ROOT)/pipeline/
	rm -rf $(OUTPUT_ROOT)/studies/
	rm -rf $(OUTPUT_ROOT)/presentation/
	rm -rf $(CACHE_ROOT)
	@echo "$(OUTPUT_ROOT)/reports preserved."

clean-studies:
	@echo "Removing generated study outputs (pid_cancellation/pidcalib_output/ preserved) ..."
	rm -rf $(OUTPUT_ROOT)/studies/

clean-snakemake:
	@echo "Removing .snakemake metadata ..."
	rm -rf .snakemake/
	for d in studies/*/; do \
	    rm -rf $$d.snakemake/; \
	done

clean: clean-main clean-studies clean-snakemake
	@echo "Clean complete."

clean-generated: clean
	@echo "Generated outputs removed."

# =============================================================================
# Utilities
# =============================================================================
.PHONY: dry-run validate-config smoke-imports smoke-subset smoke dag rerun help

dry-run:
	@echo ">>> Dry run (OPT_METHOD=$(OPT_METHOD))"
	$(SNAKEMAKE) $(SNAKEJOBS) full_analysis $(SNAKECFG) -n

validate-config:
	@echo ">>> [Validation] Config validation into $(VALIDATION_OUTPUT_ROOT)/pipeline/$(OPT_METHOD)"
	$(SNAKEMAKE) -j1 validate_config $(VALIDATION_CFG)
	test -f $(VALIDATION_OUTPUT_ROOT)/pipeline/$(OPT_METHOD)/.config_validated

smoke-imports:
	@echo ">>> [Validation] Shared-module import smoke check"
	uv run python scripts/smoke_imports.py

smoke-subset:
	@echo ">>> [Validation] One-year/category subset load-data smoke ($(SMOKE_YEAR), $(SMOKE_TRACK), $(SMOKE_MAGNET), $(SMOKE_STATE))"
	$(SNAKEMAKE) -j1 load_data $(SMOKE_CFG) --forcerun load_data
	test -f $(VALIDATION_OUTPUT_ROOT)/pipeline/$(OPT_METHOD)/.config_validated
	test -f $(VALIDATION_OUTPUT_ROOT)/pipeline/$(OPT_METHOD)/.data_loaded

smoke:
	@echo ">>> [Validation] Structural smoke suite"
	$(MAKE) dry-run OPT_METHOD=$(OPT_METHOD)
	$(MAKE) smoke-imports
	$(MAKE) validate-config OPT_METHOD=$(OPT_METHOD) VALIDATION_OUTPUT_ROOT=$(VALIDATION_OUTPUT_ROOT) VALIDATION_CACHE_ROOT=$(VALIDATION_CACHE_ROOT)
	$(MAKE) smoke-subset OPT_METHOD=$(OPT_METHOD) VALIDATION_OUTPUT_ROOT=$(VALIDATION_OUTPUT_ROOT) VALIDATION_CACHE_ROOT=$(VALIDATION_CACHE_ROOT) SMOKE_YEAR=$(SMOKE_YEAR) SMOKE_TRACK=$(SMOKE_TRACK) SMOKE_MAGNET=$(SMOKE_MAGNET) SMOKE_STATE=$(SMOKE_STATE)
	@echo ">>> [Validation] Presentation dry-run ($(SMOKE_YEAR), $(SMOKE_TRACK), $(SMOKE_MAGNET), $(SMOKE_STATE))"
	$(SNAKEMAKE) -j1 -n presentation_plots $(SMOKE_CFG)
	@echo "Structural smoke suite complete."

dag:
	@echo "Generating dag.pdf ..."
	$(SNAKEMAKE) $(SNAKEJOBS) full_analysis $(SNAKECFG) --dag | dot -Tpdf > dag.pdf
	@echo "Open dag.pdf to view."

rerun:
	@echo ">>> Force re-run (OPT_METHOD=$(OPT_METHOD))"
	$(SNAKEMAKE) $(SNAKEJOBS) full_analysis $(SNAKECFG) --forceall

help:
	@echo ""
	@echo "B+ -> Lambdabar p K- K+ Charmonium Analysis -- Make targets"
	@echo "=========================================================="
	@echo ""
	@echo "Workflow families:"
	@echo "  make                   Full analysis (studies -> pipeline -> systematics -> reports)"
	@echo "  make studies           Prerequisite studies only"
	@echo "  make main-pipeline     Main pipeline through branch comparison"
	@echo "  make systematics       PID bootstrap + systematic aggregation + LaTeX export"
	@echo "  make plots             Presentation plots (optional)"
	@echo "  make reports           collect-results + HEP report"
	@echo ""
	@echo "Study wrappers:"
	@echo "  make study-mva         MVA optimization"
	@echo "  make retrain-mva       Force MVA retraining"
	@echo "  make study-trigger     TIS/TOS trigger"
	@echo "  make study-reweighting Kinematic reweighting"
	@echo "  make study-efficiency  Efficiency steps"
	@echo ""
	@echo "Pipeline wrappers:"
	@echo "  make load-data         Validate config + load data"
	@echo "  make optimize          Optimize selection"
	@echo "  make apply-cuts        Apply optimized cuts"
	@echo "  make mass-fitting      Mass fitting"
	@echo "  make efficiency        Efficiency calculation"
	@echo "  make branching-ratios  Branching fraction ratios"
	@echo "  make compare           Branch comparison"
	@echo "  make export-latex      Export LaTeX results"
	@echo ""
	@echo "Presentation wrappers:"
	@echo "  make plot-backgrounds  Background-study plots"
	@echo "  make plot-reweighting  Reweighting plots"
	@echo "  make plot-note         Analysis-note plots"
	@echo ""
	@echo "Systematics wrappers:"
	@echo "  make study-fit-syst      Fit model variations"
	@echo "  make study-sel-syst      Selection-threshold variations"
	@echo "  make study-pid-bootstrap PID bootstrap study"
	@echo ""
	@echo "Reporting:"
	@echo "  make collect-results   Copy key outputs into $(OUTPUT_ROOT)/reports/collected/"
	@echo "  make report-hep        Generate HEP-style LaTeX + text report"
	@echo "  make reports           Run collect-results and report-hep together"
	@echo ""
	@echo "Validation:"
	@echo "  make validate-config   Run validate_config into $(VALIDATION_OUTPUT_ROOT)/pipeline/"
	@echo "  make smoke-imports     Import shared modules and shared config helpers"
	@echo "  make smoke-subset      Run load_data on one year/category/state into $(VALIDATION_OUTPUT_ROOT)/pipeline/"
	@echo "  make smoke             dry-run + imports + config + subset run + presentation dry-run"
	@echo ""
	@echo "Utilities:"
	@echo "  make dry-run           Show what Snakemake would execute"
	@echo "  make dag               Render pipeline DAG to dag.pdf"
	@echo "  make rerun             Force full-analysis re-run"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean                Remove all generated outputs (pidcalib_output/ preserved)"
	@echo "  make clean-generated      Alias for clean"
	@echo "  make clean-intermediate   Remove generated pipeline/study/presentation outputs only"
	@echo "  make clean-main           Remove $(OUTPUT_ROOT) and $(CACHE_ROOT)"
	@echo "  make clean-studies        Remove generated study outputs"
	@echo "  make clean-snakemake      Remove .snakemake directories"
	@echo ""
	@echo "Variables:"
	@echo "  OPT_METHOD   Optimization method: mva (default), box, or passthrough"
	@echo "  CORES        Parallel Snakemake jobs: 4 (default)"
	@echo "  OUTPUT_ROOT           Unified generated output root: $(OUTPUT_ROOT)"
	@echo "  CACHE_ROOT            Unified generated cache root: $(CACHE_ROOT)"
	@echo "  VALIDATION_OUTPUT_ROOT Smoke/validation output root: $(VALIDATION_OUTPUT_ROOT)"
	@echo "  VALIDATION_CACHE_ROOT  Smoke/validation cache root: $(VALIDATION_CACHE_ROOT)"
	@echo "  SMOKE_YEAR/SMOKE_TRACK/SMOKE_MAGNET/SMOKE_STATE  Subset run selectors"
	@echo ""
