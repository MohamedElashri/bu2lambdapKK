# =============================================================================
# B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis — Makefile
#
# All targets run from the analysis/ directory.
# Default optimization method is "mva"; override with OPT_METHOD=box.
#
# Quick reference:
#   make                        Full A-Z run (studies → pipeline → systematics → collect)
#   make clean                  Remove all generated outputs (keeps lxplus PID histograms)
#   make studies                Run prerequisite studies only (MVA, trigger, reweighting, efficiency)
#   make main-pipeline          Run main Snakemake pipeline only (includes fit/sel systematics)
#   make systematics            Re-run systematic studies + aggregate (PID bootstrap + main pipeline)
#   make collect-results        Copy all outputs into results/
#   make dry-run                Show what Snakemake would execute (no actual work)
#   make dag                    Render pipeline DAG to dag.pdf
#   make help                   Print this reference
#
# NOTE: fit_systematics and selection_systematic are now integrated into the main
# Snakemake pipeline (Phase D). They are triggered automatically by compute_systematics.
# Use 'make study-fit-syst' or 'make study-sel-syst' only to run them in isolation.
# =============================================================================

OPT_METHOD ?= mva
CORES      ?= 4

SNAKEMAKE        = uv run snakemake
SNAKEMAKE_JOBS   = -j$(CORES)
SNAKEMAKE_CONFIG = --config opt_method=$(OPT_METHOD)
# NOTE: always put the rule target BEFORE $(SNAKEMAKE_CONFIG); Snakemake
# treats everything after --config as key=value pairs until the next flag.
SNAKEMAKE_FLAGS  = $(SNAKEMAKE_JOBS) $(SNAKEMAKE_CONFIG)

# Sentinel files for the MVA study outputs
MVA_LL = studies/mva_optimization/output/models/catboost_bdt_LL.cbm
MVA_DD = studies/mva_optimization/output/models/catboost_bdt_DD.cbm

# =============================================================================
# DEFAULT: full A-Z pipeline
# =============================================================================
.PHONY: all
all: studies main-pipeline systematics collect-results report-hep

# =============================================================================
# PREREQUISITES: standalone studies (must run before main pipeline)
# =============================================================================
.PHONY: studies study-mva retrain-mva study-trigger study-reweighting study-efficiency

## Train CatBoost BDTs — skipped if models already exist (use retrain-mva to force)
study-mva:
	@if [ -f "$(MVA_LL)" ] && [ -f "$(MVA_DD)" ]; then \
	    echo ">>> [Study] MVA models already present — skipping (run 'make retrain-mva' to force retraining)"; \
	else \
	    echo ">>> [Study] MVA optimization — training BDTs"; \
	    cd studies/mva_optimization && $(SNAKEMAKE) -j$(CORES); \
	fi

## Force BDT retraining even if models exist (e.g. after data or feature changes)
retrain-mva:
	@echo ">>> [Study] Force re-training MVA models"
	cd studies/mva_optimization && $(SNAKEMAKE) -j$(CORES) --forcerun run_mva_study

## Compute TIS/TOS trigger efficiency ratios
study-trigger:
	@echo ">>> [Study] TIS/TOS trigger"
	cd studies/trigger_tis_tos && $(SNAKEMAKE) -j$(CORES)

## Derive per-category kinematic reweighting weights
study-reweighting:
	@echo ">>> [Study] Kinematic reweighting (derive weights)"
	$(SNAKEMAKE) -s studies/kinematic_reweighting/Snakefile -j$(CORES) derive_weights

## Plot kinematic reweighting before/after comparisons (§6 figures)
plot-reweighting:
	@echo ">>> [Plots] Kinematic reweighting figures"
	$(SNAKEMAKE) -s studies/kinematic_reweighting/Snakefile -j$(CORES) plot_reweighting

## Compute cumulative efficiency table (uses trigger + reweighting)
study-efficiency:
	@echo ">>> [Study] Efficiency steps"
	cd studies/efficiency_steps && $(SNAKEMAKE) -j$(CORES)

## Run all prerequisite studies in dependency order (load-data must precede study-mva)
studies: load-data study-mva study-trigger study-reweighting study-efficiency

# =============================================================================
# MAIN PIPELINE (Snakemake steps 1–10 in analysis/)
# =============================================================================
.PHONY: main-pipeline load-data optimize apply-cuts mass-fitting efficiency \
        branching-ratios compare export-latex

## Steps 1–2: config validation + data/MC loading
# --forcerun ensures the script always re-executes even if the sentinel is present.
# This prevents stale caches going undetected when config files change.
load-data:
	@echo ">>> [Pipeline] Validate config + load data"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) load_data $(SNAKEMAKE_CONFIG) --forcerun load_data

## Steps 1–3: selection optimization (per branch × category)
optimize:
	@echo ">>> [Pipeline] Optimize selection"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) optimize_selection $(SNAKEMAKE_CONFIG) --debug-dag

## Steps 1–4: apply optimized cuts
apply-cuts:
	@echo ">>> [Pipeline] Apply cuts"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) apply_cuts $(SNAKEMAKE_CONFIG) --debug-dag

## Steps 1–5: mass fitting
mass-fitting:
	@echo ">>> [Pipeline] Mass fitting"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) mass_fitting $(SNAKEMAKE_CONFIG) --debug-dag

## Step 6: efficiency calculation
efficiency:
	@echo ">>> [Pipeline] Efficiency calculation"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) efficiency_calculation $(SNAKEMAKE_CONFIG) --debug-dag

## Step 7: branching fraction ratios
branching-ratios:
	@echo ">>> [Pipeline] Branching ratios"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) branching_ratios $(SNAKEMAKE_CONFIG) --debug-dag

## Step 8: branch comparison
compare:
	@echo ">>> [Pipeline] Branch comparison"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) compare_branches $(SNAKEMAKE_CONFIG) --debug-dag

## Steps 9a/9b/9/10: fit+sel systematics → aggregate → LaTeX export
## fit_systematics and selection_systematic run automatically as Snakemake dependencies.
export-latex:
	@echo ">>> [Pipeline] Compute systematics + export LaTeX"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) compute_systematics export_latex_results \
	    $(SNAKEMAKE_CONFIG) --forcerun compute_systematics --debug-dag

## Run the full main pipeline (steps 1–10)
main-pipeline:
	@echo ">>> [Pipeline] Full main pipeline"
	$(SNAKEMAKE) $(SNAKEMAKE_FLAGS) --debug-dag

# =============================================================================
# ANALYSIS NOTE PLOTS
#
# Three groups of plot scripts — each runs from analysis/ via its own Snakefile:
#
#   background_studies/  — §4 background study figures (K_S0, non-resonant, misID)
#   kinematic_reweighting/ — §6 reweighting validation figures
#   ana_note_plots/      — §3+§5 pure visualization (preselection, PID, fits, spectrum)
# =============================================================================
.PHONY: plots plot-backgrounds plot-reweighting plot-note

## §4 background study figures (K_S0 veto, non-resonant MC, partial reco, misID)
plot-backgrounds:
	@echo ">>> [Plots] Background study figures"
	$(SNAKEMAKE) -s studies/background_studies/Snakefile -j$(CORES)

## §3+§5 analysis note visualization figures (preselection, PID, MC fits, data fit, spectrum)
plot-note:
	@echo ">>> [Plots] Analysis note visualization figures"
	$(SNAKEMAKE) -s studies/ana_note_plots/Snakefile -j$(CORES)

## All analysis note figures (background + reweighting + visualization)
plots: plot-backgrounds plot-reweighting plot-note
	@echo ">>> [Plots] All analysis note figures complete"

# =============================================================================
# PHASE 4: Systematic uncertainty studies
#
# fit_systematics and selection_systematic are now Snakemake rules in the main
# pipeline (Phase D integration). They run automatically as dependencies of
# compute_systematics. The targets below are convenience wrappers to run them
# in isolation (e.g. for debugging a single variation).
# =============================================================================
.PHONY: systematics study-fit-syst study-sel-syst study-pid-bootstrap

## Fit model systematic — runs via the main Snakemake pipeline (5 variations per branch × category)
study-fit-syst:
	@echo ">>> [Systematics] Fit model variations (via main Snakefile rule fit_systematics)"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) fit_systematics $(SNAKEMAKE_CONFIG)

## Selection threshold systematic — runs via the main Snakemake pipeline (BDT ±1 step)
study-sel-syst:
	@echo ">>> [Systematics] Selection threshold (via main Snakefile rule selection_systematic)"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) selection_systematic $(SNAKEMAKE_CONFIG)

## PID bootstrap systematic (100 Gaussian smearing iterations — standalone, not in main pipeline)
study-pid-bootstrap:
	@echo ">>> [Systematics] PID bootstrap"
	cd studies/pid_cancellation && uv run python apply_pidcalib.py \
	    --bootstrap-n 100 --bootstrap-seed 42

## Run PID bootstrap then re-aggregate + re-export LaTeX.
## fit_systematics and selection_systematic are pulled in automatically by compute_systematics.
systematics: study-pid-bootstrap
	@echo ">>> [Systematics] Aggregate + update final outputs"
	$(SNAKEMAKE) $(SNAKEMAKE_JOBS) compute_systematics export_latex_results \
	    $(SNAKEMAKE_CONFIG) \
	    --forcerun compute_systematics

# =============================================================================
# COLLECT: gather everything into results/
# =============================================================================
.PHONY: collect-results report-hep
collect-results:
	@echo ">>> [Collect] Gathering outputs into results/"
	uv run python scripts/collect_results.py \
	    --pipeline-dir analysis_output/$(OPT_METHOD) \
	    --studies-dir  studies \
	    --output-dir   results

## Generate HEP-style final results report (LaTeX + plain text)
report-hep:
	@echo ">>> [Report] Generating HEP-style results report"
	uv run python scripts/generate_hep_report.py \
	    --results-dir results \
	    --output-dir  results/final

# =============================================================================
# CLEAN
# =============================================================================
.PHONY: clean clean-main clean-intermediate clean-studies clean-snakemake clean-results

## Remove main pipeline outputs and collected results
clean-main:
	@echo "Removing analysis_output/ and results/ ..."
	rm -rf analysis_output/ results/

## Remove intermediate working directories only (keeps results/ intact)
clean-intermediate:
	@echo "Removing intermediate dirs (analysis_output/ + studies/*/output/) ..."
	rm -rf analysis_output/
	rm -rf studies/mva_optimization/output/
	rm -rf studies/trigger_tis_tos/output/
	rm -rf studies/kinematic_reweighting/output/
	rm -rf studies/efficiency_steps/output/
	rm -rf studies/fit_systematics/output/
	rm -rf studies/selection_systematic/output/
	rm -rf studies/pid_cancellation/output/
	@echo "results/ preserved."

## Remove study outputs (PIDCalib histograms in pidcalib_output/ are preserved)
## NOTE: mva_optimization/output/ is NOT removed here — the trained BDT models
##       are the result of a long training run and are preserved by default.
##       Use 'make retrain-mva' to explicitly rebuild them.
clean-studies:
	@echo "Removing study outputs (mva_optimization/output/ and pid_cancellation/pidcalib_output/ preserved) ..."
	rm -rf studies/trigger_tis_tos/output/
	rm -rf studies/kinematic_reweighting/output/
	rm -rf studies/efficiency_steps/output/
	rm -rf studies/fit_systematics/output/
	rm -rf studies/selection_systematic/output/
	rm -rf studies/pid_cancellation/output/

## Remove Snakemake metadata and locks (not actual outputs)
clean-snakemake:
	@echo "Removing .snakemake metadata ..."
	rm -rf .snakemake/metadata/ .snakemake/locks/ .snakemake/incomplete/
	for d in studies/*/; do \
	    rm -rf $$d.snakemake/metadata/ $$d.snakemake/locks/ $$d.snakemake/incomplete/; \
	done

## Full clean: all outputs + snakemake metadata (preserves lxplus PID histograms)
clean: clean-main clean-studies clean-snakemake
	@echo "Clean complete."

# =============================================================================
# UTILITIES
# =============================================================================
.PHONY: dry-run dag rerun help

## Show what Snakemake would run without executing anything
dry-run:
	@echo ">>> Dry run (OPT_METHOD=$(OPT_METHOD))"
	$(SNAKEMAKE) $(SNAKEMAKE_FLAGS) -n

## Render the pipeline DAG to dag.pdf (requires graphviz)
dag:
	@echo "Generating dag.pdf ..."
	$(SNAKEMAKE) $(SNAKEMAKE_FLAGS) --dag | dot -Tpdf > dag.pdf
	@echo "Open dag.pdf to view."

## Force a complete re-run of the main pipeline (ignores existing outputs)
rerun:
	@echo ">>> Force re-run (OPT_METHOD=$(OPT_METHOD))"
	$(SNAKEMAKE) $(SNAKEMAKE_FLAGS) --forceall --debug-dag

help:
	@echo ""
	@echo "B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis — Make targets"
	@echo "=================================================="
	@echo ""
	@echo "Full workflow:"
	@echo "  make                   Full A-Z: studies → pipeline → systematics → collect"
	@echo "  make OPT_METHOD=box    Same but with box-cut optimization"
	@echo "  make CORES=8           Use 8 parallel jobs"
	@echo ""
	@echo "Individual phases:"
	@echo "  make studies           Prerequisite standalone studies (MVA, trigger, reweighting, efficiency)"
	@echo "  make main-pipeline     Main Snakemake pipeline (all steps incl. fit/sel systematics)"
	@echo "  make systematics       PID bootstrap + re-aggregate + re-export LaTeX"
	@echo "  make collect-results   Copy all outputs into results/"
	@echo ""
	@echo "Prerequisite studies (run before main pipeline):"
	@echo "  make study-mva         BDT training (skipped if models already present)"
	@echo "  make retrain-mva       Force BDT retraining (use after data/feature changes)"
	@echo "  make study-trigger     TIS/TOS trigger efficiency"
	@echo "  make study-reweighting Kinematic reweighting weights"
	@echo "  make study-efficiency  Cumulative efficiency table"
	@echo ""
	@echo "Main pipeline steps (each includes all prior steps):"
	@echo "  make load-data         Steps 1-2:      config validation + data loading"
	@echo "  make optimize          Steps 1-3:      selection optimization"
	@echo "  make apply-cuts        Steps 1-4:      apply optimized cuts"
	@echo "  make mass-fitting      Steps 1-5:      mass fitting"
	@echo "  make efficiency        Step  6:        efficiency calculation"
	@echo "  make branching-ratios  Step  7:        branching fraction ratios"
	@echo "  make compare           Step  8:        high/low yield branch comparison"
	@echo "  make export-latex      Steps 9a/9b/9/10: fit+sel systs → aggregate → LaTeX"
	@echo ""
	@echo "Systematic studies:"
	@echo "  make study-fit-syst      Fit model variations (via main pipeline rule)"
	@echo "  make study-sel-syst      Selection threshold ±1 step (via main pipeline rule)"
	@echo "  make study-pid-bootstrap PID bootstrap (100 iterations — standalone)"
	@echo ""
	@echo "  NOTE: fit_systematics and selection_systematic are Snakemake rules in the"
	@echo "  main pipeline and run automatically before compute_systematics. The targets"
	@echo "  above are convenience wrappers to run them in isolation."
	@echo ""
	@echo "Reporting:"
	@echo "  make collect-results   Copy all outputs into results/"
	@echo "  make report-hep        Generate HEP-style LaTeX + text report"
	@echo ""
	@echo "Utilities:"
	@echo "  make dry-run           Show what would run (no execution)"
	@echo "  make dag               Render pipeline DAG to dag.pdf"
	@echo "  make rerun             Force complete main pipeline re-run"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean                Remove all outputs (pidcalib_output/ preserved)"
	@echo "  make clean-intermediate   Remove intermediate dirs only (keeps results/)"
	@echo "  make clean-main           Remove analysis_output/ and results/ only"
	@echo "  make clean-studies        Remove study outputs (mva models preserved)"
	@echo "  make clean-snakemake      Remove .snakemake metadata/locks only"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  OPT_METHOD   Optimization method: mva (default) or box"
	@echo "  CORES        Parallel Snakemake jobs: 4 (default)"
	@echo ""
