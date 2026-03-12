.PHONY: all clean run show-dag help

# Optimization method: "box" or "mva" (overridable via command line: make OPT_METHOD=mva)
OPT_METHOD ?=

# Default target
all: run

# Run the complete Snakemake pipeline
run:
	@echo "Running full analysis pipeline via Snakemake (Method: $(or $(OPT_METHOD),from config))..."
	uv run snakemake --cores all $(if $(OPT_METHOD),--config opt_method=$(OPT_METHOD))

# Explicit method targets
box:
	@$(MAKE) run OPT_METHOD=box

mva:
	@$(MAKE) run OPT_METHOD=mva

# Force a complete re-run
rerun:
	@echo "Forcing a complete re-run of the analysis pipeline (Method: $(or $(OPT_METHOD),from config))..."
	uv run snakemake --cores all --forceall $(if $(OPT_METHOD),--config opt_method=$(OPT_METHOD))

# Clean only output files
clean-output:
	@echo "Removing output directory..."
	rm -rf analysis_output

# Clean only cached files
clean-cache:
	@echo "Removing cache directory..."
	rm -rf cache

# Clean catboost info
clean-catboost:
	@echo "Removing catboost info directory..."
	rm -rf catboost_info

# Clean everything (output and cache)
clean: clean-output clean-cache clean-catboost
	@echo "Removing .snakemake internal directory..."
	rm -rf .snakemake

# Visualize the pipeline DAG (requires graphviz/dot)
show-dag:
	@echo "Generating DAG visualization to dag.pdf (Method: $(or $(OPT_METHOD),from config))..."
	uv run snakemake --dag $(if $(OPT_METHOD),--config opt_method=$(OPT_METHOD)) | dot -Tpdf > dag.pdf
	@echo "Done. Open dag.pdf to view."

# Dry run
dry-run:
	@echo "Performing dry run of the pipeline (Method: $(or $(OPT_METHOD),from config))..."
	uv run snakemake -n $(if $(OPT_METHOD),--config opt_method=$(OPT_METHOD))

# Help text
help:
	@echo "Available commands:"
	@echo "  make run          : Run the complete analysis pipeline (default)"
	@echo "  make rerun        : Force a complete re-run from scratch"
	@echo "  make dry-run      : Show what would be run without actually executing"
	@echo "  make clean        : Remove all outputs, cache, catboost info, and internal snakemake files"
	@echo "  make clean-output : Remove only the analysis_output directory"
	@echo "  make clean-cache  : Remove only the cache directory"
	@echo "  make clean-catboost : Remove only the catboost info directory"
	@echo "  make show-dag     : Generate a PDF of the Snakemake DAG (requires graphviz)"
