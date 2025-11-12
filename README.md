# B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis

Analysis of B⁺ decays to Λ̄pK⁻K⁺ (ΛpK⁺K⁺) with charmonium resonances (J/ψ, ηc, χc0, χc1, ηc(2S)) at LHCb.

---

## Quick Start

### Run Full Pipeline
```bash
cd analysis
python run_pipeline.py --years 2016,2017,2018
```

### Run with Manual Cuts (Skip Optimization)
```bash
# Edit config/selection.toml [manual_cuts] section first
python run_pipeline.py --use-manual-cuts --years 2016

# Or use Makefile
make pipeline-manual-2016
```

### Command-Line Options
```bash
python run_pipeline.py --help

Options:
  --years YEARS              Comma-separated years (default: 2016)
  --track-types TYPES        Track types: LL,DD (default: LL,DD)
  --force-reoptimize         Force re-running grid scan optimization
  --no-cache                 Force reprocessing (ignore cache)
  --use-manual-cuts          Use manual cuts from config (skip optimization)
```

## Makefile Targets

If we run `make` or `make help` we get this

```
Targets:

General
  help                  Show this help message
  venv                  Activate virtual environment (checks .venv in current or parent dir)

Pipeline Execution
  pipeline              Run full analysis pipeline (all years: 2016, 2017, 2018)
  pipeline-2016         Run pipeline with 2016 data only
  pipeline-2017         Run pipeline with 2017 data only
  pipeline-2018         Run pipeline with 2018 data only
  pipeline-no-cache     Run pipeline forcing reprocessing (no cache)

Workflow Examples
  workflow-production   Production workflow: full pipeline with all data

Output Management
  show-results          Show final results table
  show-yields           Show fitted yields
  show-efficiencies     Show selection efficiencies
  list-outputs          List all output files

Cleanup
  clean-cache           Remove cached intermediate results
  clean-outputs         Remove output files (tables, plots, results)
  clean-all             Remove everything (cache + outputs)
  clean                 Alias for clean-all (removes both cache and outputs)

Development
  validate-config       Validate TOML configuration files
  check-dependencies    Check if all required packages are installed
  setup-dirs            Create necessary directories

Information
  info                  Show pipeline information
  version               Show pipeline version info
```

## Pipeline Phases

1. **Phase 1**: Configuration validation
2. **Phase 2**: Load data/MC + apply Lambda cuts (cached)
3. **Phase 3**: Cut optimization (grid scan or manual) (cached)
4. **Phase 4**: Apply optimized cuts to MC
5. **Phase 5**: Mass fitting → extract yields (cached)
6. **Phase 6**: Efficiency calculation (cached)
7. **Phase 7**: Branching fraction ratios
