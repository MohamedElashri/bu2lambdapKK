# B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis

Analysis of B⁺ decays to Λ̄pK⁻K⁺ with charmonium resonances (J/ψ, ηc, χc0, χc1, ηc(2S)) at LHCb.
Measures branching fraction ratios relative to J/ψ using self-normalization.

## Quick Start

```bash
uv sync                              # Install dependencies (run once, from repo root)
cd analysis/
uv run snakemake -j1                  # Run the full pipeline
```

See [`analysis/README.md`](analysis/README.md) for full documentation of all pipeline steps,
configuration options, and output files.

## Repository Structure

```
├── analysis/          # Snakemake-based analysis pipeline (active)
│   ├── Snakefile      #   Workflow definition (7 steps)
│   ├── config/        #   11 TOML configuration files
│   ├── modules/       #   Core analysis modules
│   ├── scripts/       #   Snakemake wrapper scripts
│   └── tests/         #   Test suite
├── eff/               # Standalone efficiency study scripts
├── archive/           # Original Makefile-based pipeline (for reference)
│   └── analysis/      #   Pre-Snakemake code (run_pipeline.py + Makefile)
└── docs/              # Documentation
```

## Pipeline Overview

| Step | Description |
|------|-------------|
| 1 | Configuration validation |
| 2 | Data/MC loading + Λ pre-selection |
| 3 | Selection optimization (N-D grid scan or manual cuts) |
| 4 | Apply optimized cuts |
| 5 | Simultaneous mass fitting (RooFit) |
| 6 | Selection efficiency calculation |
| 7 | Branching fraction ratios |
