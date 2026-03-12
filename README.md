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
