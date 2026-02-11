# Sideband Background Modeling Study

## Motivation

The standard M(Λ̄pK⁻) fit uses an ARGUS function for the combinatorial background. This is a parametric assumption — the true background shape may differ. This study validates and implements a **data-driven** alternative: using M(B⁺) sideband events to construct a background template.

If the M(Λ̄pK⁻) shape is independent of M(B⁺) in sideband regions, we can use sideband data to model the background in the signal region. The yield difference between ARGUS and template fits provides a systematic uncertainty estimate.

## Methodology

### Phase A: Shape Validation (`study_validate_shapes.py`)

1. Divide M(B⁺) into 4 sideband slices:
   - Far-left [2800, 3500] MeV
   - Mid-left [3500, 4500] MeV
   - Near-left [4500, 5150] MeV
   - Right [5330, 5500] MeV
2. Project M(Λ̄pK⁻) for each slice, normalise to unit area
3. Compare shapes via KS and χ² tests
4. If shapes are consistent (p > 0.05) → sideband method is valid

### Phase B: Template Extraction (`study_extract_template.py`)

1. Select events in near-signal sidebands: [4500, 5150] + [5330, 5500] MeV
2. Create M(Λ̄pK⁻) histogram
3. Smooth with Gaussian filter (σ = 3 bins)
4. Save as ROOT histogram + RooFit workspace

### Phase C: Model Comparison (`study_compare_models.py`)

1. Select B⁺ signal region [5229, 5329] MeV (±50 MeV around PDG mass)
2. Fit M(Λ̄pK⁻) with Voigtian signal (5 charmonium states) + ARGUS background
3. Fit M(Λ̄pK⁻) with Voigtian signal (5 charmonium states) + template background
4. Compare yields → systematic uncertainty

### Charmonium states in signal model

| State | Mass [MeV] | Width [MeV] |
|-------|-----------|-------------|
| ηc(1S) | 2983.9 | from PDG |
| J/ψ | 3096.9 | from PDG |
| χc0 | 3414.7 | from PDG |
| χc1 | 3510.7 | from PDG |
| ηc(2S) | 3637.5 | from PDG |

## Outputs

### Phase A
- `output/shape_validation.pdf` — 3-page PDF: shape overlay, ratio plot, summary table
- `output/shape_validation_tests.csv` — KS and χ² test results

### Phase B
- `output/background_template.pdf` — 2-page PDF: raw + smoothed template, normalised shape
- `output/background_template.root` — ROOT file with raw/smoothed histograms + RooFit workspace

### Phase C
- `output/background_comparison.pdf` — 3-page PDF: side-by-side fits, yield comparison bar chart, summary table
- `output/background_comparison_results.csv` — yields, differences, systematic uncertainties

## Running

```bash
cd analysis/studies/sideband_background
uv run snakemake -j1
```

This runs all 3 phases in dependency order (validate → extract → compare).

To run individual phases:
```bash
uv run snakemake validate_shapes -j1
uv run snakemake extract_template -j1
uv run snakemake compare_models -j1
```

Requires Step 2 cached data from the main pipeline:
```bash
cd analysis/ && uv run snakemake load_data -j1
```

## Results

*(To be filled after running the study.)*

## Ported from

`archive/analysis/studies/sideband_background/` (5 scripts + config)

### Changes from original

- **Consolidation**: 5 scripts + config → 3 scripts + config (merged `template_fitter.py` into `compare_models.py`)
- **Plotting**: ROOT TCanvas/TH1 → matplotlib + mplhep (LHCb2 style)
- **Fitting**: RooFit kept for Voigtian + ARGUS/template fits
- **New**: Ratio panel in shape validation
- **New**: Colour-coded KS/χ² compatibility in summary table
- **New**: Yield comparison bar chart with relative difference labels
- **Output**: Text tables → CSV with full metadata
- **Data loading**: Direct ROOT file access → cached Step 2 awkward arrays
- **Smoothing**: ROOT TH1::Smooth / custom KDE → scipy gaussian_filter1d
- **Config**: Standalone config.py preserved (mass region definitions)
