# Sideband Background Modeling Study

Data-driven combinatorial background modeling using B+ mass sideband regions for the Bu → Λ̄pK⁺K⁻ analysis.

## Overview

This study uses events in the M(B+) sideband regions (away from the B+ signal peak at 5279 MeV) to model the combinatorial background shape in the M(Λ̄pK⁻) spectrum. This is a standard LHCb technique for data-driven background estimation.

## Physics Motivation

From the 2D plot (M(B+) vs M(Λ̄pK⁻)):
- Events **left of** the B+ mass line are combinatorial background (fake B+ candidates)
- These events have the same kinematic properties as background under the B+ signal peak
- The M(Λ̄pK⁻) distribution from sidebands can be used as a background template

## Directory Structure

```
sideband_background/
├── __init__.py           # Module initialization
├── config.py             # Mass region definitions and configuration
├── data_loader.py        # Data loading utilities
├── validate_shapes.py    # Phase 1: Shape validation
├── extract_template.py   # Phase 2: Template extraction
├── template_fitter.py    # Phase 3: Template-based fitting
├── README.md             # This file
└── output/               # Output directory (created on first run)
```

## Usage

### Phase 1: Validate Shape Independence

First, verify that the M(Λ̄pK⁻) shape is consistent across different M(B+) sideband regions:

```bash
cd analysis/studies/sideband_background
python validate_shapes.py
python validate_shapes.py --no-cuts  # Compare with/without cuts
```

**Output:**
- `output/shape_validation_comparison.pdf` - Overlay of normalized shapes
- `output/shape_validation_ratio.pdf` - Ratio plot
- `output/shape_validation_tests.txt` - KS and χ² test results

### Phase 2: Extract Background Template

Extract the M(Λ̄pK⁻) background template from sideband data:

```bash
python extract_template.py
python extract_template.py --smoothing kde --kde-bandwidth 15
python extract_template.py --smoothing root
python extract_template.py --use-all-sidebands  # Use all M(B+) sidebands
```

**Output:**
- `output/background_template.pdf` - Template visualization
- `output/background_template.root` - Histogram file
- `output/background_template_workspace.root` - RooFit workspace

### Phase 3: Compare Background Models

Compare ARGUS vs Template background on the **same data** with the **same signal model**:

```bash
python compare_background_models.py
python compare_background_models.py --years 2016,2017,2018
```

**Output:**
- `output/background_comparison.pdf` - Side-by-side fit comparison
- `output/background_comparison_results.txt` - Yield comparison and systematic uncertainties

This is the key comparison for systematic uncertainty estimation.

### Alternative: Simple Template Fit

For standalone template fitting (without ARGUS comparison):

```bash
python template_fitter.py
python template_fitter.py --use-smoothed   # Use smoothed template
```

**Output:**
- `output/template_fit_result.pdf` - Fit result plot
- `output/fit_results.txt` - Numerical results

## Configuration

Mass regions are defined in `config.py`:

| Region | M(B+) Range [MeV] | Purpose |
|--------|-------------------|---------|
| Signal | [5229, 5329] | B+ signal region (±50 MeV) |
| Left sideband (far) | [2800, 3500] | Shape validation |
| Left sideband (mid) | [3500, 4500] | Shape validation |
| Left sideband (near) | [4500, 5150] | Template extraction |
| Right sideband | [5330, 5500] | Template extraction |

## Validation Criteria

Shape validation is considered successful if:
- **KS test**: p-value > 0.05 for all region pairs
- **χ² test**: p-value > 0.05 and χ²/ndf ~ 1

If shapes differ significantly across M(B+) regions, consider:
1. Using only near-signal sidebands for template
2. Investigating physics sources of shape variation
3. Using parametric background model instead

## Comparison with ARGUS Model

The template fit can be compared with the standard ARGUS background model:
- Template: Data-driven, captures any non-trivial structure
- ARGUS: Parametric, assumes specific functional form

Differences in signal yields provide a systematic uncertainty estimate.

## Dependencies

- ROOT with RooFit
- Python 3.9+
- Analysis modules from `analysis/modules/`

## Related Scripts

- `analysis/scripts/plot_2d_mlambdapk_vs_mbu.py` - 2D correlation study
- `analysis/scripts/plot_mlpk_projections_signal_sideband.py` - Signal vs sideband comparison
- `analysis/modules/mass_fitter.py` - Main mass fitting module
