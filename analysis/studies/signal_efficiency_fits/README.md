# Signal Efficiency Fits Study

## Motivation

Simple event counting to measure selection efficiency is biased by background contamination — events in the B⁺ mass peak region include both signal and combinatorial background. By **fitting** the M(B⁺) distribution with a signal + background model, we can extract the true signal yield before and after selection cuts, giving a more robust efficiency estimate.

## Methodology

### Fit model

- **Signal**: Crystal Ball function (Gaussian core + power-law tail) — standard for B meson mass peaks
- **Background**: ARGUS function — standard for B meson combinatorial background with kinematic endpoint

### Procedure

For each sample (MC per signal state + Data):

1. **All events**: Fit M(B⁺) in [5150, 5450] MeV → extract N_signal(all)
2. **Events passing cuts**: Apply all 7 selection cuts, then fit M(B⁺) → extract N_signal(pass)
3. **Efficiency**: ε = N_signal(pass) / N_signal(all)

Error propagation:
```
σ_ε = sqrt((σ_pass/N_all)² + (N_pass·σ_all/N_all²)²)
```

### Selection cuts applied

| Cut | Variable | Condition |
|-----|----------|-----------|
| 1 | `Bu_DTF_chi2` | < 30 |
| 2 | `Bu_FDCHI2_OWNPV` | > 100 |
| 3 | `Bu_IPCHI2_OWNPV` | < 10 |
| 4 | `Bu_PT` | > 3000 MeV/c |
| 5 | `h1_ProbNNk` | > 0.1 |
| 6 | `h2_ProbNNk` | > 0.1 |
| 7 | `p_ProbNNp` | > 0.1 |

### Outputs

- `output/signal_efficiency_fits.pdf` — Multi-page PDF:
  - One page per sample: fit plots (all + pass) with pull distributions
  - Efficiency comparison bar chart
  - Summary table
- `output/signal_efficiency_table.csv` — Efficiency table with yields, errors, χ²/ndf, fit status

## Running

```bash
cd analysis/studies/signal_efficiency_fits
uv run snakemake -j1
```

Requires Step 2 cached data from the main pipeline:
```bash
cd analysis/ && uv run snakemake load_data -j1
```

## Results

*(To be filled after running the study.)*

## Ported from

`archive/analysis/studies/feedback_dec2024/study3_signal_efficiency_fits.py`

### Changes from original

- **Fitting**: RooFit kept (Crystal Ball + ARGUS) — same physics model
- **Visualisation**: ROOT TCanvas → matplotlib + mplhep (LHCb2 style)
- **Pull distributions**: Added below each fit plot
- **Layout**: Individual canvases → multi-page PDF with fit pairs + summary
- **Output**: Text efficiency table → CSV with full fit metadata
- **Error propagation**: Explicit formula for efficiency uncertainty
- **Data loading**: Direct ROOT file access → cached Step 2 awkward arrays
- **Scope**: Now fits each MC state separately (original combined all MC)
