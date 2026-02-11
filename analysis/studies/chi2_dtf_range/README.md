# χ²_DTF(B⁺) Distribution Study

## Motivation

The selection cut χ²_DTF(B⁺) < 30 is one of the most selective cuts in the analysis (~70% MC pass rate). The default plotting range [0, 100] wastes space because the distribution is concentrated at low values. This study plots the distribution with a restricted range [0, 35] to better visualise the shape near the cut value, and quantifies MC/Data agreement.

## Methodology

### Procedure

1. Load χ²_DTF(B⁺) from cached Step 2 data (MC per signal state + Data)
2. Restrict to range [0, 35] (cut value is 30)
3. Normalise all distributions to unit area for shape comparison
4. Plot MC (combined + per-state overlaid) vs Data
5. Add ratio panel (Data/MC) below the main plot
6. Compute KS test p-values for all pairwise comparisons

### KS test comparisons

- Data vs MC (combined)
- Data vs each MC signal state
- MC state vs MC state (inter-state consistency)

### Outputs

- `output/chi2_dtf_comparison.pdf` — Multi-page PDF:
  - Page 1: Data vs MC comparison with ratio panel, KS annotation, cut line
  - Page 2: Per-state MC overlay with inter-state KS results
  - Page 3: Summary table of all KS tests
- `output/chi2_dtf_ks_test.csv` — KS test statistics and p-values

## Running

```bash
cd analysis/studies/chi2_dtf_range
uv run snakemake -j1
```

Requires Step 2 cached data from the main pipeline:
```bash
cd analysis/ && uv run snakemake load_data -j1
```

## Results

*(To be filled after running the study.)*

## Ported from

`archive/analysis/studies/feedback_dec2024/study5_chi2_range_fix.py`

### Changes from original

- **Plotting**: ROOT TCanvas/TH1 → matplotlib + mplhep (LHCb2 style)
- **New**: Ratio panel (Data/MC) below the main plot
- **New**: Per-state MC distributions overlaid (original only had combined MC)
- **New**: KS test p-values for Data vs MC and inter-state comparisons
- **New**: Summary CSV with KS test results
- **New**: Summary table page
- **Layout**: Single canvas → multi-page PDF
- **Data loading**: Direct ROOT file access → cached Step 2 awkward arrays
