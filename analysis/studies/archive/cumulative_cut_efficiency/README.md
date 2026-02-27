# Cumulative Cut Efficiency Study

## Motivation

When applying multiple selection cuts to isolate B⁺ → Λ̄pK⁻K⁺ signal, it is important to understand how each successive cut affects the M(Λ̄pK⁻) distribution — the spectrum where charmonium resonances (J/ψ, ηc, χc0, χc1) are reconstructed.

This study visualises the **cumulative** effect of the 7 selection cuts, showing three overlaid categories at each stage:

1. **All events** — baseline distribution before any optimisable cuts (only Lambda pre-selection applied)
2. **Pass** — events surviving all cuts applied up to this stage
3. **Fail** — events rejected by at least one cut up to this stage

## Methodology

### Data sources

- **Signal MC**: J/ψ, ηc, χc0, χc1 samples (after Lambda cuts from Step 2)
- **Data**: Real data in the B⁺ signal region [5255, 5305] MeV/c² (after Lambda cuts)

### Cuts applied cumulatively

| Cut | Variable | Condition |
|-----|----------|-----------|
| 1 | `Bu_DTF_chi2` | < 30 |
| 2 | `Bu_FDCHI2_OWNPV` | > 100 |
| 3 | `Bu_IPCHI2_OWNPV` | < 10 |
| 4 | `Bu_PT` | > 3000 MeV/c |
| 5 | `h1_ProbNNk` | > 0.1 |
| 6 | `h2_ProbNNk` | > 0.1 |
| 7 | `p_ProbNNp` | > 0.1 |

These correspond to the `[manual_cuts]` in `config/selection.toml`.

At stage *k*, cuts 1 through *k* are applied simultaneously. "Pass" means all *k* cuts are satisfied; "Fail" means at least one is not.

### Outputs

- `output/cumulative_cuts_mc.pdf` — Multi-panel MC plots (one panel per cut stage, all states overlaid) + efficiency summary
- `output/cumulative_cuts_data.pdf` — Multi-panel Data plots (one panel per cut stage) + efficiency summary
- `output/cumulative_cuts_summary.csv` — Efficiency table per cut stage, per sample

## Running

```bash
cd analysis/studies/cumulative_cut_efficiency
uv run snakemake -j1
```

Requires Step 2 cached data from the main pipeline:
```bash
cd analysis/ && uv run snakemake load_data -j1
```

## Results

*(To be filled after running the study.)*

## Ported from

`archive/analysis/studies/feedback_dec2024/study1_revised_cumulative_cuts.py`

### Changes from original

- **Plotting**: ROOT TCanvas/TH1 → matplotlib + mplhep (LHCb2 style)
- **Layout**: Individual canvases → multi-panel PDF (one panel per cut stage)
- **Data loading**: Direct ROOT file access → cached Step 2 awkward arrays
- **Output**: Added summary CSV with efficiency numbers
- **Summary panel**: Added efficiency-vs-cut-stage line plot for quick comparison
