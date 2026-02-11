# Individual Cut Efficiency Study

## Motivation

The cumulative cut efficiency study (Phase 1) shows the effect of cuts applied sequentially — each cut builds on the previous ones. This makes it hard to see which **individual** cut has the largest impact, because the order matters.

This study applies each cut **independently** to the full dataset, showing what fraction of events pass or fail that specific cut alone. This reveals which cuts are the most and least selective, regardless of ordering.

## Methodology

### Procedure

For each sample (MC per signal state + Data):

1. Count total events (baseline)
2. For each of the 7 selection cuts:
   - Apply **only that cut** to the full dataset
   - Record pass/fail fractions
3. Apply **all cuts together** (cumulative) as the final entry

### Selection cuts

| # | Cut | Variable | Category |
|---|-----|----------|----------|
| 1 | χ²_DTF(B⁺) < 30 | `Bu_DTF_chi2` | Vertex quality |
| 2 | FDχ²(B⁺) > 100 | `Bu_FDCHI2_OWNPV` | Vertex quality |
| 3 | IPχ²(B⁺) < 10 | `Bu_IPCHI2_OWNPV` | Vertex quality |
| 4 | p_T(B⁺) > 3 GeV | `Bu_PT` | Kinematic |
| 5 | ProbNN_K(K⁺) > 0.1 | `h1_ProbNNk` | PID |
| 6 | ProbNN_K(K⁻) > 0.1 | `h2_ProbNNk` | PID |
| 7 | ProbNN_p(p) > 0.1 | `p_ProbNNp` | PID |

### Outputs

- `output/individual_cut_efficiency.pdf` — Multi-page PDF:
  - Page 1: MC per-state horizontal bar charts (pass/fail, colour-coded by cut category)
  - Page 2: Data horizontal bar chart
  - Page 3: MC vs Data grouped bar chart comparison
  - Page 4: Summary table
- `output/individual_cut_efficiency.csv` — Full efficiency table with per-cut and cumulative results

## Running

```bash
cd analysis/studies/individual_cut_efficiency
uv run snakemake -j1
```

Requires Step 2 cached data from the main pipeline:
```bash
cd analysis/ && uv run snakemake load_data -j1
```

## Results

*(To be filled after running the study.)*

## Ported from

`archive/analysis/studies/feedback_dec2024/study4_individual_cut_efficiency.py`

### Changes from original

- **Plotting**: ROOT THStack/TCanvas → matplotlib + mplhep (LHCb2 style)
- **Layout**: Separate per-state canvases → single multi-page PDF with all states side-by-side
- **New**: Horizontal stacked bar charts with numerical labels
- **New**: Colour-coding by cut category (vertex, kinematic, PID)
- **New**: MC vs Data grouped comparison chart
- **New**: Summary table page
- **Output**: Text table → CSV with full metadata
- **Data loading**: Direct ROOT file access → cached Step 2 awkward arrays
