# cc̄ Background Search Study

## Motivation

The B⁺ → Λ̄pK⁻K⁺ analysis reconstructs charmonium resonances in the M(Λ̄pK⁻) spectrum. Before fitting for signal yields, it is essential to understand whether cc̄ resonances (J/ψ, ηc, χc0, χc1, χc2, ψ(2S)) are visible and whether any cc̄ contamination exists from sources **other** than B⁺ decay.

This study separates events by B⁺ mass region:
- **B⁺ signal window** → cc̄ from genuine B⁺ → (cc̄)Λ̄pK⁻K⁺ decays
- **B⁺ sidebands** → cc̄ from prompt production, combinatorial background, or other non-B⁺ sources

## Methodology

### B⁺ mass regions

| Region | Definition |
|--------|-----------|
| Signal window | M(B⁺) ∈ [5255, 5305] MeV/c² |
| Sidebands | M(B⁺) < 5200 or M(B⁺) > 5350 MeV/c² |

### Known cc̄ resonances marked

| Resonance | Mass [MeV/c²] |
|-----------|---------------|
| ηc(1S) | 2983.9 |
| J/ψ | 3096.9 |
| χc0 | 3414.7 |
| χc1 | 3510.7 |
| χc2 | 3556.2 |
| ψ(2S) | 3686.1 |

### Peak significance

A simple sideband-subtraction method estimates local significance S/√B for each resonance, using ±15 MeV signal window and ±50 MeV sidebands for background estimation.

### Outputs

- `output/ccbar_signal_window.pdf` — Full spectrum + zoomed J/ψ region + zoomed left/right regions (B⁺ signal window)
- `output/ccbar_sidebands.pdf` — Same layout for B⁺ sidebands + normalised shape comparison overlay
- `output/ccbar_summary.csv` — Event counts and significance estimates per resonance per region

## Running

```bash
cd analysis/studies/ccbar_background_search
uv run snakemake -j1
```

Requires Step 2 cached data from the main pipeline:
```bash
cd analysis/ && uv run snakemake load_data -j1
```

## Results

*(To be filled after running the study.)*

## Ported from

`archive/analysis/studies/feedback_dec2024/study2_ccbar_background_search.py`

### Changes from original

- **Plotting**: ROOT TCanvas/TH1 → matplotlib + mplhep (LHCb2 style)
- **Layout**: 4 separate canvases → 2 multi-page PDFs (signal window + sidebands), each with full spectrum, zoomed regions, and J/ψ zoom panel
- **New**: Normalised shape comparison overlay (signal vs sideband)
- **New**: Quantitative peak significance estimates (S/√B) per resonance
- **New**: Summary CSV with event counts and significance
- **Data loading**: Direct ROOT file access → cached Step 2 awkward arrays
- **Resonance lines**: Proper matplotlib legend entries with mass values
