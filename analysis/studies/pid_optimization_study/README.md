# PID Optimization Study

**Analysis**: B⁺ → Λ̄pK⁻K⁺
**Purpose**: Determine whether and how to include PID variables in the
selection optimization, using three complementary approaches.

---

## 1. Background and Motivation

PID variables (`p_ProbNNp`, `h1_ProbNNk`, `h2_ProbNNk`, and their product
`PID_product = p_ProbNNp × h1_ProbNNk × h2_ProbNNk`) discriminate genuine
particle species from misidentified tracks.  For this decay the combinatorial
background consists predominantly of random track combinations where at least
one track is misidentified, so PID cuts are expected to help.

However, the standard optimization infrastructure uses **sideband events as a
background proxy**.  This creates a systematic problem:

> Sideband events (B⁺ mass sidebands, ARGUS-tail events, sPlot-weighted
> events) are **real**, well-reconstructed B⁺ candidates with genuine particle
> assignments.  Their PID quality is comparable to signal MC and is
> **systematically higher** than the true combinatorial background in the signal
> window.

As a consequence, any proxy-based FOM overestimates background PID efficiency,
and the optimizer concludes that PID cuts degrade the FOM — the opposite of the
truth.

This study was commissioned to:
1. Confirm and quantify the proxy bias for individual PID variables and for
   the product, under the corrected PID tune (MC15TuneV1 for both data and MC).
2. Obtain the **correct** optimal cut values using fit-based FOM (mass fits on data).
3. Evaluate whether including PID variables in the BDT changes classifier
   performance, and whether that change is physically meaningful.

---

## 2. Variable Corrections Applied Before This Study

Two bugs were fixed in the main pipeline before running these scans:

| File | Fix |
|------|-----|
| `modules/clean_data_loader.py` | MC now uses `MC15TuneV1_ProbNNp/k` (was `MC12TuneV4`) — same as data |
| `modules/branches_config.toml` | Alias table updated to reflect MC15TuneV1 for both data and MC |
| `config/selection.toml` | Removed stale `"pid_product"` from `seq_step2_vars` (no corresponding `[nd_optimizable_selection.pid_product]` existed) |

The MC12→MC15 fix affects PID distributions in the MC signal sample.  It does
**not** affect the fit-based FOM results, which are derived from mass fits on
data alone.

---

## 3. Study Structure

```
pid_optimization_study/
├── config.toml               Study-level configuration (grids, windows, features)
├── run_all.py                 Orchestration entry point
├── scripts/
│   ├── box_scan_proxy.py      Sub-study 1: proxy-based 1-D PID scans
│   ├── fit_based_scan.py      Sub-study 2: fit-based 1-D PID scans (authoritative)
│   ├── mva_pid_study.py       Sub-study 3: MVA with three feature-set variants
│   └── compare_results.py    Sub-study 4: cross-method comparison and recommendation
└── output/
    ├── box_proxy/             PDF plots + JSON results from sub-study 1
    ├── fit_based/             PDF plots + JSON results from sub-study 2
    ├── mva/                   PDF plots + JSON results from sub-study 3
    └── comparison/            Comparison tables (Markdown) + summary PDFs
```

---

## 4. Sub-Study Descriptions

### 4.1 Proxy-Based Box Scan (`box_scan_proxy.py`)

**What it does**:
Performs a 1-D cut scan over each PID variable separately.  At each grid
point, signal efficiency (ε_sig) is measured from MC and background
efficiency (ε_bkg) is measured from the B⁺ mass sideband events.

**FOM definitions**:
- FOM1 = ε_sig / √ε_bkg  (efficiency ratio)
- FOM2 = S_proxy / √B_proxy  (proxy significance)

**Variables scanned**: `PID_product`, `p_ProbNNp`, `h1_ProbNNk`, `h2_ProbNNk`
**Grid**: 0.00 → 0.80 in steps of 0.05

**Expected result**: FOM peaks at cut ≈ 0 for all variables.
**Interpretation**: The sideband proxy overestimates ε_bkg → optimizer always
prefers no cut.  This result is **expected and known to be wrong**; it
reconfirms the proxy bias and motivates the fit-based approach.

**Outputs** (`output/box_proxy/`):
- `proxy_scan_results_{LL,DD}.json`  — cut, ε_sig, ε_bkg, FOM1, FOM2 arrays
- `proxy_scan_{var}_{cat}.pdf`       — 4-panel plot per variable per category

---

### 4.2 Fit-Based Box Scan (`fit_based_scan.py`)

**What it does**:
For each grid point applies the PID cut to data, then runs a full RooFit
simultaneous mass fit (same fitter used in the main pipeline) to extract
signal yields N_J/ψ, N_ηc, N_χc0, N_χc1, and background N_bkg.

**FOM definitions**:
- FOM1 = (N_J/ψ + N_ηc) / √N_bkg  — high-yield group (S/√B)
- FOM2 = (N_χc0 + N_χc1) / √(N_χc0 + N_χc1 + N_bkg)  — low-yield group (S/√(S+B))

**Variables scanned**: same as above
**Runtime**: ~10 s per variable, ~40 s total (binned fit, all years merged)

**Expected result**: FOM1 peaks at a non-zero cut value (confirmed for
`PID_product` > 0.20 in the prior `fit_based_optimizer` study).
**Interpretation**: This is the **authoritative result**.  The optimal cut
from this scan should be used as the PID working point in the analysis.

**Outputs** (`output/fit_based/`):
- `fit_scan_results_{LL,DD}.json`  — per-cut yields and FOMs
- `fit_scan_{var}_{cat}.pdf`       — 4-panel plot per variable per category

---

### 4.3 MVA with PID Variants (`mva_pid_study.py`)

**What it does**:
Trains an XGBoost BDT under three feature configurations:

| Variant | Features |
|---------|----------|
| A — Baseline | Bu_DTF_chi2, Bu_FDCHI2_OWNPV, Bu_IPCHI2_OWNPV, Bu_PT |
| B — Individual PID | same as A + p_ProbNNp, h1_ProbNNk, h2_ProbNNk |
| C — PID product | same as A + PID_product |

For each variant: ROC curve, overtraining check (KS test), feature
importance, and proxy FOM scan over BDT threshold.

**Known limitation of Variants B and C**:
The training background is the B⁺ mass sideband.  Sideband events are
genuine, well-identified particles with PID quality comparable to or better
than signal MC.  The BDT cannot exploit PID to separate signal from the
true combinatorial background, because the training background has the
wrong PID distribution.  A higher AUC for B/C vs A therefore **does not
imply** that PID inclusion improves analysis sensitivity.

**Outputs** (`output/mva/`):
- `mva_pid_summary_{LL,DD}.json`         — AUC and FOM summary
- `roc_comparison_{cat}.pdf`             — all three ROC curves overlaid
- `overtraining_{variant}_{cat}.pdf`     — train/test KS check per variant
- `feature_importance_{variant}_{cat}.pdf` — XGBoost feature importance
- `fom_scan_{variant}_{cat}.pdf`         — proxy FOM scan over BDT threshold
- `xgboost_{variant}_{cat}.json`         — saved model

---

### 4.4 Comparison and Summary (`compare_results.py`)

**What it does**:
Loads all JSON outputs and produces:
- Side-by-side comparison table of optimal cuts (proxy vs fit-based)
- FOM correlation plot (proxy FOM vs fit-based FOM) to quantify bias
- MVA ROC comparison plot
- Recommendation document

**Outputs** (`output/comparison/`):
- `comparison_table_{cat}.md`     — optimal cuts from all methods
- `fom_correlation_{cat}.pdf`     — proxy FOM vs fit FOM with Pearson r
- `mva_roc_comparison_{cat}.pdf`  — ROC curves for A, B, C overlaid
- `recommendation_{cat}.md`       — structured recommendation document

---

## 5. How to Run

**Prerequisites**: Pipeline step2 cache must be current.
```bash
cd analysis/
uv run snakemake load_data -j1   # ~4 min, only needed if cache is stale
```

**Run everything**:
```bash
cd analysis/studies/pid_optimization_study/
uv run python run_all.py
```

**Run individual steps**:
```bash
uv run python run_all.py --steps proxy           # proxy scan only
uv run python run_all.py --steps fit             # fit-based scan only (requires ROOT)
uv run python run_all.py --steps mva             # MVA variants only
uv run python run_all.py --steps compare         # comparison only (needs JSON outputs)
uv run python run_all.py --steps proxy fit compare --category LL   # LL only, skip MVA
```

**Run scripts directly** (from the study root directory):
```bash
uv run python scripts/box_scan_proxy.py  --category both
uv run python scripts/fit_based_scan.py  --category both --variable all
uv run python scripts/mva_pid_study.py   --category both
uv run python scripts/compare_results.py --category both
```

**The fit-based scan** requires ROOT with RooFit.  Run in the environment where
the main pipeline fits work:
```bash
uv run python scripts/fit_based_scan.py --variable pid_product  # ~10 s
```

---

## 6. Interpreting Results

### Proxy scan output
All proxy optimal cuts will be at or near 0.  This is expected.  Do not
use these cuts for the analysis.  The proxy result documents the bias and
allows for direct comparison with the fit-based result.

### Fit-based scan output
The FOM vs cut plot will show a clear peak at a non-zero cut value.  This
is the optimal working point.  Key questions to answer from these plots:
- Does the FOM increase monotonically up to the peak, or is there a plateau?
- Is the product FOM sharper (more localized) than the individual variable FOMs?
- Do LL and DD categories prefer the same cut value?

### MVA output
Compare ROC AUC across variants A, B, C:
- If B/C AUC ≫ A: the BDT has learned a PID boundary, but this boundary
  applies to the sideband (proxy) background, not the true combinatorial
  background.  The BDT may perform worse on data in the signal window.
- If B/C AUC ≈ A: the BDT's PID features are not contributing (as expected
  when signal and background have similar PID distributions in training data).

Check feature importance for variants B and C: if PID features have high
importance, it indicates the BDT is exploiting the spurious PID separation
between signal MC and sideband background.

### Cross-method comparison
The `fom_correlation_{cat}.pdf` plots show proxy FOM vs fit-based FOM as
a function of cut value.  If the curves are anti-correlated (Pearson r < 0),
the proxy bias is confirmed.  The magnitude of |r| indicates how strongly
the proxy is biased.

---

## 7. Expected Conclusions

Based on prior studies (`pid_proxy_comparison`, `fit_based_optimizer`,
`pid_cancellation`) the expected outcomes of this study are:

| Finding | Expected value |
|---------|---------------|
| Proxy optimal cut (all variables) | ≈ 0.00 |
| Fit-based optimal cut (PID_product) | ≈ 0.20 (matches prior result) |
| Fit-based optimal cut (individual vars) | 0.20–0.40 depending on variable |
| Proxy vs fit FOM Pearson r | < −0.6 (anti-correlation) |
| MVA AUC: Variant A (baseline) | established baseline |
| MVA AUC: Variant B/C vs A | small difference, driven by proxy bias |

If the corrected MC15TuneV1 tune significantly changes the signal PID
distribution relative to the prior study's MC12TuneV4, the fit-based
optimal cut may shift.  The fit-based scan will capture this correctly.

---

## 8. Files and Branches Used

| Branch | Description |
|--------|-------------|
| `p_ProbNNp` | Proton PID probability (bachelor proton), MC15TuneV1 |
| `h1_ProbNNk` | Kaon PID probability (K⁺), MC15TuneV1 |
| `h2_ProbNNk` | Kaon PID probability (K⁻), MC15TuneV1 |
| `PID_product` | Derived: `p_ProbNNp × h1_ProbNNk × h2_ProbNNk` |
| `Bu_MM_corrected` | B⁺ mass corrected for Λ mass |
| `M_LpKm_h2` | Invariant mass M(Λ̄pK⁻), h2 = K⁻ |

PID tune: MC15TuneV1 used for **both** data and MC (fixed in commit f181be7).
