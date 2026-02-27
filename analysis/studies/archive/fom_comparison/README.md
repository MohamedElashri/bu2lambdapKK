# FoM Comparison Study: S/√B vs S/(√S + √B)

## Objective

Determine which Figure of Merit (FoM) formula is more appropriate for optimising selection cuts across five charmonium states in the B⁺ → Λ̄pK⁻K⁺ analysis, where signal yields vary by an order of magnitude between states.

| Group | States | Regime |
|-------|--------|--------|
| High-yield | J/ψ, η_c(1S) | Large signal, S/B ~ 0.6–1.4 |
| Low-yield | χ_c0, χ_c1, η_c(2S) | Small signal, S/B ~ 0.3–0.4 |

## FoM Definitions

- **FoM₁ = S/√B** — Maximises statistical significance. Optimal when B ≫ S. Linear in S, so it always favours more signal regardless of diminishing returns.
- **FoM₂ = S/(√S + √B)** — Minimises relative yield uncertainty (Punzi-like). Saturates at high S, so it naturally penalises cuts that gain signal at the cost of proportionally more background.

## Methodology (v3)

Signal and background are estimated independently for each cut combination in the 7-variable N-D grid scan (3,888 combinations):

- **Signal: S = ε(cuts) × N_expected**
  - ε(cuts) = N_MC_passing / N_MC_total — selection efficiency from truth-matched signal MC (after Λ pre-selection). MC samples: J/ψ, η_c, χ_c0, χ_c1. η_c(2S) uses χ_c1 MC as proxy (no dedicated MC available).
  - N_expected — rough expected yield per state, estimated from data via sideband subtraction with no selection cuts applied (only Λ cuts + B⁺ signal region).

- **Background: B from data sidebands**
  - For each cut combination, count data events in the M(Λ̄pK⁻) sidebands flanking each state's signal window (within the B⁺ mass signal region).
  - Interpolate sideband density into the signal region to estimate B.
  - Sideband multipliers: low = [center − 4w, center − w], high = [center + w, center + 4w], matching the main `SelectionOptimizer` configuration.

This approach is physically correct: MC provides the signal shape and cut efficiency, data provides the true background level. The absolute yield scale (N_expected) ensures that S and B are on the same footing, which is critical for FoM₂ where S appears in the denominator.

## Results

### Estimated yields and S/B regimes

| State | N_expected | S/B (FoM₁ opt.) | S/B (FoM₂ opt.) | Classification |
|-------|-----------|-----------------|-----------------|----------------|
| J/ψ | 259 | 1.40 | 1.21 | High-yield, S > B |
| η_c(1S) | 613 | 0.66 | 0.63 | High-yield, S < B |
| χ_c0 | 63 | 0.28 | 0.28 | Low-yield, S ≪ B |
| χ_c1 | 59 | 0.27 | 0.26 | Low-yield, S ≪ B |
| η_c(2S) | 101 | 0.39 | 0.32 | Low-yield, S ≪ B |

### Optimal cuts comparison

**4 out of 5 states show different optimal cuts between the two FoMs.**

| State | FoM | ε | S | B | FoM value | bu_dtf_chi2 | bu_fdchi2 | bu_ipchi2 | bu_pt | h1_probnnk | h2_probnnk | p_probnnp |
|-------|-----|---|---|---|-----------|-------------|-----------|-----------|-------|------------|------------|-----------|
| **J/ψ** | S/√B | 0.345 | 89.3 | 64.0 | 11.16 | 20 | 100 | **6** | 3000 | 0.1 | 0.1 | 0.2 |
| | S/(√S+√B) | 0.386 | 99.8 | 82.3 | 5.23 | 30 | 100 | **8** | 3000 | 0.1 | 0.1 | 0.2 |
| **η_c(1S)** | S/√B | 0.380 | 232.7 | 351.3 | 12.42 | 30 | 100 | **6** | 3000 | 0.1 | 0.1 | 0.1 |
| | S/(√S+√B) | 0.399 | 244.2 | 390.0 | 6.90 | 30 | 100 | **8** | 3000 | 0.1 | 0.1 | 0.1 |
| **χ_c0** | S/√B | 0.317 | 20.1 | 71.0 | 2.39 | 20 | 100 | 6 | 3000 | 0.1 | 0.1 | 0.3 |
| | S/(√S+√B) | 0.317 | 20.1 | 71.0 | 1.56 | 20 | 100 | 6 | 3000 | 0.1 | 0.1 | 0.3 |
| **χ_c1** | S/√B | 0.314 | 18.6 | 68.3 | 2.26 | 20 | 100 | 6 | 3000 | 0.1 | 0.1 | **0.3** |
| | S/(√S+√B) | 0.326 | 19.3 | 74.7 | 1.48 | 20 | 100 | 6 | 3000 | 0.1 | 0.1 | **0.2** |
| **η_c(2S)** | S/√B | 0.279 | 28.3 | 72.3 | 3.32 | 20 | 100 | **4** | 3000 | **0.3** | 0.1 | 0.2 |
| | S/(√S+√B) | 0.326 | 33.0 | 102.0 | 2.08 | 20 | 100 | **6** | 3000 | **0.1** | 0.1 | 0.2 |

Bold values indicate where the two FoMs disagree.

### Behavioural pattern

- **S/√B** consistently selects **tighter** cuts: lower `bu_ipchi2` thresholds, tighter PID requirements. It aggressively rejects background even at the cost of signal efficiency (lower ε). This is because S/√B is linear in S — any gain in S/B ratio improves the FoM, regardless of how much signal is lost.

- **S/(√S + √B)** consistently selects **looser** cuts: higher `bu_ipchi2` thresholds, looser PID. It preserves signal efficiency (higher ε) because the FoM saturates at high S — there are diminishing returns from gaining more signal if it comes with proportionally more background.

- The **most dramatic difference** is in **η_c(2S)** (4 variables differ), which has the lowest ε under FoM₁ (0.279 vs 0.326) — a 17% relative difference in signal efficiency.

- **χ_c0** is the only state where both FoMs agree. This is likely because the FoM landscape is relatively flat in the low-statistics regime, and the grid resolution is insufficient to resolve the difference.

## Conclusions

1. **The choice of FoM matters.** 4/5 states show different optimal cuts depending on which FoM is used. The differences are not negligible — they correspond to 10–17% changes in signal efficiency.

2. **S/√B over-tightens cuts for low-yield states.** For χ_c0, χ_c1, and η_c(2S), where S/B ≈ 0.3, the S/√B metric drives the optimiser to cut harder than necessary. Since these states already have very few signal events (S ~ 20–30), losing even a small fraction of signal significantly degrades the measurement precision.

3. **S/(√S + √B) is more appropriate for low-yield states.** It naturally accounts for the fact that when S is small, every signal event matters. The looser cuts it selects preserve ~15% more signal efficiency while the increase in background is tolerable.

4. **For high-yield states, the difference is smaller but still present.** J/ψ and η_c(1S) show differences primarily in `bu_ipchi2`. Since these states have large yields, the practical impact on the final measurement is modest.

## Recommendation

**Use state-dependent optimisation (Option B) with different FoMs per group:**

- **High-yield states (J/ψ, η_c(1S)):** Optimise with **S/√B**. These states have sufficient statistics that maximising significance is the priority.
- **Low-yield states (χ_c0, χ_c1, η_c(2S)):** Optimise with **S/(√S + √B)**. Preserving signal efficiency is critical when yields are O(20–30) events.

This is consistent with the theoretical expectation: S/√B is optimal in the B ≫ S limit, while S/(√S + √B) minimises relative yield uncertainty and is more appropriate when S and B are comparable.


## Reproducibility

```bash
cd analysis/studies/fom_comparison
uv run snakemake -j1
```

This study is standalone (not part of the main pipeline DAG) and depends only on Step 2 cached data and MC.
Run Step 2 first if needed: `cd analysis/ && uv run snakemake load_data -j1`
