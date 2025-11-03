# Calculating Absolute Branching Fraction Products


This document outlines how to convert our measured **branching fraction ratios** (Table 3) into **absolute branching fraction products** using a normalization channel.

### Current Status

**We have:** Ratios relative to J/ψ
```
R_X = [ℬ(B⁺ → X K⁺) × ℬ(X → Λ̄pK⁻)] / [ℬ(B⁺ → J/ψ K⁺) × ℬ(J/ψ → Λ̄pK⁻)]
```

**We want:** Absolute products
```
ℬ(B⁺ → X K⁺) × ℬ(X → Λ̄pK⁻)
```

**Why products?** The charmonium decay BRs to Λ̄pK⁻ are unknown, so we can only measure the product.

---

## Decay Topology

```
B⁺ → X K⁺    (B⁺ produces charmonium X and a K⁺)
     ↓
     X → Λ̄pK⁻  (X decays to Λ̄pK⁻)

Final state: Λ̄pK⁺K⁻
```

Where X ∈ {J/ψ, η_c(1S), χ_c0(1P), χ_c1(1P)}

**Key:** We fit M(Λ̄pK⁻) to see charmonium peaks, confirming X is made from Λ̄pK⁻, not K⁺K⁻.

---

## What We Need

### Already Have (From Our Analysis)

| Input | Value | Source |
|:------|:------|:-------|
| N_X (yields) | Table 2 | phase5_yields.csv |
| ε_X (efficiencies) | Table 4 | efficiencies.csv |
| L (luminosity) | 5.12 fb⁻¹ | config/luminosity.toml |
| R_X (ratios) | Table 3 | branching_fraction_ratios.csv |

### From PDG

| Input | Value | Reference |
|:------|:------|:----------|
| ℬ(B⁺ → J/ψ K⁺) | (1.016 ± 0.033) × 10⁻³ | PDG 2024 |
| ℬ(J/ψ → μ⁺μ⁻) | (5.961 ± 0.033) × 10⁻² | PDG 2024 |

### Need to Measure: Normalization Channel

**Channel:** B⁺ → J/ψ(→μ⁺μ⁻) K⁺ in 2016-2018 data

**Required:**
- N_norm: Yields from M(μμK) fits (~50k events expected)
- ε_norm: Efficiency from MC (~0.10-0.20 typical)

---

## The Formula

### Main Equation

```
ℬ(B⁺ → X K⁺) × ℬ(X → Λ̄pK⁻) = 
    N_X × ε_norm × ℬ(J/ψ→μμ) × ℬ(B⁺→J/ψK⁺)
    ────────────────────────────────────────
    N_norm × ε_X
```

### Calculation Steps

1. **Calculate normalization factor:**
   ```
   R_norm = N_norm / (L × ε_norm × ℬ(J/ψ→μμ))
   ```

2. **Calculate J/ψ product:**
   ```
   ℬ(B⁺ → J/ψ K⁺) × ℬ(J/ψ → Λ̄pK⁻) = [N_jpsi/(L×ε_jpsi)] / R_norm × ℬ(B⁺→J/ψK⁺)
   ```

3. **Use ratios for other states:**
   ```
   ℬ(B⁺ → X K⁺) × ℬ(X → Λ̄pK⁻) = R_X × ℬ(B⁺ → J/ψ K⁺) × ℬ(J/ψ → Λ̄pK⁻)
   ```
   
   Where R_X from Table 3: R_etac = 2.299, R_chic0 = 0.279, R_chic1 = 0.201



---

## Example Calculation

### Inputs

**Our analysis (combined):**
- N_jpsi = 232.2, ε_jpsi = 0.408, L = 5.12 fb⁻¹

**Hypothetical normalization:**
- N_norm = 50,000, ε_norm = 0.15

**PDG:**
- ℬ(B⁺→J/ψK⁺) = 1.016 × 10⁻³, ℬ(J/ψ→μμ) = 5.961 × 10⁻²

### Calculation

```python
# Step 1: Normalization
R_norm = 50000 / (5.12 × 0.15 × 0.05961) = 1.092e6

# Step 2: J/ψ product
BR_jpsi = (232.2/(5.12×0.408)) / 1.092e6 × 1.016e-3 = 1.03e-7

# Step 3: Other states (using Table 3 ratios)
BR_etac  = 1.03e-7 × 2.299 = 2.37e-7
BR_chic0 = 1.03e-7 × 0.279 = 2.88e-8
BR_chic1 = 1.03e-7 × 0.201 = 2.08e-8
```

---

## Uncertainties

### Propagation Formula

```
(δBR/BR)² = (δN_X/N_X)² + (δε_X/ε_X)² + (δN_norm/N_norm)² 
          + (δε_norm/ε_norm)² + (δℬ_norm/ℬ_norm)² + (δℬ_jpsi_mumu/ℬ_jpsi_mumu)²
```

### Typical Contributions

| Source | δ/value | Note |
|:-------|:--------|:-----|
| N_X | 1-50% | J/ψ ~1%, χ states ~30% (statistics) |
| ε_X, ε_norm | 0.5-5% | MC statistics |
| N_norm | 0.5-1% | Large sample |
| ℬ_norm, ℬ_jpsi_mumu | 0.6-3.2% | PDG |

---

## Implementation Steps

1. **Request data:** B⁺ → J/ψ(→μ⁺μ⁻) K⁺ for 2016-2018 + MC

2. **Develop selection:**
   - Adapt main analysis (replace Λ̄p → J/ψ → μ⁺μ⁻)
   - Similar B⁺ and K⁺ cuts
   - J/ψ window: 3.00-3.20 GeV

3. **Measure normalization:**
   - Fit M(μμK) to extract N_norm
   - Calculate ε_norm from MC

4. **Calculate products:**
   - Apply formula to all states
   - Propagate uncertainties

5. **Validate:**
   - Check ratios match Table 3
   - Verify order of magnitude reasonable

---

## Alternatives (If No Normalization Data)

### Option 1: bb̄ Production Cross-Section (Theory-Based)

**Concept:** Calculate total B⁺ production from pp→bb̄ cross-section, then use fitted yields directly.

**Formula:**
```
ℬ(B⁺ → X K⁺) × ℬ(X → Λ̄pK⁻) = N_X / (N_B⁺ × ε_X)

where: N_B⁺ = σ(pp→bb̄) × L × f_u × 2 × A_LHCb
```

**Inputs:**
- σ(pp→bb̄) = 144 ± 29 μb at √s = 13 TeV (LHCb measurement)
- L = 5.12 fb⁻¹ (our integrated luminosity)
- f_u = 0.407 ± 0.006 (fraction of b→B⁺ from PDG)
- A_LHCb ≈ 0.025 (geometric acceptance, η ∈ [2,5])
- Factor of 2: both b and b̄ produce B mesons

**Example Calculation:**
```python
# Total B⁺ produced in LHCb acceptance
sigma_bb = 144e-6  # pb (144 μb)
L = 5.12e6         # pb⁻¹ (5.12 fb⁻¹)
f_u = 0.407
A_LHCb = 0.025
N_Bp = sigma_bb * L * f_u * 2 * A_LHCb
# → N_Bp ≈ 1.5 × 10¹⁰ B⁺ mesons

# Absolute BR for J/ψ
N_jpsi = 232.2
eps_jpsi = 0.408
BR_jpsi = N_jpsi / (N_Bp * eps_jpsi)
# → BR_jpsi ≈ 3.8 × 10⁻⁸
```

**Issues:**
1. **Large theory uncertainty:** σ(pp→bb̄) has ~20% uncertainty
2. **Acceptance correction:** A_LHCb depends on B⁺ momentum spectrum (model-dependent)
3. **Trigger efficiency:** Not all B⁺ are triggered (need trigger-averaged correction)
4. **Selection efficiency:** ε_X includes only reconstruction, not trigger

**Systematic Uncertainties:**
- σ(pp→bb̄): ±20%
- f_u: ±1.5%
- Acceptance: ±5-10%
- Trigger: ±5%
- **Total:** ±25-30%

**When to Use:**
- No normalization channel available
- Order-of-magnitude estimates acceptable
- Combined with published results for cross-checks

**Not Recommended For:** Precision measurements or competitive results (ratios are better!)

---

### Option 2: Published LHCb Results

| Method | Approach | Issue |
|:-------|:---------|:------|
| **Published B→J/ψK** | Use LHCb Run 2 J/ψK⁺ normalization | Selection differences (~5-10%) |
| **Internal control** | Find ϕ→KK in our sample | ℬ(B⁺→Λ̄pϕ) unknown |

**Recommendation:** For publication, use **relative ratios** (Table 3) as primary result. These are model-independent and valuable physics!

---

## Expected Results

| State | ℬ(B⁺ → X K⁺) × ℬ(X → Λ̄pK⁻) |
|:------|:----------------------------|
| J/ψ | ~1.0 × 10⁻⁷ |
| η_c | ~2.4 × 10⁻⁷ |
| χ_c0 | ~2.9 × 10⁻⁸ |
| χ_c1 | ~2.1 × 10⁻⁸ |

---

## Action Checklist

- [x] Yields (Table 2) ✅
- [x] Efficiencies (Table 4) ✅  
- [x] Luminosity ✅
- [x] Ratios (Table 3) ✅
- [x] PDG values ✅
- [ ] N_norm (need to measure) ❌
- [ ] ε_norm (need to measure) ❌

**Next:** Request B⁺ → J/ψ(→μμ) K⁺ data for 2016-2018

---

## References

- PDG 2024: https://pdglive.lbl.gov/
- ℬ(B⁺→J/ψK⁺) = (1.016 ± 0.033) × 10⁻³
- ℬ(J/ψ→μμ) = (5.961 ± 0.033) × 10⁻²
- Our results: `tables/branching_fraction_ratios.csv`, `tables/phase5_yields.csv`
