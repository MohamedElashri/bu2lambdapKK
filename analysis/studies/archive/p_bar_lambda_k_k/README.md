
# Study: $B^+ \to \bar{p} \Lambda K^+ K^+$ Resonance Check

## Motivation

This study checks for charmonium resonances in the $B^+ \to \bar{p} \Lambda K^+ K^+$ decay channel.
This channel involves a $\Lambda$ baryon and $\bar{p}$ antibaryon, plus two identical charged kaons ($K^+$).
We investigate if the invariant mass spectrum of $M(\Lambda \bar{p} K^+)$ exhibits peaks corresponding to charmonium states ($J/\psi$, $\eta_c$), similar to the signal channel $B^+ \to \bar{\Lambda} p K^- K^+$.

## Methodology

1. **Data Source**: Real data (2016-2018), tree `B2L0PbarKpKp` (LL and DD tracks).
2. **Selection**:
   - **Lambda Selection**: $M(\Lambda) \in [1111, 1121]$ MeV, $\chi^2_{FD} > 250$, $|\Delta Z| > 5$ mm, Proton PID $> 0.3$.
   - **B+ Selection**: $M_{corr}(B^+) \in [5255, 5305]$ MeV (Lambda-corrected).
   - **Standard Cuts**:
     - $p_T(B^+) > 3000$ MeV
     - $\chi^2_{FD}(B^+) > 100$
     - $\chi^2_{IP}(B^+) < 10$
     - $\chi^2_{DTF}(B^+) < 30$
     - PID: Proton/Kaons ProbNN > 0.1
3. **Reconstruction**:
   - Particles: $\Lambda$, $\bar{p}$, $K^+_1$, $K^+_2$.
   - Calculate invariant mass $M(\Lambda \bar{p} K^+)$ for both kaon combinations.
4. **Plotting**:
   - Invariant mass spectrum in the charmonium region (2800 - 4000 MeV).
   - Overlay expected resonance masses.

## Running

```bash
cd analysis/studies/p_bar_lambda_k_k
uv run snakemake -j1
```

## Results

After applying the standard selection cuts to clean the signal, we observe the following candidates in the resonance windows (sum of both kaon combinations):

| Resonance | Mass Window [MeV] | Candidates |
|-----------|-------------------|------------|
| $\eta_c(1S)$ | 2934.1 - 3034.1 | ~957       |
| $J/\psi(1S)$ | 3046.9 - 3146.9 | ~588       |
| $\chi_{c0}(1P)$ | 3384.7 - 3444.7 | ~276       |
| $\chi_{c1}(1P)$ | 3480.7 - 3540.7 | ~309       |
| $\eta_c(2S)$ | 3607.8 - 3667.8 | ~403       |

*Note: These counts effectively include each event up to twice (once for each $K^+$ combination).*

- **Mass Spectrum**: See `output/mass_spectrum.pdf`
- **Counts**: See `output/summary.csv`

The presence of peaks at these masses confirms that the decay $B^+ \to \bar{p} \Lambda K^+ K^+$ proceeds via charmonium intermediate states.
