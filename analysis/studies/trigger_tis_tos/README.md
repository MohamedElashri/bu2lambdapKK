# Trigger TIS/TOS Efficiency Study

This is an active study owned by the top-level analysis workflow. It calculates
the data-driven trigger efficiency using the TIS/TOS overlap method and writes
the correction factors consumed by `studies/efficiency_steps/`.

Because Monte Carlo does not perfectly replicate the detector's trigger response, the trigger efficiency must be corrected using real data. The method relies on the assumption that for events where the rest of the underlying event fired the trigger (TIS: Trigger Independent of Signal), the signal itself is unbiased with respect to the signal-dependent trigger (TOS: Trigger On Signal).

Therefore, the TOS efficiency can be measured as:
$$
\varepsilon_{TOS} = \frac{N_{TIS \text{ and } TOS}}{N_{TIS}}
$$

This script computes this for the normalization channel ($B^+ \to J/\psi K^+$) on both Real Data and Monte Carlo.

## Method
1. Select events passing the basic lambda selection cuts.
2. For real data, subtract the combinatorial background under the $B^+$ mass peak using a simple sideband subtraction method.
3. Count the yield of events passing L0 TIS.
4. Count the yield of events passing L0 TIS **and** HLT TOS.
5. Compute the ratio for both Data and MC.
6. The final scale factor (correction) is $\varepsilon_{data} / \varepsilon_{MC}$.

## Usage

For normal workflow execution, run it from the top-level analysis workflow:

```bash
cd analysis
make study-trigger
```

The study-local Snakefile can still be used for isolated study work:

```bash
cd analysis/studies/trigger_tis_tos
uv run snakemake -j1
```
