# Split Charmonium State Optimization Study

## Objective
Optimize selection cuts across two groups of charmonium states using two different FOM formulas based on their differing signal-to-background regimes:
- **Set 1 (High-yield):** $J/\psi$ and $\eta_c(1S)$ - metric: $S/\sqrt{B}$
- **Set 2 (Low-yield):** $\chi_{c0}$, $\chi_{c1}$, and $\eta_c(2S)$ - metric: $S/(\sqrt{S} + \sqrt{B})$

## Methodology
Data is stitched at 3300 MeV: events below 3300 MeV get Set 1 cuts, while events above 3300 MeV get Set 2 cuts.

## Output
- `output/split_opt_cut_table.csv`: The optimized cuts for Set 1 and Set 2.
- `output/fits/mass_fit_combined_stitched.pdf`: Mass fit on the combined stitched data.
