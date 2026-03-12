# Efficiency Steps Calculation

This standalone study calculates the efficiency for each step in the analysis pipeline.

The steps are defined as:

1. $\varepsilon_{gen}$: The generator level efficiencies are determined by counting the number of particles that fall into the standard LHCb acceptance criterion of pseudorapidity 2 < $\eta$ < 5 and comparing it to the total number generated. (Currently a placeholder 0.22 since this comes from the MC production report, not the NTuple directly).
2. $\varepsilon_{reco+str}$: The reconstruction and stripping efficiencies are determined by ensuring that the B+ daughter proton and kaons satisfy the criteria for being reconstructed as long tracks, while the $\Lambda^0$ daughters are reconstructed as either downstream (DD) or long tracks (LL). All daughter particles, as well as the B+ meson, are required to match the truth information.
3. $\varepsilon_{trig}$: The trigger efficiencies are defined as the ratio of events that pass the trigger requirements to those from stripping.
4. $\varepsilon_{presel}$: We applied some pre-selection cuts when we prepare our ntuple data so we need to assess it with respect to triggered and stripped events.

The final efficiency for each mode is $\varepsilon_{total} = \varepsilon_{gen} \times \varepsilon_{reco+str} \times \varepsilon_{trig} \times \varepsilon_{presel}$. The script outputs this as a formatted markdown table separated by state and $\Lambda^0$ category (LL/DD).

## Usage

Run the study using Snakemake:

```bash
cd analysis/studies/efficiency_steps
uv run snakemake -j1
```

The output will be saved to `output/efficiencies.json` and printed to the standard output as Markdown tables.
