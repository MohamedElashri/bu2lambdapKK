# Efficiency Steps Calculation

This is an active study owned by the top-level analysis workflow. It calculates
the efficiency for each step in the analysis pipeline and writes the study-level
JSON consumed by the main pipeline's `efficiency_calculation.py`.

The steps are defined as:

1. $\varepsilon_{gen}$: The generator level efficiencies are determined by counting the number of particles that fall into the standard LHCb acceptance criterion of pseudorapidity 2 < $\eta$ < 5 and comparing it to the total number generated. (Currently a placeholder 0.22 since this comes from the MC production report, not the NTuple directly).
2. $\varepsilon_{reco+str}$: The reconstruction and stripping efficiencies are determined by ensuring that the B+ daughter proton and kaons satisfy the criteria for being reconstructed as long tracks, while the $\Lambda^0$ daughters are reconstructed as either downstream (DD) or long tracks (LL). All daughter particles, as well as the B+ meson, are required to match the truth information.
3. $\varepsilon_{trig}$: The trigger efficiencies are defined as the ratio of events that pass the trigger requirements to those from stripping.
4. $\varepsilon_{presel}$: We applied some pre-selection cuts when we prepare our ntuple data so we need to assess it with respect to triggered and stripped events.

The final efficiency for each mode is
$\varepsilon_{total} = \varepsilon_{gen} \times \varepsilon_{reco+str}
\times \varepsilon_{trig} \times \varepsilon_{presel} \times \varepsilon_{mva}$.
The script outputs this as JSON plus formatted markdown tables separated by
state and $\Lambda^0$ category (LL/DD).

## Role In The Active Workflow

- authoritative orchestration lives in the top-level `analysis/Snakefile`
- branch-specific outputs are written as
  `generated/output/studies/efficiency_steps/efficiencies_<branch>.json`
- those outputs feed the main pipeline efficiency step

## Usage

For normal workflow execution, run it from the top-level analysis workflow:

```bash
cd analysis
make study-efficiency
```

The study-local Snakefile can still be used for isolated study work:

```bash
cd analysis/studies/efficiency_steps
uv run snakemake -j1
```

The active branch-specific JSON outputs are written under
`generated/output/studies/efficiency_steps/`, and markdown tables are written
next to the JSON output.
