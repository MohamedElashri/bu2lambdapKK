# Archive

This directory contains superseded, completed, or legacy code that is no longer
part of the active analysis pipeline. It is kept for reproducibility and reference
but should not be modified.

---

## `box_optimization/`

**Status**: Superseded by MVA-based selection
**Frozen**: 2026-03
**Origin**: `analysis/studies/box_optimization/`

The original rectangular (N-dimensional box) cut optimiser that scanned grids of
`Bu_DTF_chi2`, `Bu_FDCHI2`, `Bu_IPCHI2`, `Bu_PT`, and `PID_product` using a
sideband proxy for background estimation. Superseded by the MVA (CatBoost)
selection documented in `analysis/studies/mva_optimization/`. The proxy-based
FOM for PID was later shown to be fundamentally incorrect
(see `analysis/studies/pid_cancellation/` and the memory notes for details);
the fit-based optimizer results recorded in memory are the canonical selection cuts.

Key output: `output/` contains optimisation scans.
Final adopted cuts are in `analysis/config/selection.toml`.

---

## `mva_tmva/`

**Status**: Superseded by CatBoost MVA
**Frozen**: 2026-03
**Origin**: `analysis/studies/mva_optimization/tmva/`

ROOT TMVA-based classifier prototype, written before the project switched to
CatBoost. The TMVA approach required ROOT data format conversion and was harder
to iterate on. CatBoost results in `analysis/studies/mva_optimization/mva/`
and `analysis/studies/mva_optimization/output/` are the ones used in the
final analysis.

---

## Notes on other completed studies

The following studies were conducted and their output directories remain
in `analysis/studies/` because they contain figures referenced in the analysis
note (§4 BackgroundStudies, §7 Systematics):

- `analysis/studies/tracking_systematic/` — 0% tracking syst in ratio
- `analysis/studies/selection_systematic/` — MVA threshold variation
- `analysis/studies/fit_systematics/` — resolution ±2 MeV, ARGUS endpoint variation
- `analysis/studies/kinematic_reweighting/` — 2% kin. reweighting syst
- `analysis/studies/efficiency_steps/` — per-step efficiency tables
- `analysis/studies/pid_cancellation/` — PID syst derivation (PIDCalib2)
- `analysis/studies/ana_note_plots/` — figure generation for the analysis note

The following studies reached definitive conclusions and should be stored in git history. Their directories were removed after the conclusions were
captured:

- `pid_proxy_comparison/` — proved all 4 proxy strategies anti-correlated with fit FOM
- `sideband_pid_validity/` — proved sideband proxy misleading for PID optimisation
- `fit_based_optimizer/` — 252-point fit scan; canonical selection cuts (Set1/Set2)
- `cross_validation_optimizer/` — odd/even split bias check; Set1 robust, Set2 unreliable
