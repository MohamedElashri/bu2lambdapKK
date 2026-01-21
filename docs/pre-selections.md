## Pre-selections

This document describes all selection stages applied to the data, from initial trigger decisions through to the final analysis selections. There are 4 distinct selection stages:

## 1. Trigger Level Pre-selection

The trigger selection requires events to pass all three trigger levels with the following logic:

**L0_TIS (Trigger Independent of Signal)** - At least one of the following must fire:
- `Bu_L0GlobalDecision_TIS`
- `Bu_L0PhysDecision_TIS`
- `Bu_L0HadronDecision_TIS`
- `Bu_L0MuonDecision_TIS`
- `Bu_L0MuonHighDecision_TIS`
- `Bu_L0DiMuonDecision_TIS`
- `Bu_L0PhotonDecision_TIS`
- `Bu_L0ElectronDecision_TIS`

**HLT1_TOS (Trigger On Signal)** - At least one of the following must fire:
- `Bu_Hlt1TrackMVADecision_TOS`
- `Bu_Hlt1TwoTrackMVADecision_TOS`

**HLT2_TOS (Trigger On Signal)** - At least one of the following must fire:
- `Bu_Hlt2Topo2BodyDecision_TOS`
- `Bu_Hlt2Topo3BodyDecision_TOS`
- `Bu_Hlt2Topo4BodyDecision_TOS`

**Combined Trigger Logic**: `L0_TIS AND HLT1_TOS AND HLT2_TOS`

All three levels must pass (with at least one line firing per level).

---

## 2. Restripping Pre-selection

| Name | Explanation | Value | Unit |
| --- | --- | --- | --- |
| `nbody` | Number of bodies in the decay | 4 | Dimensionless |
| `MinBMass` | Minimum invariant mass of the B+ candidate | 4500.0 | MeV/c^2 |
| `MaxBMass` | Maximum invariant mass of the B+ candidate | 7000.0 | MeV/c^2 |
| `MinBPt` | Minimum transverse momentum of the B+ candidate | 1200.0 | MeV/c |
| `MaxBVertChi2DOF` | Maximum chi-squared per degree of freedom of the B+ candidate vertex | 15.0 | Dimensionless |
| `MinBPVVDChi2` | Minimum flight chi-squared of the B+ candidate | 20.0 | Dimensionless |
| `MaxBPVIPChi2` | Maximum impact parameter chi-squared of the B+ candidate | 12.0 | Dimensionless |
| `MinBPVDIRA` | Minimum direction angle of the B+ candidate | 0.999 | Dimensionless |
| `MaxMass` | Maximum mass of the decay products | 6000.0 | MeV/c^2 |
| `doPi` | Whether to include pions in the decay | True | Boolean |
| `doK` | Whether to include kaons in the decay | True | Boolean |
| `prescale` | Prescale factor for the trigger | 1.0 | Dimensionless |
| `MaxTrLong` | Maximum longitudinal track length | 250 | mm |
| `MinPiPt` | Minimum transverse momentum of the pions | 300.0 | MeV/c |
| `MinPiIPChi2DV` | Minimum impact parameter chi-squared of the pions | 5.0 | Dimensionless |
| `MaxPiChi2` | Maximum chi-squared of the pion tracks | 3.0 | Dimensionless |
| `MinPiPIDK` | Minimum PIDK of the pions | -2.0 | Dimensionless |
| `MinPiPIDp` | Minimum PIDp of the pions | -2.0 | Dimensionless |
| `MaxPiGHP` | Maximum ghost probability of the pions | 0.25 | Dimensionless |
| `MinKPt` | Minimum transverse momentum of the kaons | 200.0 | MeV/c |
| `MinKIPChi2DV` | Minimum impact parameter chi-squared of the kaons | 5.0 | Dimensionless |
| `MaxKChi2` | Maximum chi-squared of the kaon tracks | 3.0 | Dimensionless |
| `MinKPIDPi` | Minimum PIDPi of the kaons | -2.0 | Dimensionless |
| `MinKPIDp` | Minimum PIDp of the kaons | -2.0 | Dimensionless |
| `MaxKGHP` | Maximum ghost probability of the kaons | 0.3 | Dimensionless |
| `MinpPt` | Minimum transverse momentum of the protons | 250.0 | MeV/c |
| `MinpIPChi2DV` | Minimum impact parameter chi-squared of the protons | 5.0 | Dimensionless |
| `MaxpChi2` | Maximum chi-squared of the proton tracks | 3.0 | Dimensionless |
| `MinpPIDPi` | Minimum PIDPi of the protons | -2.0 | Dimensionless |
| `MinpPIDK` | Minimum PIDK of the protons | -2.0 | Dimensionless |
| `MaxpGHP` | Maximum ghost probability of the protons | 0.3 | Dimensionless |
| `MaxLmDeltaM` | Maximum mass difference for the Lambda0 candidates | 18.0 | MeV/c^2 |
| `MinLmPt` | Minimum transverse momentum of the Lambda0 candidates | 400.0 | MeV/c |
| `MaxLmVertChi2DOF` | Maximum chi-squared per degree of freedom of the Lambda0 vertex | 15.0 | Dimensionless |
| `MinLmPVVDChi2` | Minimum flight chi-squared of the Lambda0 candidates | 12.0 | Dimensionless |
| `MinLmIPChi2` | Minimum impact parameter chi-squared of the Lambda0 candidates | 0.0 | Dimensionless |
| `MinLmPrtPt` | Minimum transverse momentum of the protons in the Lambda0 decay | 300.0 | MeV/c |
| `MinLmPiPt` | Minimum transverse momentum of the pions in the Lambda0 decay | 100.0 | MeV/c |
| `MinLmPrtPIDPi` | Minimum PIDPi of the protons in the Lambda0 decay | -3.0 | Dimensionless |
| `MinLmPrtIPChi2` | Minimum impact parameter chi-squared of the protons in the Lambda0 decay | 4.0 | Dimensionless |
| `MinLmPiIPChi2` | Minimum impact parameter chi-squared of the pions in the Lambda0 decay | 4.0 | Dimensionless |
| `MaxLmPrtTrkChi2` | Maximum chi-squared of the proton tracks in the Lambda0 decay | 4.0 | Dimensionless |
| `MaxLmPiTrkChi2` | Maximum chi-squared of the pion tracks in the Lambda0 decay | 4.0 | Dimensionless |

---

## 3. Data Reduction Pre-selection

During the data reduction and preparation stage (converting restripped data to analysis-ready reduced files), the following additional selections are applied:

| Name | Explanation | Value | Unit |
| --- | --- | --- | --- |
| `Bu_FDCHI2_OWNPV` | Flight distance chi-squared of the B+ candidate | > 175 | Dimensionless |
| `Delta_Z` | Difference in Z-coordinate between the end vertices of the Lambda0 and B+ particles | > 2.5 | mm |
| `Lp_ProbNNp` | Probability of the Lambda0 proton being a proton according to the neural network | > 0.05 | Dimensionless |
| `p_ProbNNp` | Probability of the bachelor proton being a proton according to the neural network | > 0.05 | Dimensionless |
| `prodProbKK` | Product of the probabilities of the two kaons being kaons (h1_ProbNNk × h2_ProbNNk) | > 0.10 | Dimensionless |
| `Bu_DTF_chi2` | Chi-squared of the decay tree fit for the B+ candidate | < 30 | Dimensionless |
| `Bu_PT` | Transverse momentum of the B+ candidate | > 3000 | MeV/c |

**Note**: These cuts are applied during the data processing stage (see `B2L0barPKpKm_processing.py`). The data reduction also includes branch selection, keeping only relevant variables for the analysis.

---

## 4. Analysis (Offline) Pre-selection

These are the current analysis-level selections applied during the final analysis stage. Note that these have been updated from the original document:

### Fixed Lambda Selection (Applied to All States)

| Name | Explanation | Value | Unit |
| --- | --- | --- | --- |
| `L0_MM` | Invariant mass of the Lambda0 particle | 1111.0 - 1121.0 | MeV/c² |
| `Lp_ProbNNp` | Lambda proton PID probability | > 0.3 | Dimensionless |
| `Delta_Z` | Vertex separation between Lambda0 and B+ | > 5.0 | mm |
| `L0_FDCHI2_OWNPV` | Lambda flight distance chi-squared | > 250.0 | Dimensionless |

### Fixed B+ Selection (Applied to All States)

| Name | Explanation | Value | Unit |
| --- | --- | --- | --- |
| `Bu_MM` (corrected) | B+ invariant mass (Lambda-corrected) | 5255.0 - 5305.0 | MeV/c² |

### Manual/Baseline Cuts (Unoptimized)

These cuts can be used as baseline selections when not running optimization:

| Name | Explanation | Value | Unit |
| --- | --- | --- | --- |
| `Bu_DTF_chi2` | B+ decay tree fit χ² | < 30 | Dimensionless |
| `Bu_FDCHI2_OWNPV` | B+ flight distance χ² | > 100 | Dimensionless |
| `Bu_IPCHI2_OWNPV` | B+ impact parameter χ² | < 10 | Dimensionless |
| `Bu_PT` | B+ transverse momentum | > 3000 | MeV/c |
| `h1_ProbNNk` | First kaon PID probability | > 0.1 | Dimensionless |
| `h2_ProbNNk` | Second kaon PID probability | > 0.1 | Dimensionless |
| `p_ProbNNp` | Bachelor proton PID probability | > 0.1 | Dimensionless |

### Optimizable Selections

The analysis also includes various optimizable selections (for FOM optimization) covering:
- B+ kinematic variables (`FDCHI2`, `PT`, `DTF_chi2`, `IPCHI2`)
- Bachelor proton properties (`IPCHI2`, `ProbNNp`, track quality)
- Kaon properties (`PIDK`, `IPCHI2`, track quality)
- Lambda decay topology (`Delta_Z`, `L0_FDCHI2`)

These optimizable selections have ranges and steps defined in `analysis/config/selection.toml` and are subject to optimization during the analysis.

**Note**: Some analysis selections (especially manual cuts) overlap with data reduction selections but may have different thresholds. The data reduction selections are applied first to reduce file size, and the analysis selections provide the final event selection.

---

## Summary of Selection Flow

1. **Trigger Selection**: Events must pass trigger requirements (`L0_TIS` AND `HLT1_TOS` AND `HLT2_TOS`)
2. **Restripping**: Loose pre-selections at the stripping level
3. **Data Reduction**: Intermediate selections to reduce file size (`Bu_FDCHI2 > 175`, `Delta_Z > 2.5`, etc.)
4. **Analysis**: Final selections including Lambda mass window, optimizable cuts, and mass constraint



## Final Effective Cuts Summary

This table lists all variables with selection cuts applied across any stage, showing the **most restrictive (final effective) value** that determines the actual cut in the analysis. For variables with cuts at multiple stages, the strictest value is shown.

| Branch Name | Final Effective Cut |
| --- | --- |
| **B+ Selections** | |
| `Bu_MM` (corrected) | 5255.0 - 5305.0 MeV/c² |
| `Bu_PT` | > 3000 MeV/c |
| `Bu_FDCHI2_OWNPV` | > 175 (Data Reduction) → > 100 (Analysis baseline) |
| `Bu_IPCHI2_OWNPV` | < 10 |
| `Bu_DTF_chi2` | < 30 |
| **Lambda Selections** | |
| `L0_MM` | 1111.0 - 1121.0 MeV/c² |
| `L0_FDCHI2_OWNPV` | > 250.0 |
| `Delta_Z` | > 5.0 mm |
| **Lambda Proton** | |
| `Lp_ProbNNp` | > 0.3 |
| **Bachelor Proton** | |
| `p_ProbNNp` | > 0.1 |
| **Kaons** | |
| `h1_ProbNNk` | > 0.1 |
| `h2_ProbNNk` | > 0.1 |
| `prodProbKK` (h1 × h2) | > 0.10 |
| **Trigger Requirements** | |
| Trigger Logic | `L0_TIS` AND `HLT1_TOS` AND `HLT2_TOS` |

**Notes:**
- `Bu_FDCHI2_OWNPV`: Data reduction applies > 175, but analysis baseline uses > 100 (the data reduction cut of > 175 is already applied, so effectively > 175)
- `prodProbKK` is calculated as h1_ProbNNk × h2_ProbNNk
- All trigger lines are OR'd within each level (L0, HLT1, HLT2), but all three levels must pass
- Optimizable cuts may vary depending on the optimization strategy used in the analysis


---

## Variable Definitions (Branches)
| Name | Meaning | Dimension |
| --- | --- | --- |
| `h1_P` | Total momentum of the first hadron | MeV/c |
| `h1_PT` | Transverse momentum of the first hadron | MeV/c |
| `h1_PE` | Energy of the first hadron | MeV |
| `h1_PX` | X-component of the momentum of the first hadron | MeV/c |
| `h1_PY` | Y-component of the momentum of the first hadron | MeV/c |
| `h1_PZ` | Z-component of the momentum of the first hadron | MeV/c |
| `h1_ID` | Particle ID of the first hadron | Dimensionless |
| `h1_PIDK` | Probability of the first hadron being a kaon according to the PID algorithm | Dimensionless |
| `h1_PIDp` | Probability of the first hadron being a proton according to the PID algorithm | Dimensionless |
| `h1_TRACK_Type` | Type of the track of the first hadron (LL or DD) | Dimensionless |
| `h1_IPCHI2_OWNPV` | Impact parameter chi-squared of the first hadron | Dimensionless |
| `h2_P` | Total momentum of the second hadron | MeV/c |
| `h2_PT` | Transverse momentum of the second hadron | MeV/c |
| `h2_PE` | Energy of the second hadron | MeV |
| `h2_PX` | X-component of the momentum of the second hadron | MeV/c |
| `h2_PY` | Y-component of the momentum of the second hadron | MeV/c |
| `h2_PZ` | Z-component of the momentum of the second hadron | MeV/c |
| `h2_ID` | Particle ID of the second hadron | Dimensionless |
| `h2_PIDK` | Probability of the second hadron being a kaon according to the PID algorithm | Dimensionless |
| `h2_PIDp` | Probability of the second hadron being a proton according to the PID algorithm | Dimensionless |
| `h2_TRACK_Type` | Type of the track of the second hadron (LL or DD) | Dimensionless |
| `h2_IPCHI2_OWNPV` | Impact parameter chi-squared of the second hadron | Dimensionless |
| `p_P` | Total momentum of the proton | MeV/c |
| `p_PT` | Transverse momentum of the proton | MeV/c |
| `p_PE` | Energy of the proton | MeV |
| `p_PX` | X-component of the momentum of the proton | MeV/c |
| `p_PY` | Y-component of the momentum of the proton | MeV/c |
| `p_PZ` | Z-component of the momentum of the proton | MeV/c |
| `p_ID` | Particle ID of the proton | Dimensionless |
| `p_PIDK` | Probability of the proton being a kaon according to the PID algorithm | Dimensionless |
| `p_PIDp` | Probability of the proton being a proton according to the PID algorithm | Dimensionless |
| `p_TRACK_Type` | Type of the track of the proton (LL or DD) | Dimensionless |
| `p_ProbNNp` | Probability of the proton being a proton according to the neural network | Dimensionless |
| `p_IPCHI2_OWNPV` | Impact parameter chi-squared of the proton | Dimensionless |
| `Lp_P` | Total momentum of the Lambda0 proton | MeV/c |
| `Lp_PT` | Transverse momentum of the Lambda0 proton | MeV/c |
| `Lp_PE` | Energy of the Lambda0 proton | MeV |
| `Lp_PX` | X-component of the momentum of the Lambda0 proton | MeV/c |
| `Lp_PY` | Y-component of the momentum of the Lambda0 proton | MeV/c |
| `Lp_PZ` | Z-component of the momentum of the Lambda0 proton | MeV/c |
| `Lp_ID` | Particle ID of the Lambda0 proton | Dimensionless |
| `Lp_TRACK_Type` | Type of the track of the Lambda0 proton (LL or DD) | Dimensionless |
| `Lp_ProbNNp` | Probability of the Lambda0 proton being a proton according to the neural network | Dimensionless |
| `LL` | Boolean mask for events where the track type of the Lambda0 proton is Long (LL) | Boolean |
| `DD` | Boolean mask for events where the track type of the Lambda0 proton is Downstream (DD) | Boolean |
| `Lpi_P` | Total momentum of the Lambda0 pion | MeV/c |
| `Lpi_PT` | Transverse momentum of the Lambda0 pion | MeV/c |
| `Lpi_PE` | Energy of the Lambda0 pion | MeV |
| `Lpi_PX` | X-component of the momentum of the Lambda0 pion | MeV/c |
| `Lpi_PY` | Y-component of the momentum of the Lambda0 pion | MeV/c |
| `Lpi_PZ` | Z-component of the momentum of the Lambda0 pion | MeV/c |
| `Lpi_ID` | Particle ID of the Lambda0 pion | Dimensionless |
| `Lpi_TRACK_Type` | Type of the track of the Lambda0 pion (LL or DD) | Dimensionless |
| `Lpi_ProbNNpi` | Probability of the Lambda0 pion being a pion according to the neural network | Dimensionless |
| `L0_P` | Total momentum of the Lambda0 particle | MeV/c |
| `L0_PT` | Transverse momentum of the Lambda0 particle | MeV/c |
| `L0_PE` | Energy of the Lambda0 particle | MeV |
| `L0_PX` | X-component of the momentum of the Lambda0 particle | MeV/c |
| `L0_PY` | Y-component of the momentum of the Lambda0 particle | MeV/c |
| `L0_PZ` | Z-component of the momentum of the Lambda0 particle | MeV/c |
| `L0_ID` | Particle ID of the Lambda0 particle | Dimensionless |
| `L0_MM` | Invariant mass of the Lambda0 particle | MeV/c^2 |
| `Bu_FDCHI2_OWNPV` | Flight distance chi-squared of the B+ particle | Dimensionless |
| `L0_ENDVERTEX_X` | X-coordinate of the end vertex of the Lambda0 particle | mm |
| `L0_ENDVERTEX_Y` | Y-coordinate of the end vertex of the Lambda0 particle | mm |
| `L0_ENDVERTEX_Z` | Z-coordinate of the end vertex of the Lambda0 particle | mm |
| `L0_ENDVERTEX_XERR` | Uncertainty on the X-coordinate of the end vertex of the Lambda0 particle | mm |
| `L0_ENDVERTEX_YERR` | Uncertainty on the Y-coordinate of the end vertex of the Lambda0 particle | mm |
| `L0_ENDVERTEX_ZERR` | Uncertainty on the Z-coordinate of the end vertex of the Lambda0 particle | mm |
| `L0_OWNPV_Z` | Z-coordinate of the Lambda0 particle's own primary vertex | mm |
| `L0_OWNPV_ZERR` | Uncertainty on the Z-coordinate of the Lambda0 particle's own primary vertex | mm |
| `L0_FD_OWNPV` | Flight distance of the Lambda0 particle from its own primary vertex | mm |
| `L0_FDCHI2_OWNPV` | Flight distance chi-squared of the Lambda0 particle from its own primary vertex | Dimensionless |
| `Bu_ENDVERTEX_X` | X-coordinate of the end vertex of the B+ particle | mm |
| `Bu_ENDVERTEX_Y` | Y-coordinate of the end vertex of the B+ particle | mm |
| `Bu_ENDVERTEX_Z` | Z-coordinate of the end vertex of the B+ particle | mm |
| `Bu_ENDVERTEX_XERR` | Uncertainty on the X-coordinate of the end vertex of the B+ particle | mm |
| `Bu_ENDVERTEX_YERR` | Uncertainty on the Y-coordinate of the end vertex of the B+ particle | mm |
| `Bu_ENDVERTEX_ZERR` | Uncertainty on the Z-coordinate of the end vertex of the B+ particle | mm |
| `Bu_IPCHI2_OWNPV` | Impact parameter chi-squared of the B+ particle | Dimensionless |
| `Bu_MM` | Invariant mass of the B+ particle | MeV/c^2 |
| `Bu_MMERR` | Uncertainty on the invariant mass of the B+ particle | MeV/c^2 |
| `Bu_ID` | Particle ID of the B+ particle | Dimensionless |
| `Bu_P` | Total momentum of the B+ particle | MeV/c |
| `Bu_PT` | Transverse momentum of the B+ particle | MeV/c |
| `Bu_PE` | Energy of the B+ particle | MeV |
| `Bu_PX` | X-component of the momentum of the B+ particle | MeV/c |
| `Bu_PY` | Y-component of the momentum of the B+ particle | MeV/c |
| `Bu_PZ` | Z-component of the momentum of the B+ particle | MeV/c |
| `Delta_Z` | Difference in Z-coordinate between the end vertices of the Lambda0 and B+ particles | mm |
| `Delta_X` | Difference in X-coordinate between the end vertices of the Lambda0 and B+ particles | mm |
| `Delta_Y` | Difference in Y-coordinate between the end vertices of the Lambda0 and B+ particles | mm |
| `Delta_X_ERR` | Uncertainty on the difference in X-coordinate between the end vertices of the Lambda0 and B+ particles | mm |
| `Delta_Y_ERR` | Uncertainty on the difference in Y-coordinate between the end vertices of the Lambda0 and B+ particles | mm |
| `Delta_Z_ERR` | Uncertainty on the difference in Z-coordinate between the end vertices of the Lambda0 and B+ particles | mm |
| `delta_x` | Normalized difference in X-coordinate between the end vertices of the Lambda0 and B+ particles | Dimensionless |
| `delta_y` | Normalized difference in Y-coordinate between the end vertices of the Lambda0 and B+ particles | Dimensionless |
| `delta_z` | Normalized difference in Z-coordinate between the end vertices of the Lambda0 and B+ particles | Dimensionless |
| `L0_FD_CHISQ` | Chi-squared of the flight distance of the Lambda0 particle | Dimensionless |
| `Bu_DTF_decayLength` | Decay length of the B+ particle from the DecayTreeFitter | mm |
| `Bu_DTF_decayLengthErr` | Uncertainty on the decay length of the B+ particle from the DecayTreeFitter | mm |
| `Bu_DTF_ctau` | c*tau of the B+ particle from the DecayTreeFitter | mm |
| `Bu_DTF_ctauErr` | Uncertainty on the c*tau of the B+ particle from the DecayTreeFitter | mm |
| `Bu_DTF_status` | Status of the DecayTreeFitter for the B+ particle | Dimensionless |
| `Bu_DTF_nPV` | Number of primary vertices in the DecayTreeFitter for the B+ particle | Dimensionless |
| `Bu_DTF_chi2` | Chi-squared of the DecayTreeFitter for the B+ particle | Dimensionless |
| `Bu_DTF_nDOF` | Number of degrees of freedom in the DecayTreeFitter for the B+ particle | Dimensionless |
| `Bu_DTFL0_M` | Invariant mass of the Lambda0 particle from the DecayTreeFitter | MeV/c^2 |
| `Bu_DTFL0_MERR` | Uncertainty on the invariant mass of the Lambda0 particle from the DecayTreeFitter | MeV/c^2 |
| `Bu_DTFL0_ctau` | c*tau of the Lambda0 particle from the DecayTreeFitter | mm |
| `Bu_DTFL0_ctauErr` | Uncertainty on the c*tau of the Lambda0 particle from the DecayTreeFitter | mm |
| `Bu_DTFL0_chi2` | Chi-squared of the DecayTreeFitter for the Lambda0 particle | Dimensionless |
| `Bu_DTFL0_nDOF` | Number of degrees of freedom in the DecayTreeFitter for the Lambda0 particle | Dimensionless |
| `Bu_DTFL0_status` | Status of the DecayTreeFitter for the Lambda0 particle | Dimensionless |
| `Bu_L0Global_TIS` | L0 global trigger decision for the B+ particle (Trigger Independent of Signal) | Boolean |
| `Bu_L0HadronDecision_TOS` | L0 hadron trigger decision for the B+ particle (Trigger On Signal) | Boolean |
| `Bu_Hlt1Global_TIS` | Hlt1 global trigger decision for the B+ particle (Trigger Independent of Signal) | Boolean |
| `Bu_Hlt1TrackMVADecision_TOS` | Hlt1 track MVA trigger decision for the B+ particle (Trigger On Signal) | Boolean |
| `Bu_Hlt1TwoTrackMVADecision_TOS` | Hlt1 two-track MVA trigger decision for the B+ particle (Trigger On Signal) | Boolean |
| `Bu_Hlt2Topo2BodyDecision_TOS` | Hlt2 two-body topology trigger decision for the B+ particle (Trigger On Signal) | Boolean |
| `Bu_Hlt2Topo3BodyDecision_TOS` | Hlt2 three-body topology trigger decision for the B+ particle (Trigger On Signal) | Boolean |
| `Bu_Hlt2Topo4BodyDecision_TOS` | Hlt2 four-body topology trigger decision for the B+ particle (Trigger On Signal) | Boolean |
| `Bu_Hlt2Topo2BodyBBDTDecision_TOS` | Hlt2 two-body BBDT topology trigger decision for the B+ particle (Trigger On Signal) | Boolean |
| `Bu_Hlt2Topo3BodyBBDTDecision_TOS` | Hlt2 three-body BBDT topology trigger decision for the B+ particle (Trigger On Signal) | Boolean |
| `Bu_Hlt2Topo4BodyBBDTDecision_TOS` | Hlt2 four-body BBDT topology trigger decision for the B+ particle (Trigger On Signal) | Boolean |

---
