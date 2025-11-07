"""
Efficiency Calculation Module for B+ -> Lambda pK-K+ Analysis

Implements Phase 6 efficiency calculation (simplified approach).
Following plan.md specification.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import awkward as ak
from pathlib import Path


class EfficiencyCalculator:
    """
    Calculate efficiencies from MC samples (Phase 6)
    
    SIMPLIFIED EFFICIENCY STRATEGY FOR DRAFT ANALYSIS:
    =================================================
    Total efficiency ≈ SELECTION EFFICIENCY ONLY
    
    ε_total ≈ ε_sel = N_pass_all_cuts / N_after_lambda_cuts
    
    Components NOT included (deferred to full analysis):
    - ε_acc: Acceptance (from truth) - assumed ~1.0 (cancels in ratios)
    - ε_reco×strip: Reconstruction + stripping - assumed similar for all states
    - ε_trig: Trigger - assumed similar for all states (OR already applied)
    
    This simplification is justified because:
    1. We measure RATIOS (ε_state / ε_J/ψ), many factors cancel
    2. Similar detector acceptance for all charmonium states
    3. Similar trigger efficiency (all have high-pT tracks)
    4. Focus is on statistical precision first, systematics later
    
    Following plan.md Phase 6 specification exactly.
    """
    
    def __init__(self, config: Any, optimized_cuts_df=None):
        """
        Initialize efficiency calculator
        
        Args:
            config: TOMLConfig with paths and parameters
            optimized_cuts_df: DataFrame from Phase 4 with optimal cuts per state
        """
        self.config = config
        self.optimized_cuts = optimized_cuts_df
    
    def get_cuts_for_state(self, state: str):
        """
        Extract optimized cuts for a specific charmonium state
        
        Args:
            state: "jpsi", "etac", "chic0", "chic1"
            
        Returns:
            DataFrame with cuts for this state only
        """
        if self.optimized_cuts is None:
            raise ValueError("Optimized cuts not provided to EfficiencyCalculator")
        
        return self.optimized_cuts[self.optimized_cuts["state"] == state]
    
    def apply_optimized_cuts(self, mc_events, state: str):
        """
        Apply all optimized cuts from Phase 4 to MC events
        
        Args:
            mc_events: Awkward array (MC events after Lambda cuts)
            state: "jpsi", "etac", "chic0", "chic1"
            
        Returns:
            Filtered awkward array with events passing all cuts
        """
        import awkward as ak
        
        state_cuts = self.get_cuts_for_state(state)
        
        # Get first branch to initialize mask
        first_row = state_cuts.iloc[0]
        first_branch_data = mc_events[first_row["branch_name"]]
        
        # Flatten if jagged
        if 'var' in str(ak.type(first_branch_data)):
            first_branch_data = ak.firsts(first_branch_data)
        
        # Start with all events passing
        mask = ak.ones_like(first_branch_data, dtype=bool)
        
        # Apply each optimized cut
        for _, row in state_cuts.iterrows():
            branch = row["branch_name"]
            cut_val = row["optimal_cut"]
            cut_type = row["cut_type"]
            
            branch_data = mc_events[branch]
            
            # Flatten jagged arrays if needed
            if 'var' in str(ak.type(branch_data)):
                branch_data = ak.firsts(branch_data)
            
            if cut_type == "greater":
                mask = mask & (branch_data > cut_val)
            elif cut_type == "less":
                mask = mask & (branch_data < cut_val)
            else:
                raise ValueError(f"Unknown cut type: {cut_type}")
        
        return mc_events[mask]
    
    def calculate_selection_efficiency(self, mc_events, state: str):
        """
        Calculate SELECTION EFFICIENCY ONLY (Phase 6 simplified approach)
        
        ε_sel = N_pass_all_cuts / N_after_lambda_cuts
        
        This is the key quantity for draft analysis.
        Other efficiency components assumed to cancel in ratios.
        
        Args:
            mc_events: Awkward array with MC events after Lambda pre-selection
            state: "jpsi", "etac", "chic0", "chic1"
            
        Returns:
            (efficiency, statistical_error)
        """
        import awkward as ak
        import numpy as np
        
        n_before = len(mc_events)
        
        # Apply optimized cuts
        mc_after = self.apply_optimized_cuts(mc_events, state)
        n_after = len(mc_after)
        
        # Calculate efficiency
        eff = n_after / n_before if n_before > 0 else 0.0
        
        # Statistical error from binomial distribution
        # For large N: σ_eff ≈ sqrt(eff × (1-eff) / N)
        if n_before > 0:
            error = np.sqrt(eff * (1 - eff) / n_before)
        else:
            error = 0.0
        
        return {
            "eff": eff,
            "err": error,
            "n_before": n_before,
            "n_after": n_after
        }
    
    def calculate_all_efficiencies(self, mc_by_state):
        """
        Calculate selection efficiencies for all (state, year) pairs
        
        Args:
            mc_by_state: {state: {year: awkward_array}}
                        MC events already after Lambda pre-selection
        
        Returns:
            {state: {year: {"eff": value, "err": error, "n_before": N, "n_after": N}}}
        """
        import awkward as ak
        
        efficiencies = {}
        
        print("\n" + "="*80)
        print("PHASE 6: EFFICIENCY CALCULATION")
        print("="*80)
        print("\nDRAFT ANALYSIS STRATEGY:")
        print("  - Calculate SELECTION EFFICIENCY ONLY: ε_sel")
        print("  - Other components (acc, reco, strip, trig) assumed similar")
        print("  - Focus on RATIOS: ε_state / ε_J/ψ (many factors cancel)")
        print("="*80)
        
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            efficiencies[state] = {}
            
            print(f"\n[State: {state}]")
            
            for year in sorted(mc_by_state[state].keys()):
                mc_events = mc_by_state[state][year]
                n_before = len(mc_events)
                
                print(f"  Year {year}: N_after_lambda = {n_before}")
                
                # Calculate efficiency
                eff, err = self.calculate_selection_efficiency(mc_events, state)
                
                # Get number after cuts for validation
                mc_after = self.apply_optimized_cuts(mc_events, state)
                n_after = len(mc_after)
                
                efficiencies[state][year] = {
                    "eff": eff,
                    "err": err,
                    "n_before": n_before,
                    "n_after": n_after
                }
                
                print(f"            N_pass_cuts = {n_after}")
                print(f"            ε_sel = {n_after}/{n_before} = {eff:.4f} ± {err:.4f} ({100*eff:.2f}%)")
        
        return efficiencies
    
    def calculate_efficiency_ratios(self, efficiencies):
        """
        Calculate efficiency RATIOS relative to J/ψ
        
        This is what enters the branching fraction ratio formula!
        R(BR) = [N_state / N_J/ψ] × [ε_J/ψ / ε_state] × [BR_norm]
        
        Many systematic uncertainties cancel in the ratio ε_J/ψ / ε_state.
        
        Args:
            efficiencies: {state: {year: {"eff": value, "err": error, ...}}}
            
        Returns:
            DataFrame with efficiency ratios per year
        """
        import pandas as pd
        import numpy as np
        
        results = []
        
        print("\n" + "="*80)
        print("EFFICIENCY RATIOS (relative to J/ψ)")
        print("="*80)
        print("\nThese ratios enter the BR calculation:")
        print("  R(BR_state / BR_J/ψ) = [N_state / N_J/ψ] × [ε_J/ψ / ε_state]")
        print("="*80)
        
        for state in ["etac", "chic0", "chic1"]:
            for year in sorted(efficiencies["jpsi"].keys()):
                eff_state = efficiencies[state][year]["eff"]
                eff_jpsi = efficiencies["jpsi"][year]["eff"]
                
                err_state = efficiencies[state][year]["err"]
                err_jpsi = efficiencies["jpsi"][year]["err"]
                
                # Ratio: ε_J/ψ / ε_state (note: inverted for BR formula)
                ratio = eff_jpsi / eff_state if eff_state > 0 else 0.0
                
                # Error propagation for ratio R = A/B:
                # σ_R = R × sqrt((σ_A/A)² + (σ_B/B)²)
                rel_err_jpsi = err_jpsi / eff_jpsi if eff_jpsi > 0 else 0.0
                rel_err_state = err_state / eff_state if eff_state > 0 else 0.0
                
                error = ratio * np.sqrt(rel_err_jpsi**2 + rel_err_state**2)
                
                results.append({
                    "state": state,
                    "year": year,
                    "eff_state": eff_state,
                    "eff_jpsi": eff_jpsi,
                    "ratio_eps_jpsi_over_state": ratio,
                    "ratio_error": error
                })
                
                print(f"\n  {state} / {year}:")
                print(f"    ε_{state} = {eff_state:.4f} ± {err_state:.4f}")
                print(f"    ε_J/ψ = {eff_jpsi:.4f} ± {err_jpsi:.4f}")
                print(f"    ε_J/ψ / ε_{state} = {ratio:.3f} ± {error:.3f}")
        
        df = pd.DataFrame(results)
        
        # Save to file
        from pathlib import Path
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_dir / "efficiency_ratios.csv", index=False)
        
        print("\n✓ Saved: " + str(output_dir / "efficiency_ratios.csv"))
        
        return df
    
    def generate_efficiency_table(self, efficiencies):
        """
        Generate efficiency summary table (similar to Table 26 in reference)
        
        Format: States (rows) × Years (columns)
        Shows ε_sel ± error for each combination
        
        Args:
            efficiencies: {state: {year: {"eff": value, "err": error, ...}}}
        """
        import pandas as pd
        from pathlib import Path
        
        rows = []
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            row = {"State": state}
            for year in sorted(efficiencies[state].keys()):
                eff = efficiencies[state][year]["eff"]
                err = efficiencies[state][year]["err"]
                n_before = efficiencies[state][year]["n_before"]
                n_after = efficiencies[state][year]["n_after"]
                
                row[f"{year}_eff"] = f"{eff:.4f} ± {err:.4f}"
                row[f"{year}_N"] = f"{n_after}/{n_before}"
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        df.to_csv(output_dir / "efficiencies.csv", index=False)
        df.to_markdown(output_dir / "efficiencies.md", index=False)
        
        print("\n" + "="*80)
        print("EFFICIENCY TABLE (Selection Efficiency Only)")
        print("="*80)
        print(df.to_string(index=False))
        
        print("\n✓ Saved: " + str(output_dir / "efficiencies.csv"))
        print("✓ Saved: " + str(output_dir / "efficiencies.md"))
        
        return df