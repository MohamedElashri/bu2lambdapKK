import awkward as ak
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_handler import TOMLConfig

class LambdaSelector:
    """
    Apply fixed Lambda reconstruction quality cuts
    These are NOT optimized per charmonium state
    """
    
    def __init__(self, config: "TOMLConfig"):
        self.config = config
        self.cuts = config.get_lambda_cuts()
        
    def apply_lambda_cuts(self, events: ak.Array) -> ak.Array:
        """
        Apply all Lambda selection cuts
        
        Fixed cuts:
        - Lambda mass: 1111 < M < 1121 MeV
        - Lambda FD χ²: > 250
        - Delta Z: > 5 mm (absolute, not significance!)
        - Proton PID: ProbNNp > 0.3
        
        Returns: Filtered awkward array
        """
        # Use a reference branch that always exists
        if "Bu_MM" not in events.fields:
            if "Bu_M" not in events.fields:
                print("  ⚠️  Cannot apply Lambda cuts: neither Bu_MM nor Bu_M found")
                return events
            mask = ak.ones_like(events["Bu_M"], dtype=bool)
        else:
            mask = ak.ones_like(events["Bu_MM"], dtype=bool)
        
        # Lambda mass window (use L0_MM which should be available after normalization)
        if "L0_MM" in events.fields:
            mask = mask & (events["L0_MM"] > self.cuts["mass_min"])
            mask = mask & (events["L0_MM"] < self.cuts["mass_max"])
        elif "L0_M" in events.fields:
            mask = mask & (events["L0_M"] > self.cuts["mass_min"])
            mask = mask & (events["L0_M"] < self.cuts["mass_max"])
        else:
            print("  ⚠️  WARNING: Lambda mass branch (L0_MM or L0_M) not found - skipping mass cut")
        
        # Lambda flight distance χ²
        if "L0_FDCHI2_OWNPV" in events.fields:
            mask = mask & (events["L0_FDCHI2_OWNPV"] > self.cuts["fd_chisq_min"])
        else:
            print("  ⚠️  WARNING: Lambda FDCHI2 branch (L0_FDCHI2_OWNPV) not found - skipping FDCHI2 cut")
        
        # Delta Z (absolute value in mm, not significance!)
        # NOTE: The cut is on |Delta_Z| > 5 mm
        if "Delta_Z_mm" in events.fields:
            mask = mask & (np.abs(events["Delta_Z_mm"]) > self.cuts["delta_z_min"])
        else:
            print("  ⚠️  WARNING: Delta_Z_mm branch not found - skipping Delta Z cut")
        
        # Proton PID from Lambda decay (Lp is the proton from Lambda)
        if "Lp_ProbNNp" in events.fields:
            mask = mask & (events["Lp_ProbNNp"] > self.cuts["proton_probnnp_min"])
        else:
            print("  ⚠️  WARNING: Proton PID branch (Lp_ProbNNp) not found - skipping PID cut")
        
        n_before = len(events)
        n_after = ak.sum(mask)
        print(f"  Lambda selection: {n_before} → {n_after} ({100*n_after/n_before:.1f}%)")
        
        return events[mask]
    
    def apply_bu_fixed_cuts(self, events: ak.Array) -> ak.Array:
        """
        Apply fixed B+ mass cut (using Lambda-corrected mass)
        
        Fixed cut: 5255 < Bu_MM_corrected < 5305 MeV
        This is a ±25 MeV window around the B+ mass (5279 MeV)
        
        Returns: Filtered awkward array
        """
        bu_fixed_cuts = self.config.get_bu_fixed_cuts()
        
        # Check if Bu_MM_corrected exists (should be computed in derived branches)
        if "Bu_MM_corrected" not in events.fields:
            print("  ⚠️  Bu_MM_corrected not found, cannot apply B+ mass cut")
            return events
        
        mask = (events["Bu_MM_corrected"] > bu_fixed_cuts["mass_corrected_min"]) & \
               (events["Bu_MM_corrected"] < bu_fixed_cuts["mass_corrected_max"])
        
        n_before = len(events)
        n_after = ak.sum(mask)
        print(f"  B+ mass cut: {n_before} → {n_after} ({100*n_after/n_before:.1f}%)")
        
        return events[mask]
    
    def get_lambda_efficiency_from_mc(self, 
                                     mc_events_truth_matched: ak.Array,
                                     mc_events_all: ak.Array) -> float:
        """
        Calculate Lambda selection efficiency from truth-matched MC
        
        Efficiency = N(pass Lambda cuts) / N(in acceptance)
        
        Args:
            mc_events_truth_matched: Truth-matched MC events
            mc_events_all: All MC events after trigger
            
        Returns:
            Efficiency as float
        """
        n_total = len(mc_events_all)
        events_pass = self.apply_lambda_cuts(mc_events_all)
        n_pass = len(events_pass)
        
        efficiency = n_pass / n_total if n_total > 0 else 0.0
        
        print(f"  Lambda efficiency: {n_pass}/{n_total} = {100*efficiency:.2f}%")
        
        return efficiency