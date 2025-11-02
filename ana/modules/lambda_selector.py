class LambdaSelector:
    """
    Apply fixed Lambda reconstruction quality cuts
    These are NOT optimized per charmonium state
    """
    
    def __init__(self, config: TOMLConfig):
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
        mask = ak.ones_like(events.Bu_MM, dtype=bool)
        
        # Lambda mass window
        lambda_mass_branch = self.config.get_branch_name("lambda_mass")
        mask = mask & (events[lambda_mass_branch] > self.cuts["mass_min"])
        mask = mask & (events[lambda_mass_branch] < self.cuts["mass_max"])
        
        # Lambda flight distance χ²
        lambda_fd_branch = self.config.get_branch_name("lambda_fdchi2")
        mask = mask & (events[lambda_fd_branch] > self.cuts["fd_chisq_min"])
        
        # Delta Z (absolute value in mm, not significance!)
        # NOTE: The cut is on |Delta_Z| > 5 mm
        mask = mask & (np.abs(events["Delta_Z_mm"]) > self.cuts["delta_z_min"])
        
        # Proton PID from Lambda decay
        lambda_proton_pid_branch = self.config.get_branch_name("lambda_proton_probnnp")
        mask = mask & (events[lambda_proton_pid_branch] > self.cuts["proton_probnnp_min"])
        
        n_before = len(events)
        n_after = ak.sum(mask)
        print(f"  Lambda selection: {n_before} → {n_after} ({100*n_after/n_before:.1f}%)")
        
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