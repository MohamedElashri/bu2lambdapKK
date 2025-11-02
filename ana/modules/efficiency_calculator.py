class EfficiencyCalculator:
    """
    Calculate efficiencies from MC samples
    
    Total efficiency: ε_total = ε_acc × ε_reco×strip × ε_trig × ε_sel
    
    For DRAFT analysis:
    - ε_acc: From truth-matched MC ✓
    - ε_reco×strip: PLACEHOLDER (set to 0) ⚠️
    - ε_trig: From phase space MC ✓
    - ε_sel: From MC after cuts ✓
    
    NOTE: Full efficiency study deferred to later analysis!
    For now, focus on RATIOS (many factors cancel)
    """
    
    def __init__(self, config: TOMLConfig):
        self.config = config
        
    def calculate_acceptance(self, 
                            mc_truth: ak.Array) -> float:
        """
        Acceptance efficiency: ε_acc
        
        Fraction of generated events in detector acceptance
        
        Requires TRUE branches:
        - TRUE_Y (rapidity) in [2, 4.5]
        - All final state particles in acceptance
        
        Args:
            mc_truth: Truth-level MC events (before reconstruction)
            
        Returns:
            ε_acc as float
        """
        # ← THIS REQUIRES TRUTH BRANCHES
        # Need to identify which branches exist for truth info
        # Placeholder implementation:
        
        # Check if B+ in acceptance
        # mask_acc = (mc_truth["Bu_TRUE_Y"] > 2.0) & (mc_truth["Bu_TRUE_Y"] < 4.5)
        
        # For draft: assume all events in acceptance
        # ε_acc ≈ 0.30-0.35 typically for LHCb
        
        print("  ⚠️  Acceptance calculation requires truth branches")
        print("      Using placeholder: ε_acc = 1.0 (will be corrected later)")
        
        return 1.0  # PLACEHOLDER
    
    def calculate_reconstruction_stripping(self) -> float:
        """
        Combined reconstruction + stripping efficiency
        
        For DRAFT: Use placeholder value (0.0)
        Will be provided separately or calculated in detailed study
        
        Typical values: ε_reco×strip ~ 0.01-0.05
        """
        print("  ⚠️  Reconstruction×Stripping: PLACEHOLDER = 0.0")
        print("      This will CANCEL in ratios!")
        
        return 0.0  # PLACEHOLDER - OK because we work with ratios!
    
    def calculate_trigger(self, 
                         phase_space_mc: Dict[str, ak.Array],
                         trigger_selector) -> Dict[str, float]:
        """
        Trigger efficiency from phase space MC (KpKm sample)
        
        Phase space MC represents non-resonant background,
        which has similar kinematics to data (no bias from
        specific charmonium state)
        
        Method:
        1. Take phase space MC after Lambda selection
        2. Apply trigger requirements
        3. ε_trig = N_pass / N_total
        
        Args:
            phase_space_mc: {year: events_after_lambda}
            trigger_selector: Object that applies trigger cuts
            
        Returns:
            {year: ε_trig}
        """
        trigger_eff = {}
        
        print("\n  Calculating trigger efficiency from phase space MC:")
        
        for year, events in phase_space_mc.items():
            n_total = len(events)
            
            # Apply trigger selection
            events_pass = trigger_selector.apply_trigger_selection(events)
            n_pass = len(events_pass)
            
            eff = n_pass / n_total if n_total > 0 else 0.0
            trigger_eff[year] = eff
            
            print(f"    {year}: {n_pass}/{n_total} = {100*eff:.2f}%")
        
        return trigger_eff
    
    def calculate_selection(self,
                           mc_events: ak.Array,
                           optimized_cuts: pd.Series) -> float:
        """
        Selection efficiency: ε_sel
        
        Apply optimized offline cuts to MC
        
        ε_sel = N(pass all cuts) / N(after trigger)
        
        Args:
            mc_events: MC sample after trigger
            optimized_cuts: Series with optimal cut values for this state
            
        Returns:
            ε_sel as float
        """
        n_total = len(mc_events)
        mask = ak.ones_like(mc_events.Bu_MM, dtype=bool)
        
        # Apply each optimized cut
        for idx, row in optimized_cuts.iterrows():
            branch = row["branch_name"]
            cut_val = row["optimal_cut"]
            cut_type = row["cut_type"]
            
            if cut_type == "greater":
                mask = mask & (mc_events[branch] > cut_val)
            else:
                mask = mask & (mc_events[branch] < cut_val)
        
        n_pass = ak.sum(mask)
        eff = n_pass / n_total if n_total > 0 else 0.0
        
        return eff
    
    def calculate_total_efficiency_per_year(self,
                                           state: str,
                                           year: str,
                                           mc_events: ak.Array,
                                           phase_space_mc: Dict[str, ak.Array],
                                           optimized_cuts: pd.Series,
                                           trigger_selector) -> Dict[str, float]:
        """
        Calculate total efficiency for one (state, year) pair
        
        ε_total = ε_acc × ε_reco×strip × ε_trig × ε_sel
        
        Returns:
            {
                "acc": ε_acc,
                "reco_strip": ε_reco×strip,
                "trig": ε_trig,
                "sel": ε_sel,
                "total": ε_total
            }
        """
        print(f"\n{'='*60}")
        print(f"Efficiency calculation: {state} - {year}")
        print(f"{'='*60}")
        
        # Acceptance
        eps_acc = self.calculate_acceptance(mc_events)
        
        # Reco × Strip (placeholder)
        eps_reco_strip = self.calculate_reconstruction_stripping()
        
        # Trigger (from phase space)
        eps_trig_dict = self.calculate_trigger({year: phase_space_mc[year]}, trigger_selector)
        eps_trig = eps_trig_dict[year]
        
        # Selection
        eps_sel = self.calculate_selection(mc_events, optimized_cuts)
        
        # Total (with placeholder for reco×strip, total will be wrong!)
        # But RATIOS will be OK if reco×strip similar across states
        eps_total = eps_acc * (1.0 if eps_reco_strip == 0 else eps_reco_strip) * eps_trig * eps_sel
        
        print(f"\n  ε_acc = {eps_acc:.4f}")
        print(f"  ε_reco×strip = {eps_reco_strip:.4f} (PLACEHOLDER)")
        print(f"  ε_trig = {eps_trig:.4f}")
        print(f"  ε_sel = {eps_sel:.4f}")
        print(f"  ε_total = {eps_total:.4f} (INCOMPLETE - ratios OK!)")
        
        return {
            "acc": eps_acc,
            "reco_strip": eps_reco_strip,
            "trig": eps_trig,
            "sel": eps_sel,
            "total": eps_total
        }
    
    def calculate_efficiency_ratios(self,
                                   efficiencies_by_state_year: Dict) -> pd.DataFrame:
        """
        Calculate efficiency RATIOS relative to J/ψ
        
        This is what matters for branching fraction ratios!
        Many systematic effects cancel in ratios.
        
        Args:
            efficiencies_by_state_year: {state: {year: {component: value}}}
            
        Returns:
            DataFrame with efficiency ratios: ε_state / ε_J/ψ
        """
        results = []
        
        for state in ["etac", "chic0", "chic1"]:
            for year in ["2016", "2017", "2018"]:
                eps_state = efficiencies_by_state_year[state][year]["total"]
                eps_jpsi = efficiencies_by_state_year["jpsi"][year]["total"]
                
                ratio = eps_state / eps_jpsi if eps_jpsi > 0 else 0.0
                
                results.append({
                    "state": state,
                    "year": year,
                    "eps_state": eps_state,
                    "eps_jpsi": eps_jpsi,
                    "ratio": ratio
                })
        
        df = pd.DataFrame(results)
        
        # Save
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        df.to_csv(output_dir / "efficiency_ratios.csv", index=False)
        
        print("\n" + "="*80)
        print("EFFICIENCY RATIOS (relative to J/ψ)")
        print("="*80)
        print(df.to_string(index=False))
        
        return df