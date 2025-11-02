import matplotlib.pyplot as plt
import seaborn as sns

class SelectionOptimizer:
    """
    Optimize cuts on B+, bachelor p̄, K+, K- using Figure of Merit
    Perform 2D optimization: (variable × charmonium_state)
    
    Lambda cuts are already applied (pre-selection)
    """
    
    def __init__(self, 
                 signal_mc: Dict[str, Dict[str, ak.Array]],
                 phase_space_mc: Dict[str, ak.Array],
                 data: Dict[str, ak.Array],
                 config: TOMLConfig):
        """
        signal_mc: {state: {year: events_after_lambda_cuts}}
        phase_space_mc: {year: events_after_lambda_cuts} (KpKm non-resonant)
        data: {year: events_after_lambda_cuts}
        """
        self.signal_mc = signal_mc
        self.phase_space_mc = phase_space_mc
        self.data = data
        self.config = config
        
    def compute_fom(self, n_sig: float, n_bkg: float) -> float:
        """
        Figure of Merit: FOM = n_sig / sqrt(n_bkg + n_sig)
        
        Maximizing FOM balances:
        - Signal efficiency (want high n_sig)
        - Background rejection (want low n_bkg)
        """
        if n_sig <= 0:
            return 0.0
        return n_sig / np.sqrt(n_bkg + n_sig)
    
    def define_signal_region(self, state: str) -> Tuple[float, float]:
        """Get mass window for counting signal events in FOM"""
        return self.config.get_signal_region(state)
    
    def define_sideband_regions(self, state: str) -> List[Tuple[float, float]]:
        """
        Define mass sidebands for background estimation
        
        Returns: [(low_min, low_max), (high_min, high_max)]
        """
        center_val = self.config.particles["signal_regions"][state.lower()]["center"]
        window = self.config.particles["signal_regions"][state.lower()]["window"]
        
        # Low sideband: 4 windows below to 1 window below signal
        low_sb = (center_val - 4*window, center_val - window)
        
        # High sideband: 1 window above to 4 windows above signal
        high_sb = (center_val + window, center_val + 4*window)
        
        return [low_sb, high_sb]
    
    def count_events_in_region(self, 
                               events: ak.Array,
                               region: Tuple[float, float]) -> int:
        """Count events in M_LpKm mass window"""
        mask = (events["M_LpKm"] > region[0]) & (events["M_LpKm"] < region[1])
        return ak.sum(mask)
    
    def estimate_background_in_signal_region(self,
                                            data: ak.Array,
                                            state: str) -> float:
        """
        Estimate combinatorial background in signal region from sidebands
        
        Method: Linear interpolation between low and high sidebands
        """
        signal_region = self.define_signal_region(state)
        sidebands = self.define_sideband_regions(state)
        
        n_low_sb = self.count_events_in_region(data, sidebands[0])
        n_high_sb = self.count_events_in_region(data, sidebands[1])
        
        # Widths
        low_width = sidebands[0][1] - sidebands[0][0]
        high_width = sidebands[1][1] - sidebands[1][0]
        signal_width = signal_region[1] - signal_region[0]
        
        # Average background density
        bkg_density = (n_low_sb/low_width + n_high_sb/high_width) / 2.0
        n_bkg_estimate = bkg_density * signal_width
        
        return n_bkg_estimate
    
    def scan_single_variable(self,
                            state: str,
                            variable_name: str,
                            branch_name: str,
                            scan_config: dict) -> pd.DataFrame:
        """
        Scan a single variable and compute FOM at each cut value
        
        Args:
            state: "jpsi", "etac", "chic0", "chic1"
            variable_name: Logical name (e.g., "bu_pt")
            branch_name: Actual branch name in tree
            scan_config: {begin, end, step, cut_type, description}
            
        Returns:
            DataFrame: [cut_value, n_sig, n_bkg, fom]
        """
        # Generate scan points
        cut_values = np.arange(
            scan_config["begin"],
            scan_config["end"] + scan_config["step"],
            scan_config["step"]
        )
        
        # Combine all years for this state
        sig_mc_combined = ak.concatenate([
            self.signal_mc[state][year] for year in self.signal_mc[state].keys()
        ])
        
        data_combined = ak.concatenate([
            self.data[year] for year in self.data.keys()
        ])
        
        results = []
        
        for cut_val in cut_values:
            # Apply cut
            if scan_config["cut_type"] == "greater":
                sig_pass = sig_mc_combined[sig_mc_combined[branch_name] > cut_val]
                data_pass = data_combined[data_combined[branch_name] > cut_val]
            else:  # "less"
                sig_pass = sig_mc_combined[sig_mc_combined[branch_name] < cut_val]
                data_pass = data_combined[data_combined[branch_name] < cut_val]
            
            # Count signal in signal region
            signal_region = self.define_signal_region(state)
            n_sig = self.count_events_in_region(sig_pass, signal_region)
            
            # Estimate background
            n_bkg = self.estimate_background_in_signal_region(data_pass, state)
            
            # Compute FOM
            fom = self.compute_fom(n_sig, n_bkg)
            
            results.append({
                "cut_value": cut_val,
                "n_sig": n_sig,
                "n_bkg": n_bkg,
                "fom": fom
            })
        
        return pd.DataFrame(results)
    
    def optimize_2d_all_variables(self) -> pd.DataFrame:
        """
        Perform 2D optimization: variables × states
        
        Returns:
            DataFrame with optimal cuts for each (variable, state) pair
            
        Columns: [variable, state, optimal_cut, max_fom]
        """
        states = ["jpsi", "etac", "chic0", "chic1"]
        
        # Collect all optimizable variables from config
        categories = ["bu", "bachelor_p", "kplus", "kminus"]
        
        all_results = []
        
        for category in categories:
            opt_config = self.config.get_optimizable_cuts(category)
            
            for var_name, var_config in opt_config.items():
                if var_name == "notes":  # Skip notes section
                    continue
                
                # Get actual branch name
                branch_name = self.config.get_branch_name(f"{category}_{var_name}")
                
                print(f"\n{'='*60}")
                print(f"Optimizing: {category}.{var_name}")
                print(f"Branch: {branch_name}")
                print(f"{'='*60}")
                
                for state in states:
                    print(f"\n  State: {state}")
                    
                    # Scan this variable for this state
                    scan_results = self.scan_single_variable(
                        state, var_name, branch_name, var_config
                    )
                    
                    # Find optimal cut (max FOM)
                    idx_max = scan_results["fom"].idxmax()
                    optimal_row = scan_results.loc[idx_max]
                    
                    print(f"    Optimal cut: {optimal_row['cut_value']:.3f}")
                    print(f"    Max FOM: {optimal_row['fom']:.3f}")
                    print(f"    n_sig: {optimal_row['n_sig']:.0f}, n_bkg: {optimal_row['n_bkg']:.1f}")
                    
                    all_results.append({
                        "category": category,
                        "variable": var_name,
                        "branch_name": branch_name,
                        "state": state,
                        "optimal_cut": optimal_row["cut_value"],
                        "max_fom": optimal_row["fom"],
                        "n_sig_at_optimal": optimal_row["n_sig"],
                        "n_bkg_at_optimal": optimal_row["n_bkg"],
                        "cut_type": var_config["cut_type"]
                    })
                    
                    # Save scan plot
                    self._plot_fom_scan(
                        scan_results, 
                        category, 
                        var_name, 
                        state, 
                        var_config
                    )
        
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)
        results_df.to_csv(output_dir / "optimized_cuts_2d.csv", index=False)
        
        # Generate summary table
        self._generate_optimization_summary(results_df)
        
        return results_df
    
    def _plot_fom_scan(self, 
                       scan_results: pd.DataFrame,
                       category: str,
                       var_name: str,
                       state: str,
                       var_config: dict):
        """
        Plot FOM scan for a single (variable, state) pair
        Similar to Appendix A figures in reference analysis
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Normalize to max value for better visualization
        max_sig = scan_results["n_sig"].max()
        max_bkg = scan_results["n_bkg"].max()
        max_fom = scan_results["fom"].max()
        
        # Plot three curves
        ax.plot(scan_results["cut_value"], 
                scan_results["n_sig"] / max_sig,
                'b-', label='$n_{sig}$ (normalized)', linewidth=2)
        
        ax.plot(scan_results["cut_value"],
                scan_results["n_bkg"] / max_bkg,
                'r-', label='$n_{bkg}$ (normalized)', linewidth=2)
        
        ax.plot(scan_results["cut_value"],
                scan_results["fom"] / max_fom,
                'g-', label='FOM (normalized)', linewidth=2)
        
        # Mark optimal point
        idx_max = scan_results["fom"].idxmax()
        optimal_cut = scan_results.loc[idx_max, "cut_value"]
        
        ax.axvline(optimal_cut, color='k', linestyle='--', 
                   label=f'Optimal: {optimal_cut:.2f}')
        
        ax.set_xlabel(f'{var_config["description"]} ({var_config["cut_type"]})')
        ax.set_ylabel('Ratio to maximum value')
        ax.set_title(f'{category}.{var_name} optimization for {state}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "optimization"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"fom_scan_{category}_{var_name}_{state}.png"
        plt.savefig(plot_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_optimization_summary(self, results_df: pd.DataFrame):
        """
        Generate summary table similar to Table 7 in reference analysis
        
        Shows optimal cuts for each variable across all states
        """
        # Pivot table: variables × states
        pivot = results_df.pivot_table(
            index=['category', 'variable'],
            columns='state',
            values='optimal_cut'
        )
        
        # Add cut type and description
        cut_info = results_df.drop_duplicates(['category', 'variable'])[
            ['category', 'variable', 'cut_type', 'branch_name']
        ].set_index(['category', 'variable'])
        
        summary = pivot.join(cut_info)
        
        # Reorder columns
        col_order = ['branch_name', 'cut_type', 'jpsi', 'etac', 'chic0', 'chic1']
        summary = summary[col_order]
        
        # Save as markdown and CSV
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        
        summary.to_csv(output_dir / "optimized_cuts_summary.csv")
        summary.to_markdown(output_dir / "optimized_cuts_summary.md")
        
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(summary.to_string())
        print("\n✓ Saved to:", output_dir)