from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .exceptions import OptimizationError


class SelectionOptimizer:
    """
    Optimize cuts using UNBIASED data-driven method.
    NEW STRATEGY (Phase 3 revision):
    - Signal proxy: "no-charmonium" data (M(Λ̄pK⁻) > 4 GeV)
    - Background proxy: B⁺ mass sidebands
    - NO MC USED in optimization → unbiased!

    Supports two modes:
    - Option A (universal): Same cuts for all states
    - Option B (state-specific): Different cuts per state

    Attributes:
        data: Real data events by year (after Lambda cuts)
        config: Configuration object
    """

    def __init__(
        self,
        data: dict[str, ak.Array],
        config: Any,
    ) -> None:
        """
        Initialize selection optimizer with data-driven approach.

        Args:
            data: {year: events_after_lambda_cuts} - Real data only!
            config: Configuration object
        """
        self.data: dict[str, ak.Array] = data
        self.config: Any = config

        # Get optimization strategy from config
        self.opt_config = self.config.selection.get("optimization_strategy", {})
        self.state_dependent = self.opt_config.get("state_dependent", False)

        # Print optimization mode
        print("\n" + "=" * 80)
        print("SELECTION OPTIMIZER - UNBIASED DATA-DRIVEN METHOD")
        print("=" * 80)
        if self.state_dependent:
            print("Mode: Option B (STATE-SPECIFIC cuts)")
            print("  → Each state gets its own optimized cuts")
        else:
            print("Mode: Option A (UNIVERSAL cuts)")
            print("  → Same cuts applied to all states")
        print("=" * 80)

    def get_no_charmonium_data(self, data: ak.Array) -> ak.Array:
        """
        Get "no-charmonium" events as signal proxy.

        These are real B⁺ → Λ̄pK⁻K⁺ events with M(Λ̄pK⁻) above charmonium limit.
        Same final state and similar kinematics as signal, but UNBIASED.

        Args:
            data: Awkward array of events

        Returns:
            Filtered array with M(Λ̄pK⁻) > charmonium_limit
        """
        mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in data.fields else "M_LpKm"
        no_cc_min = self.opt_config.get("no_charmonium_mass_min", 4000.0)
        no_cc_max = self.opt_config.get("no_charmonium_mass_max", 6000.0)

        mask = (data[mass_branch] > no_cc_min) & (data[mass_branch] < no_cc_max)
        return data[mask]

    def get_b_mass_sideband_data(self, data: ak.Array) -> ak.Array:
        """
        Get B⁺ mass sideband events as background proxy.

        Events with:
        - M(Λ̄pK⁻) in charmonium region [2.9-3.8 GeV]
        - M(B⁺) in far sidebands (below or above signal window)

        Args:
            data: Awkward array of events

        Returns:
            Filtered array in B⁺ mass sidebands
        """
        mass_LpKm = data["M_LpKm_h2"] if "M_LpKm_h2" in data.fields else data["M_LpKm"]
        mass_Bu = data["Bu_MM_corrected"] if "Bu_MM_corrected" in data.fields else data["Bu_M"]

        # Charmonium region
        cc_min = self.opt_config.get("charmonium_region_min", 2900.0)
        cc_max = self.opt_config.get("charmonium_region_max", 3800.0)
        in_charmonium = (mass_LpKm > cc_min) & (mass_LpKm < cc_max)

        # B+ sidebands
        b_low_sb_min = self.opt_config.get("b_low_sideband_min", 5150.0)
        b_low_sb_max = self.opt_config.get("b_low_sideband_max", 5230.0)
        b_high_sb_min = self.opt_config.get("b_high_sideband_min", 5330.0)
        b_high_sb_max = self.opt_config.get("b_high_sideband_max", 5410.0)

        in_low_sb = (mass_Bu > b_low_sb_min) & (mass_Bu < b_low_sb_max)
        in_high_sb = (mass_Bu > b_high_sb_min) & (mass_Bu < b_high_sb_max)
        in_sidebands = in_low_sb | in_high_sb

        return data[in_charmonium & in_sidebands]

    def validate_data_regions(self) -> dict[str, int]:
        """
        Validate that data regions have sufficient statistics.

        Returns:
            Dictionary with event counts in each region
        """
        # Combine all years
        data_arrays = [self.data[year] for year in self.data.keys()]
        data_combined = ak.concatenate(data_arrays, axis=0)

        # Count events in each region
        no_cc_data = self.get_no_charmonium_data(data_combined)
        bkg_data = self.get_b_mass_sideband_data(data_combined)

        counts = {
            "total": len(data_combined),
            "no_charmonium": len(no_cc_data),
            "b_sidebands": len(bkg_data),
        }

        print("\n" + "=" * 80)
        print("DATA REGION VALIDATION")
        print("=" * 80)
        print(f"Total events (after Lambda cuts): {counts['total']:,}")
        print(f"No-charmonium region (signal proxy): {counts['no_charmonium']:,}")
        print(f"B+ sidebands (background proxy): {counts['b_sidebands']:,}")

        if counts["no_charmonium"] < 100:
            print("⚠️  WARNING: Low statistics in no-charmonium region!")
        if counts["b_sidebands"] < 100:
            print("⚠️  WARNING: Low statistics in B+ sidebands!")

        print("=" * 80)

        return counts

    def compute_fom(self, n_sig: float, n_bkg: float) -> float:
        """
        Figure of Merit: FOM = n_sig / sqrt(n_bkg + n_sig)

        Maximizing FOM balances:
        - Signal efficiency (want high n_sig)
        - Background rejection (want low n_bkg)

        Args:
            n_sig: Number of signal events
            n_bkg: Number of background events

        Returns:
            FOM value (higher is better), 0.0 if invalid
        """
        if n_bkg + n_sig <= 0:
            return 0.0
        return n_sig / np.sqrt(n_bkg + n_sig)

    def define_signal_region(self, state: str) -> tuple[float, float]:
        """Get mass window for counting signal events in FOM"""
        return self.config.get_signal_region(state)

    def define_sideband_regions(self, state: str) -> list[tuple[float, float]]:
        """
        Define mass sidebands for background estimation

        Returns: [(low_min, low_max), (high_min, high_max)]
        """
        center_val = self.config.particles["signal_regions"][state.lower()]["center"]
        window = self.config.particles["signal_regions"][state.lower()]["window"]

        # Get sideband multipliers from config (with defaults for backward compatibility)
        opt_config = self.config.selection.get("optimization_strategy", {})
        sb_low_mult = opt_config.get("sideband_low_multiplier", 4.0)
        sb_low_end_mult = opt_config.get("sideband_low_end_multiplier", 1.0)
        sb_high_start_mult = opt_config.get("sideband_high_start_multiplier", 1.0)
        sb_high_mult = opt_config.get("sideband_high_multiplier", 4.0)

        # Low sideband: (center - sb_low_mult*window) to (center - sb_low_end_mult*window)
        low_sb = (center_val - sb_low_mult * window, center_val - sb_low_end_mult * window)

        # High sideband: (center + sb_high_start_mult*window) to (center + sb_high_mult*window)
        high_sb = (center_val + sb_high_start_mult * window, center_val + sb_high_mult * window)

        return [low_sb, high_sb]

    def define_optimization_mass_region(self, state: str) -> tuple[float, float]:
        """
        Define a broader mass region for optimization

        This region includes signal + both sidebands, so we only optimize
        cuts using events that are actually relevant to this state's mass region.

        This prevents contamination from other states' mass regions during optimization.

        Returns: (low_mass, high_mass) for filtering data
        """
        center_val = self.config.particles["signal_regions"][state.lower()]["center"]
        window = self.config.particles["signal_regions"][state.lower()]["window"]

        # Use range from 5 windows below to 5 windows above center
        # This includes both sidebands plus signal region plus some margin
        low_mass = center_val - 5 * window
        high_mass = center_val + 5 * window

        return (low_mass, high_mass)

    def count_events_in_region(self, events: ak.Array, region: tuple[float, float]) -> int:
        """
        Count events in mass region.

        Uses M_LpKm_h2 (Lambdabar-p-Kminus system mass).

        Args:
            events: Awkward array of events
            region: Tuple of (low_mass, high_mass) in MeV

        Returns:
            Number of events in region
        """
        # Use M_LpKm_h2 (h2 = K- from charmonium, h1 = K+)
        mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in events.fields else "M_LpKm"
        mask = (events[mass_branch] > region[0]) & (events[mass_branch] < region[1])
        return ak.sum(mask)

    def estimate_background_in_signal_region(self, data_events: ak.Array, state: str) -> float:
        """
        Estimate combinatorial background in signal region from real data sidebands

        Uses sideband interpolation to estimate background under the peak.
        This gives the true background level in data for cut optimization.

        Method:
        1. Count events in low sideband [center - 4*window, center - window]
        2. Count events in high sideband [center + window, center + 4*window]
        3. Average the two sidebands
        4. Scale by width ratio: background_in_signal = avg_sideband * (signal_width / sideband_width)

        Args:
            data_events: Real data events after cuts
            state: Charmonium state ("jpsi", "etac", "chic0", "chic1") - Note: etac_2s uses chi_c1 cuts

        Returns:
            float: Estimated background in signal region
        """
        signal_region = self.define_signal_region(state)
        sidebands = self.define_sideband_regions(state)

        # Count events in each sideband
        n_low_sb = self.count_events_in_region(data_events, sidebands[0])
        n_high_sb = self.count_events_in_region(data_events, sidebands[1])

        # Calculate sideband widths
        low_sb_width = sidebands[0][1] - sidebands[0][0]
        high_sb_width = sidebands[1][1] - sidebands[1][0]
        signal_width = signal_region[1] - signal_region[0]

        # Average density from both sidebands (events per MeV)
        density_low = n_low_sb / low_sb_width if low_sb_width > 0 else 0
        density_high = n_high_sb / high_sb_width if high_sb_width > 0 else 0
        avg_density = (density_low + density_high) / 2.0

        # Estimate background in signal region
        n_bkg_estimate = avg_density * signal_width

        return n_bkg_estimate

    def _get_branch_name_for_variable(self, category: str, var_name: str) -> str:
        """
        Map (category, variable) to actual branch name in normalized data.

        This mapping is based on branch naming conventions after normalization.
        """
        # Mapping for each category
        # NOTE: h1 is K+, h2 is K- (confirmed from PDG ID analysis)
        branch_map = {
            "bu": {
                "pt": "Bu_PT",
                "dtf_chi2": "Bu_DTF_chi2",
                "ipchi2": "Bu_IPCHI2_OWNPV",
                "fdchi2": "Bu_FDCHI2_OWNPV",
            },
            "bachelor_p": {
                "probnnp": "p_ProbNNp",
                "track_chi2ndof": "p_TRACK_CHI2NDOF",
                "ipchi2": "p_IPCHI2_OWNPV",
            },
            "kplus": {
                "probnnk": "h1_ProbNNk",  # h1 is K+ (confirmed)
                "track_chi2ndof": "h1_TRACK_CHI2NDOF",
                "ipchi2": "h1_IPCHI2_OWNPV",
            },
            "kminus": {
                "probnnk": "h2_ProbNNk",  # h2 is K- (confirmed - used for charmonium)
                "track_chi2ndof": "h2_TRACK_CHI2NDOF",
                "ipchi2": "h2_IPCHI2_OWNPV",
            },
        }

        if category in branch_map and var_name in branch_map[category]:
            return branch_map[category][var_name]
        # Fallback: construct from category_var_name
        return f"{category}_{var_name}"

    def _plot_fom_scan(
        self, scan_results: pd.DataFrame, category: str, var_name: str, state: str, var_config: dict
    ) -> None:
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
        ax.plot(
            scan_results["cut_value"],
            scan_results["n_sig"] / max_sig,
            "b-",
            label="$n_{sig}$ (normalized)",
            linewidth=2,
        )

        ax.plot(
            scan_results["cut_value"],
            scan_results["n_bkg"] / max_bkg,
            "r-",
            label="$n_{bkg}$ (normalized)",
            linewidth=2,
        )

        ax.plot(
            scan_results["cut_value"],
            scan_results["fom"] / max_fom,
            "g-",
            label="FOM (normalized)",
            linewidth=2,
        )

        # Mark optimal point
        idx_max = scan_results["fom"].idxmax()
        optimal_cut = scan_results.loc[idx_max, "cut_value"]

        ax.axvline(optimal_cut, color="k", linestyle="--", label=f"Optimal: {optimal_cut:.2f}")

        ax.set_xlabel(f'{var_config["description"]} ({var_config["cut_type"]})')
        ax.set_ylabel("Ratio to maximum value")
        ax.set_title(f"{category}.{var_name} optimization for {state}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "optimization"
        plot_dir.mkdir(exist_ok=True, parents=True)

        filename = f"fom_scan_{category}_{var_name}_{state}.pdf"
        plt.savefig(plot_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_2d_fom_heatmap(
        self, scan_2d_results: pd.DataFrame, var1: dict[str, Any], var2: dict[str, Any], state: str
    ) -> None:
        """
        Plot 2D heatmap of FOM as function of two variables

        Similar to correlation plots in appendices of reference analysis
        Shows optimal point in 2D variable space
        """
        # Pivot to create 2D grid
        fom_grid = scan_2d_results.pivot(index="cut2", columns="cut1", values="fom")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.imshow(
            fom_grid.values,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=(
                fom_grid.columns.min(),
                fom_grid.columns.max(),
                fom_grid.index.min(),
                fom_grid.index.max(),
            ),
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Figure of Merit (FOM)", rotation=270, labelpad=20)

        # Mark optimal point
        idx_max = scan_2d_results["fom"].idxmax()
        optimal = scan_2d_results.loc[idx_max]
        ax.plot(
            optimal["cut1"],
            optimal["cut2"],
            "r*",
            markersize=20,
            markeredgecolor="white",
            markeredgewidth=2,
            label=f'Optimal: FOM={optimal["fom"]:.2f}',
        )

        # Labels
        ax.set_xlabel(f'{var1["category"]}.{var1["var_name"]} ({var1["config"]["cut_type"]})')
        ax.set_ylabel(f'{var2["category"]}.{var2["var_name"]} ({var2["config"]["cut_type"]})')
        ax.set_title(
            f"2D FOM Optimization for {state}\n" f'{var1["var_name"]} vs {var2["var_name"]}'
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

        # Save
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "optimization" / "2d_scans"
        plot_dir.mkdir(exist_ok=True, parents=True)

        filename = f"fom_2d_{var1['category']}_{var1['var_name']}_vs_{var2['category']}_{var2['var_name']}_{state}.pdf"
        plt.savefig(plot_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _generate_optimization_summary(self, results_df: pd.DataFrame) -> None:
        """
        Generate summary table similar to Table 7 in reference analysis

        Shows optimal cuts for each variable across all states
        """
        # Pivot table: variables × states
        pivot = results_df.pivot_table(
            index=["category", "variable"], columns="state", values="optimal_cut"
        )

        # Add cut type and description
        cut_info = results_df.drop_duplicates(["category", "variable"])[
            ["category", "variable", "cut_type", "branch_name"]
        ].set_index(["category", "variable"])

        summary = pivot.join(cut_info)

        # Reorder columns
        col_order = ["branch_name", "cut_type", "jpsi", "etac", "chic0", "chic1", "etac_2s"]
        summary = summary[col_order]

        # Save as markdown and CSV
        output_dir = Path(self.config.paths["output"]["tables_dir"])

        summary.to_csv(output_dir / "optimized_cuts_summary.csv")
        summary.to_markdown(output_dir / "optimized_cuts_summary.md")

        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)
        print(summary.to_string())
        print("\n✓ Saved to:", output_dir)

    def optimize_nd_grid_scan(self) -> pd.DataFrame:
        """
        Perform N-dimensional GRID scan using unbiased data-driven method.

        APPROACH:
        - Signal proxy: "no-charmonium" data (M(Λ̄pK⁻) > 4 GeV)
        - Background proxy: B⁺ mass sidebands
        - NO MC used → completely unbiased!

        Supports two modes:
        - Option A (universal): Optimize once, apply to all states
        - Option B (state-specific): Optimize per state

        Uses only 7 variables from nd_optimizable_selection config:
        - h1_ProbNNk, h2_ProbNNk, p_ProbNNp (PID)
        - Bu_PT, Bu_FDCHI2, Bu_IPCHI2, Bu_DTF_chi2 (B+ kinematics)

        Lambda cuts are already FIXED and applied in Phase 2.

        Grid size: 3×3×3×2×4×6×3 = 3,888 combinations

        Returns:
            DataFrame with optimal cuts (universal or per state)
        """
        import itertools

        # Get N-D grid scan variables from config
        nd_config = self.config.selection.get("nd_optimizable_selection", {})

        if not nd_config:
            raise OptimizationError(
                "No 'nd_optimizable_selection' section found in config/selection.toml!\n"
                "N-D grid scan requires this section to define variables and ranges to optimize."
            )

        # Build variable list and grid points
        all_variables = []
        grid_axes = []  # Each element is a list of values to scan

        for var_name, var_config in nd_config.items():
            if var_name == "notes":
                continue

            # Generate grid points for this variable
            begin = var_config["begin"]
            end = var_config["end"]
            step = var_config["step"]
            grid_points = np.arange(begin, end + step / 2, step)  # Include endpoint

            all_variables.append(
                {
                    "var_name": var_name,
                    "branch_name": var_config["branch_name"],
                    "cut_type": var_config["cut_type"],
                    "description": var_config.get("description", ""),
                }
            )
            grid_axes.append(grid_points)

        n_vars = len(all_variables)
        total_combinations = int(np.prod([len(axis) for axis in grid_axes]))

        print(f"\n{'='*80}")
        print(f"N-D GRID SCAN (UNBIASED): {n_vars} variables, {total_combinations:,} combinations")
        print(f"{'='*80}")
        for i, var in enumerate(all_variables):
            n_points = len(grid_axes[i])
            print(f"  {var['var_name']:20s} ({var['cut_type']:>7s}): {n_points} points")
        print(f"{'='*80}\n")

        # Combine all years
        data_arrays = [self.data[year] for year in self.data.keys()]
        data_combined = ak.concatenate(data_arrays, axis=0)

        # Get signal and background proxies
        sig_data = self.get_no_charmonium_data(data_combined)
        bkg_data = self.get_b_mass_sideband_data(data_combined)

        print(f"Signal proxy (no-charmonium): {len(sig_data):,} events")
        print(f"Background proxy (B+ sidebands): {len(bkg_data):,} events\n")

        all_results = []

        if self.state_dependent:
            # Option B: State-specific cuts (NOT YET IMPLEMENTED - needs discussion)
            print("⚠️  State-specific optimization not yet implemented!")
            print("    Using universal cuts instead.\n")
            states = ["universal"]
        else:
            # Option A: Universal cuts
            states = ["universal"]
            print("Running Option A: UNIVERSAL optimization\n")

        for state in states:
            print(f"\n{'='*60}")
            if state == "universal":
                print("Optimizing UNIVERSAL cuts for all states")
            else:
                print(f"Optimizing cuts for state: {state}")
            print(f"{'='*60}")

            # Extract branch data (once, before loop)
            sig_branches = []
            bkg_branches = []

            for var in all_variables:
                sig_branch = sig_data[var["branch_name"]]
                bkg_branch = bkg_data[var["branch_name"]]

                # Flatten jagged arrays
                if "var" in str(ak.type(sig_branch)):
                    sig_branch = ak.firsts(sig_branch)
                if "var" in str(ak.type(bkg_branch)):
                    bkg_branch = ak.firsts(bkg_branch)

                sig_branches.append(sig_branch)
                bkg_branches.append(bkg_branch)

            # Grid scan: test all combinations
            best_fom = -np.inf
            best_cuts = None
            best_n_sig = 0.0
            best_n_bkg = 0.0

            print(f"  Scanning {total_combinations:,} combinations...")

            # Use tqdm progress bar for grid scan
            desc = f"  {state:8s}" if state != "universal" else "  Universal"
            with tqdm(total=total_combinations, desc=desc, unit="combo", ncols=100) as pbar:
                for i, cut_combination in enumerate(itertools.product(*grid_axes)):
                    # Apply this combination of cuts
                    sig_mask = ak.ones_like(sig_branches[0], dtype=bool)
                    bkg_mask = ak.ones_like(bkg_branches[0], dtype=bool)

                    for j, (cut_val, var) in enumerate(
                        zip(cut_combination, all_variables, strict=False)
                    ):
                        if var["cut_type"] == "greater":
                            sig_mask = sig_mask & (sig_branches[j] > cut_val)
                            bkg_mask = bkg_mask & (bkg_branches[j] > cut_val)
                        else:
                            sig_mask = sig_mask & (sig_branches[j] < cut_val)
                            bkg_mask = bkg_mask & (bkg_branches[j] < cut_val)

                    # Count events passing cuts
                    n_sig = ak.sum(sig_mask)
                    n_bkg = ak.sum(bkg_mask)

                    # Calculate FOM
                    fom = self.compute_fom(n_sig, n_bkg)

                    # Update best if this is better
                    if fom > best_fom:
                        best_fom = fom
                        best_cuts = cut_combination
                        best_n_sig = float(n_sig)
                        best_n_bkg = float(n_bkg)
                        pbar.set_postfix(
                            FOM=f"{best_fom:.3f}", S=int(best_n_sig), B=int(best_n_bkg)
                        )

                    pbar.update(1)

            print("  ✓ Grid scan complete!")
            print(f"  Best FOM: {best_fom:.3f}")
            print(f"  n_sig: {best_n_sig:.0f}, n_bkg: {best_n_bkg:.1f}")

            # Store results
            if best_cuts is None:
                print("  WARNING: No valid cuts found!")
                continue

            print("\n  Optimal cuts:")
            for j, var in enumerate(all_variables):
                cut_val = best_cuts[j]
                print(f"    {var['var_name']:20s} {var['cut_type']:>7s} {cut_val:8.3f}")

                # For universal cuts, apply to all states
                if state == "universal":
                    for actual_state in ["jpsi", "etac", "chic0", "chic1"]:
                        all_results.append(
                            {
                                "state": actual_state,
                                "variable": var["var_name"],
                                "branch_name": var["branch_name"],
                                "optimal_cut": cut_val,
                                "cut_type": var["cut_type"],
                                "fom": best_fom,
                            }
                        )
                else:
                    # State-specific
                    all_results.append(
                        {
                            "state": state,
                            "variable": var["var_name"],
                            "branch_name": var["branch_name"],
                            "optimal_cut": cut_val,
                            "cut_type": var["cut_type"],
                            "fom": best_fom,
                        }
                    )

        # Convert to DataFrame
        df_results = pd.DataFrame(all_results)

        # Add etac_2s by copying chi_c1 (no MC available)
        if "chic1" in df_results["state"].values:
            etac_2s_cuts = df_results[df_results["state"] == "chic1"].copy()
            etac_2s_cuts["state"] = "etac_2s"
            df_results = pd.concat([df_results, etac_2s_cuts], ignore_index=True)
            print("\n✓ Added etac_2s cuts (copied from chi_c1)")

        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Method: {'State-specific' if self.state_dependent else 'Universal'}")
        print(f"Total states: {df_results['state'].nunique()}")
        print(f"Variables optimized: {df_results['variable'].nunique()}")
        print(f"{'='*80}\n")

        return df_results
