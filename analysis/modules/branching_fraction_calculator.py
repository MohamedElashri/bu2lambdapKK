from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BranchingFractionCalculator:
    """
    Calculate branching fraction ratios relative to J/ψ.

    Key formula:

    Br(B⁺ → cc̄ X) × Br(cc̄ → Λ̄pK⁻)
    ─────────────────────────────── = Σ(N_cc/ε_cc) / Σ(N_J/ψ/ε_J/ψ)
    Br(B⁺ → J/ψ X) × Br(J/ψ → Λ̄pK⁻)

    These ratios are physics-meaningful and don't require
    knowing individual branching fractions!

    Phase 7 Implementation:
    - Uses yields from Phase 5 (mass fitting)
    - Uses efficiency ratios from Phase 6
    - Combines all years with proper error propagation
    - Statistical uncertainties only (draft analysis)

    Attributes:
        yields: Dictionary of yields per year and state with errors
        efficiencies: Dictionary of efficiencies per state and year
        config: Configuration object
    """

    def __init__(
        self,
        yields: dict[str, dict[str, tuple[float, float]]],
        efficiencies: dict[str, dict[str, dict[str, float]]],
        config: Any,
    ) -> None:
        """
        Initialize branching fraction calculator.

        Args:
            yields: {year: {state: (value, error)}}
                   e.g., {"2016": {"jpsi": (1000.0, 50.0), "etac": (200.0, 20.0)}}
            efficiencies: {state: {year: {"eff": value, "err": error}}}
                   e.g., {"jpsi": {"2016": {"eff": 0.85, "err": 0.03}}}
            config: Configuration object with paths
        """
        self.yields: dict[str, dict[str, tuple[float, float]]] = yields
        self.efficiencies: dict[str, dict[str, dict[str, float]]] = efficiencies
        self.config: Any = config

    def calculate_efficiency_corrected_yield(self, state: str) -> tuple[float, float]:
        """
        Calculate Σ(N^year / ε^year) for a given state.

        Sum over all years, propagating uncertainties properly.

        Error propagation for Y = N/ε:
        σ_Y = Y × sqrt((σ_N/N)² + (σ_ε/ε)²)

        Args:
            state: State name ("jpsi", "etac", "chic0", "chic1")

        Returns:
            Tuple of (corrected_yield, error)
        """
        corrected_yields = []
        errors_sq = []

        for year in sorted(self.yields.keys()):
            # Skip "combined" - we use per-year efficiencies
            if year == "combined":
                continue

            n_year, n_err = self.yields[year][state]
            eps_year = self.efficiencies[state][year]["eff"]
            eps_err = self.efficiencies[state][year]["err"]

            if eps_year > 0:
                corrected = n_year / eps_year

                # Full error propagation including efficiency uncertainty
                rel_err_n = n_err / n_year if n_year > 0 else 0.0
                rel_err_eps = eps_err / eps_year
                error = corrected * np.sqrt(rel_err_n**2 + rel_err_eps**2)

                corrected_yields.append(corrected)
                errors_sq.append(error**2)

        total_corrected = sum(corrected_yields)
        total_error = np.sqrt(sum(errors_sq))

        return total_corrected, total_error

    def calculate_ratio_to_jpsi(self, state: str) -> tuple[float, float]:
        """
        Calculate ratio of branching fractions relative to J/ψ.

        R = [Br(B⁺→cc̄ X) * Br(cc̄→Λ̄pK⁻)] / [Br(B⁺→J/ψ X) * Br(J/ψ→Λ̄pK⁻)]
          = Σ(N_cc/ε_cc) / Σ(N_J/ψ/ε_J/ψ)

        Args:
            state: State name ("etac", "chic0", "chic1")

        Returns:
            Tuple of (ratio, error)
        """
        # Efficiency-corrected yields
        yield_cc, err_cc = self.calculate_efficiency_corrected_yield(state)
        yield_jpsi, err_jpsi = self.calculate_efficiency_corrected_yield("jpsi")

        # Ratio
        ratio = yield_cc / yield_jpsi if yield_jpsi > 0 else 0.0

        # Error propagation: R = A/B → σ_R = R × sqrt((σ_A/A)² + (σ_B/B)²)
        rel_err_cc = err_cc / yield_cc if yield_cc > 0 else 0.0
        rel_err_jpsi = err_jpsi / yield_jpsi if yield_jpsi > 0 else 0.0

        error = ratio * np.sqrt(rel_err_cc**2 + rel_err_jpsi**2)

        return ratio, error

    def calculate_all_ratios(self) -> pd.DataFrame:
        """
        Calculate all branching fraction ratios.

        Results:
        - ηc/J/ψ
        - χc0/J/ψ
        - χc1/J/ψ
        - ηc(2S)/J/ψ
        - χc1/χc0 (derived from above)

        Returns:
            DataFrame with ratio results, saved to CSV
        """
        results = []

        print("\n" + "=" * 80)
        print("BRANCHING FRACTION RATIOS (Statistical uncertainties only)")
        print("=" * 80)

        # Direct ratios to J/ψ
        for state in ["etac", "chic0", "chic1", "etac_2s"]:
            ratio, error = self.calculate_ratio_to_jpsi(state)

            print(f"\nBr(B⁺ → {state} X) * Br({state} → Λ̄pK⁻)")
            print("───────────────────────────────────────────")
            print("Br(B⁺ → J/ψ X) * Br(J/ψ → Λ̄pK⁻)")
            print(f"= {ratio:.3f} ± {error:.3f}")

            results.append(
                {"numerator": state, "denominator": "jpsi", "ratio": ratio, "stat_error": error}
            )

        # Derived ratio: χc1/χc0
        ratio_chic1_jpsi, err1 = self.calculate_ratio_to_jpsi("chic1")
        ratio_chic0_jpsi, err0 = self.calculate_ratio_to_jpsi("chic0")

        ratio_chic1_chic0 = ratio_chic1_jpsi / ratio_chic0_jpsi if ratio_chic0_jpsi > 0 else 0.0

        # Error propagation for derived ratio
        rel_err1 = err1 / ratio_chic1_jpsi if ratio_chic1_jpsi > 0 else 0.0
        rel_err0 = err0 / ratio_chic0_jpsi if ratio_chic0_jpsi > 0 else 0.0
        error_chic1_chic0 = ratio_chic1_chic0 * np.sqrt(rel_err1**2 + rel_err0**2)

        print("\nBr(B⁺ → χc1 X) * Br(χc1 → Λ̄pK⁻)")
        print("───────────────────────────────────────────")
        print("Br(B⁺ → χc0 X) * Br(χc0 → Λ̄pK⁻)")
        print(f"= {ratio_chic1_chic0:.3f} ± {error_chic1_chic0:.3f}")

        results.append(
            {
                "numerator": "chic1",
                "denominator": "chic0",
                "ratio": ratio_chic1_chic0,
                "stat_error": error_chic1_chic0,
            }
        )

        df = pd.DataFrame(results)

        # Save
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        df.to_csv(output_dir / "branching_fraction_ratios.csv", index=False)

        return df

    def check_yield_consistency_per_year(self) -> pd.DataFrame:
        """
        Check consistency of yields per year.

        Following Table 28 in reference analysis.
        Plot: N/(L*ε) vs year
        Should be consistent if detector stable.

        Returns:
            DataFrame with N/(L*ε) for each (state, year), saved to CSV with plot
        """
        results = []

        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
            for year in sorted(self.yields.keys()):
                # Skip "combined" - we check per-year consistency
                if year == "combined":
                    continue

                n_year, n_err = self.yields[year][state]
                eps_year = self.efficiencies[state][year]["eff"]
                eps_err = self.efficiencies[state][year]["err"]
                lumi = self.config.luminosity["integrated_luminosity"][year]

                if eps_year > 0 and lumi > 0:
                    normalized_yield = n_year / (lumi * eps_year)

                    # Error propagation for N/(L×ε)
                    rel_err_n = n_err / n_year if n_year > 0 else 0.0
                    rel_err_eps = eps_err / eps_year
                    error = normalized_yield * np.sqrt(rel_err_n**2 + rel_err_eps**2)

                    results.append(
                        {
                            "state": state,
                            "year": year,
                            "N": n_year,
                            "L": lumi,
                            "eps": eps_year,
                            "N_over_L_eps": normalized_yield,
                            "error": error,
                        }
                    )

        df = pd.DataFrame(results)

        # Plot (2x3 grid for 5 states)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        states_to_plot = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]
        for i, state in enumerate(states_to_plot):
            ax = axes[i]
            data_state = df[df["state"] == state]

            years = [int(y) for y in data_state["year"]]
            values = data_state["N_over_L_eps"].values
            errors = data_state["error"].values

            ax.errorbar(years, values, yerr=errors, fmt="o-", markersize=8, capsize=5, linewidth=2)
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("N / (L × ε)", fontsize=12)
            ax.set_title(f"{state}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(years)

        plt.tight_layout()

        # Save
        plot_dir = Path(self.config.paths["output"]["plots_dir"])
        plt.savefig(plot_dir / "yield_consistency_check.pdf", dpi=150, bbox_inches="tight")
        plt.close()

        print("\n✓ Yield consistency check saved")

        # Also save table
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        df.to_csv(output_dir / "yield_consistency.csv", index=False)

        return df

    def generate_final_summary(self, ratios_df: pd.DataFrame) -> None:
        """
        Generate final summary of results.

        Creates:
        1. Markdown summary file
        2. Formatted results table
        3. Next steps documentation

        Args:
            ratios_df: DataFrame containing calculated ratios
        """
        output_dir = Path(self.config.paths["output"]["results_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)

        # Markdown summary
        md = "# Branching Fraction Ratios - Final Results\n\n"
        md += "## Self-Normalization to J/ψ\n\n"
        md += "All results are **statistical uncertainties only** (draft analysis).\n\n"
        md += "Systematic uncertainties and external branching fraction uncertainties\n"
        md += "will be added in full analysis.\n\n"

        md += "## Results\n\n"
        md += "| Ratio | Value | Stat. Error |\n"
        md += "|-------|-------|-------------|\n"

        for _, row in ratios_df.iterrows():
            num = row["numerator"]
            den = row["denominator"]
            val = row["ratio"]
            err = row["stat_error"]

            md += f"| Br(B⁺→{num} X)×Br({num}→Λ̄pK⁻) / "
            md += f"Br(B⁺→{den} X)×Br({den}→Λ̄pK⁻) | "
            md += f"{val:.3f} | ±{err:.3f} |\n"

        md += "\n## Comparison with Theory\n\n"
        md += "For χc states, NRQCD predicts (Colour Octet dominance):\n"
        md += "- Br(χc1)/Br(χc0) ≈ 3\n"
        md += "- Br(χc2)/Br(χc0) ≈ 5\n\n"

        md += f"Our result: Br(χc1)/Br(χc0) = {ratios_df[ratios_df['numerator']=='chic1']['ratio'].values[0]:.3f}\n\n"
        md += "**Note**: These predictions were not observed in B⁺→φφ analysis either.\n\n"

        md += "## Next Steps\n\n"
        md += "1. ✓ Draft analysis complete with statistical uncertainties\n"
        md += "2. ⚠️ Add systematic uncertainties:\n"
        md += "   - Fit model variations\n"
        md += "   - Selection optimization uncertainties\n"
        md += "   - Efficiency uncertainties\n"
        md += "   - Multiple candidate effects\n"
        md += "3. ⚠️ Complete efficiency calculations:\n"
        md += "   - Reconstruction efficiency (currently placeholder)\n"
        md += "   - Stripping efficiency (currently placeholder)\n"
        md += "4. ⚠️ Consider interference effects (if significant)\n"
        md += "5. ⚠️ Measure multiple candidate fraction\n"

        with open(output_dir / "final_results.md", "w") as f:
            f.write(md)

        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        print(md)
        print(f"\n✓ Saved to: {output_dir / 'final_results.md'}")
