#!/usr/bin/env python3
"""
Generate presentation tables for November 3rd, 2025 slides

Reads results from pipeline output files and generates:
1. Optimal cuts per state (with FOM)
2. Yield tables (per state, per year)
3. Branching fraction ratios with statistical uncertainties
4. Selection efficiencies

All tables are output in markdown format for easy inclusion in slides.

Author: Analysis Pipeline
Date: November 3, 2025
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


class PresentationTableGenerator:
    """Generate markdown tables for presentation slides"""

    def __init__(self, tables_dir: str = "../tables", output_dir: str = "../tables/presentation"):
        """
        Args:
            tables_dir: Directory containing pipeline output tables
            output_dir: Directory to save presentation tables
        """
        self.tables_dir = Path(tables_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load all required data
        self.load_data()

    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")

        try:
            # Optimal cuts
            self.optimal_cuts = pd.read_csv(self.tables_dir / "optimized_cuts.csv")
            print(f"  ✓ Loaded optimal cuts: {len(self.optimal_cuts)} rows")

            # Yields
            self.yields = pd.read_csv(self.tables_dir / "phase5_yields.csv")
            print(f"  ✓ Loaded yields: {len(self.yields)} rows")

            # Branching fraction ratios
            self.br_ratios = pd.read_csv(self.tables_dir / "branching_fraction_ratios.csv")
            print(f"  ✓ Loaded BR ratios: {len(self.br_ratios)} rows")

            # Efficiencies
            self.efficiencies = pd.read_csv(self.tables_dir / "efficiencies.csv")
            print(f"  ✓ Loaded efficiencies: {len(self.efficiencies)} rows")

            print("✓ All data loaded successfully!\n")

        except FileNotFoundError as e:
            print(f"ERROR: Could not find required file: {e}")
            print("Please run the pipeline first to generate output tables.")
            sys.exit(1)

    def generate_optimal_cuts_table(self):
        """
        Generate Table 1: Optimal Selection Cuts

        Shows optimal cuts for each variable and state with FOM values
        FOM formula: FOM = n_sig / sqrt(n_bkg + n_sig)
        """
        print("=" * 80)
        print("TABLE 1: OPTIMAL SELECTION CUTS")
        print("=" * 80)

        # Define the FOM formula in LaTeX for table caption
        fom_formula = r"FOM = n_{sig} / \sqrt{n_{bkg} + n_{sig}}"

        # Group by state
        states = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]
        state_names = {
            "jpsi": r"$J/\psi$",
            "etac": r"$\eta_c$",
            "chic0": r"$\chi_{c0}$",
            "chic1": r"$\chi_{c1}$",
            "etac_2s": r"$\eta_c(2S)$",
        }

        for state in states:
            state_data = self.optimal_cuts[self.optimal_cuts["state"] == state].copy()

            if len(state_data) == 0:
                print(f"Warning: No cuts found for {state}")
                continue

            # Get unique FOM value (all variables have same FOM for a state)
            fom_value = state_data["max_fom"].iloc[0]
            n_sig = state_data["n_sig_at_optimal"].iloc[0]
            n_bkg = state_data["n_bkg_at_optimal"].iloc[0]

            # Create presentation table
            table_data = []
            for _, row in state_data.iterrows():
                # Format cut type with symbol
                if row["cut_type"] == "greater":
                    cut_symbol = ">"
                else:
                    cut_symbol = "<"

                table_data.append(
                    {
                        "Variable": row["description"],
                        "Cut": f"{cut_symbol} {row['optimal_cut']:.1f}",
                        "Branch": f"`{row['branch_name']}`",
                    }
                )

            df = pd.DataFrame(table_data)

            # Save to markdown
            filename = f"table1_optimal_cuts_{state}.md"
            filepath = self.output_dir / filename

            with open(filepath, "w") as f:
                f.write(f"# Optimal Selection Cuts: {state_names[state]}\n\n")
                f.write(f"**Figure of Merit:** ${fom_formula}$\n\n")
                f.write(f"**FOM Value:** {fom_value:.2f}\n\n")
                f.write(f"**Signal events:** {n_sig:.0f}  \n")
                f.write(f"**Background events:** {n_bkg:.0f}\n\n")
                f.write(df.to_markdown(index=False))
                f.write("\n")

            print(f"\n{state_names[state]} (FOM = {fom_value:.2f}):")
            print(df.to_markdown(index=False))
            print(f"✓ Saved to: {filepath}\n")

        # Generate combined summary table
        self._generate_combined_cuts_table(states, state_names)

    def _generate_combined_cuts_table(self, states, state_names):
        """Generate a combined table showing all cuts for all states side-by-side"""

        print("\nCOMBINED TABLE: All States")
        print("-" * 80)

        # Get all unique variables (should be same for all states)
        example_state = self.optimal_cuts[self.optimal_cuts["state"] == states[0]]
        variables = example_state["variable"].tolist()

        # Build combined table
        combined_data = []
        for var in variables:
            row = {
                "Variable": example_state[example_state["variable"] == var]["description"].iloc[0]
            }

            for state in states:
                state_data = self.optimal_cuts[
                    (self.optimal_cuts["state"] == state) & (self.optimal_cuts["variable"] == var)
                ]

                if len(state_data) > 0:
                    cut_val = state_data["optimal_cut"].iloc[0]
                    cut_type = state_data["cut_type"].iloc[0]
                    symbol = ">" if cut_type == "greater" else "<"
                    row[state_names[state]] = f"{symbol} {cut_val:.1f}"
                else:
                    row[state_names[state]] = "N/A"

            combined_data.append(row)

        df_combined = pd.DataFrame(combined_data)

        # Add FOM row at the bottom
        fom_row = {"Variable": "**FOM**"}
        for state in states:
            state_data = self.optimal_cuts[self.optimal_cuts["state"] == state]
            if len(state_data) > 0:
                fom_val = state_data["max_fom"].iloc[0]
                fom_row[state_names[state]] = f"**{fom_val:.2f}**"
            else:
                fom_row[state_names[state]] = "N/A"

        df_combined = pd.concat([df_combined, pd.DataFrame([fom_row])], ignore_index=True)

        # Save
        filepath = self.output_dir / "table1_optimal_cuts_combined.md"
        with open(filepath, "w") as f:
            f.write("# Optimal Selection Cuts: All States\n\n")
            f.write("**Figure of Merit:** $FOM = n_{sig} / \\sqrt{n_{bkg} + n_{sig}}$\n\n")
            f.write(df_combined.to_markdown(index=False))
            f.write("\n")

        print(df_combined.to_markdown(index=False))
        print(f"✓ Saved to: {filepath}\n")

    def generate_yield_tables(self):
        """
        Generate Table 2: Signal Yields

        Shows fitted yields for each state, separated by year and combined
        """
        print("=" * 80)
        print("TABLE 2: SIGNAL YIELDS")
        print("=" * 80)

        states = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]
        state_names = {
            "jpsi": r"$J/\psi$",
            "etac": r"$\eta_c$",
            "chic0": r"$\chi_{c0}$",
            "chic1": r"$\chi_{c1}$",
            "etac_2s": r"$\eta_c(2S)$",
        }

        # Yields by year
        years = ["2016", "2017", "2018"]

        # Build table: states × years
        yield_data = []
        for state in states:
            row = {"State": state_names[state]}

            for year in years:
                year_data = self.yields[
                    (self.yields["year"] == year) & (self.yields["state"] == state)
                ]

                if len(year_data) > 0:
                    y = year_data["yield"].iloc[0]
                    e = year_data["error"].iloc[0]
                    row[year] = f"{y:.1f} ± {e:.1f}"
                else:
                    row[year] = "N/A"

            # Combined
            combined_data = self.yields[
                (self.yields["year"] == "combined") & (self.yields["state"] == state)
            ]

            if len(combined_data) > 0:
                y = combined_data["yield"].iloc[0]
                e = combined_data["error"].iloc[0]
                row["Combined"] = f"{y:.1f} ± {e:.1f}"
            else:
                row["Combined"] = "N/A"

            yield_data.append(row)

        df = pd.DataFrame(yield_data)

        # Save
        filepath = self.output_dir / "table2_yields.md"
        with open(filepath, "w") as f:
            f.write("# Signal Yields from Mass Fits\n\n")
            f.write(
                "Yields obtained from simultaneous fits to $M(\\Lambda K^+K^-)$ mass distributions.\n\n"
            )
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            f.write("**Note:** Errors are statistical only.\n")

        print("\nSignal Yields (all years):")
        print(df.to_markdown(index=False))
        print(f"✓ Saved to: {filepath}\n")

        # Also generate per-year tables
        self._generate_per_year_yield_tables(states, state_names)

    def _generate_per_year_yield_tables(self, states, state_names):
        """Generate individual yield tables for each year"""

        years = ["2016", "2017", "2018", "combined"]

        for year in years:
            year_data_list = []

            for state in states:
                year_state = self.yields[
                    (self.yields["year"] == year) & (self.yields["state"] == state)
                ]

                if len(year_state) > 0:
                    y = year_state["yield"].iloc[0]
                    e = year_state["error"].iloc[0]
                    year_data_list.append(
                        {
                            "State": state_names[state],
                            "Yield": f"{y:.1f}",
                            "Error": f"{e:.1f}",
                            "Yield ± Error": f"{y:.1f} ± {e:.1f}",
                        }
                    )

            if len(year_data_list) > 0:
                df = pd.DataFrame(year_data_list)

                filepath = self.output_dir / f"table2_yields_{year}.md"
                with open(filepath, "w") as f:
                    f.write(f"# Signal Yields: {year.upper()}\n\n")
                    f.write(df.to_markdown(index=False))
                    f.write("\n")

                print(f"  ✓ Saved {year} yields to: {filepath}")

        print()

    def generate_br_ratio_tables(self):
        """
        Generate Table 3: Branching Fraction Ratios

        Shows measured ratios of branching fractions with statistical uncertainties
        """
        print("=" * 80)
        print("TABLE 3: BRANCHING FRACTION RATIOS")
        print("=" * 80)

        # Format the BR ratios table
        br_data = []

        ratio_names = {
            ("etac", "jpsi"): r"$\mathcal{B}(\eta_c) / \mathcal{B}(J/\psi)$",
            ("chic0", "jpsi"): r"$\mathcal{B}(\chi_{c0}) / \mathcal{B}(J/\psi)$",
            ("chic1", "jpsi"): r"$\mathcal{B}(\chi_{c1}) / \mathcal{B}(J/\psi)$",
            ("chic1", "chic0"): r"$\mathcal{B}(\chi_{c1}) / \mathcal{B}(\chi_{c0})$",
        }

        for _, row in self.br_ratios.iterrows():
            num = row["numerator"]
            den = row["denominator"]
            ratio = row["ratio"]
            error = row["stat_error"]

            # Get formatted name
            key = (num, den)
            if key in ratio_names:
                ratio_name = ratio_names[key]
            else:
                ratio_name = f"{num}/{den}"

            br_data.append(
                {
                    "Ratio": ratio_name,
                    "Value": f"{ratio:.3f}",
                    "Stat. Error": f"{error:.3f}",
                    "Result": f"{ratio:.3f} ± {error:.3f}",
                }
            )

        df = pd.DataFrame(br_data)

        # Save
        filepath = self.output_dir / "table3_br_ratios.md"
        with open(filepath, "w") as f:
            f.write("# Branching Fraction Ratios\n\n")
            f.write("Ratios of branching fractions $\\mathcal{B}(B^+ \\to \\Lambda \\bar{p} X)$ ")
            f.write("for different charmonium states $X$.\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            f.write("**Note:** Uncertainties are statistical only. ")
            f.write("Systematic uncertainties from efficiency and luminosity are not included.\n")

        print("\nBranching Fraction Ratios:")
        print(df.to_markdown(index=False))
        print(f"✓ Saved to: {filepath}\n")

    def generate_efficiency_tables(self):
        """
        Generate Table 4: Selection Efficiencies

        Shows selection × reconstruction efficiencies for each state and year
        """
        print("=" * 80)
        print("TABLE 4: SELECTION EFFICIENCIES")
        print("=" * 80)

        state_names = {
            "jpsi": r"$J/\psi$",
            "etac": r"$\eta_c$",
            "chic0": r"$\chi_{c0}$",
            "chic1": r"$\chi_{c1}$",
            "etac_2s": r"$\eta_c(2S)$",
        }

        # Parse the efficiency table (it has a specific format)
        eff_data = []

        for _, row in self.efficiencies.iterrows():
            state = row["State"]

            row_data = {"State": state_names.get(state, state)}

            # Parse each year column: "eff ± error, N_pass/N_total"
            for year in ["2016", "2017", "2018"]:
                col_name = f"{year}_eff"
                if col_name in row:
                    row_data[year] = row[col_name]

            eff_data.append(row_data)

        df = pd.DataFrame(eff_data)

        # Save
        filepath = self.output_dir / "table4_efficiencies.md"
        with open(filepath, "w") as f:
            f.write("# Selection Efficiencies\n\n")
            f.write("Combined selection and reconstruction efficiencies ")
            f.write("$\\epsilon_{sel} \\times \\epsilon_{reco}$ measured from MC simulation.\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            f.write("**Format:** efficiency ± error, N_pass/N_total\n")

        print("\nSelection Efficiencies:")
        print(df.to_markdown(index=False))
        print(f"✓ Saved to: {filepath}\n")

        # Also create a simplified version (just efficiencies, no N values)
        self._generate_simplified_efficiency_table(eff_data, state_names)

    def _generate_simplified_efficiency_table(self, eff_data, state_names):
        """Create a cleaner efficiency table with just the efficiency values"""

        simplified_data = []

        for item in eff_data:
            row = {"State": item["State"]}

            for year in ["2016", "2017", "2018"]:
                if year in item:
                    # Extract just the efficiency part (before the comma)
                    eff_str = item[year].split(",")[0].strip()
                    row[year] = eff_str

            simplified_data.append(row)

        df = pd.DataFrame(simplified_data)

        # Save
        filepath = self.output_dir / "table4_efficiencies_simple.md"
        with open(filepath, "w") as f:
            f.write("# Selection Efficiencies (Simplified)\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")

        print("  ✓ Saved simplified efficiency table to:", filepath, "\n")

    def generate_summary_table(self):
        """
        Generate Table 5: Analysis Summary

        One-page summary with key results for each state
        """
        print("=" * 80)
        print("TABLE 5: ANALYSIS SUMMARY")
        print("=" * 80)

        states = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]
        state_names = {
            "jpsi": r"$J/\psi$",
            "etac": r"$\eta_c$",
            "chic0": r"$\chi_{c0}$",
            "chic1": r"$\chi_{c1}$",
            "etac_2s": r"$\eta_c(2S)$",
        }

        summary_data = []

        for state in states:
            # Get FOM
            cuts_data = self.optimal_cuts[self.optimal_cuts["state"] == state]
            if len(cuts_data) > 0:
                fom = cuts_data["max_fom"].iloc[0]
                fom_str = f"{fom:.2f}"
            else:
                fom_str = "N/A"

            # Get combined yield
            yield_data = self.yields[
                (self.yields["year"] == "combined") & (self.yields["state"] == state)
            ]
            if len(yield_data) > 0:
                y = yield_data["yield"].iloc[0]
                e = yield_data["error"].iloc[0]
                yield_str = f"{y:.1f} ± {e:.1f}"
            else:
                yield_str = "N/A"

            # Get average efficiency (across years)
            eff_data = self.efficiencies[self.efficiencies["State"] == state]
            if len(eff_data) > 0:
                # Extract efficiency values from "eff ± error, N/N" format
                effs = []
                for year in ["2016", "2017", "2018"]:
                    col = f"{year}_eff"
                    if col in eff_data.columns:
                        eff_str = eff_data[col].iloc[0].split("±")[0].strip()
                        try:
                            effs.append(float(eff_str))
                        except (ValueError, IndexError):
                            pass

                if len(effs) > 0:
                    avg_eff = np.mean(effs)
                    eff_str = f"{avg_eff:.3f}"
                else:
                    eff_str = "N/A"
            else:
                eff_str = "N/A"

            summary_data.append(
                {
                    "State": state_names[state],
                    "FOM": fom_str,
                    "Yield (Combined)": yield_str,
                    "Avg. Efficiency": eff_str,
                }
            )

        df = pd.DataFrame(summary_data)

        # Save
        filepath = self.output_dir / "table5_summary.md"
        with open(filepath, "w") as f:
            f.write("# Analysis Summary\n\n")
            f.write("Key results for all charmonium states.\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")

        print("\nAnalysis Summary:")
        print(df.to_markdown(index=False))
        print(f"✓ Saved to: {filepath}\n")

    def generate_all_tables(self):
        """Generate all presentation tables"""
        print("\n" + "=" * 80)
        print("GENERATING PRESENTATION TABLES")
        print("Date: November 3, 2025")
        print("=" * 80 + "\n")

        self.generate_optimal_cuts_table()
        self.generate_yield_tables()
        self.generate_br_ratio_tables()
        self.generate_efficiency_tables()
        self.generate_summary_table()

        print("=" * 80)
        print("✓ ALL TABLES GENERATED SUCCESSFULLY!")
        print(f"✓ Output directory: {self.output_dir.absolute()}")
        print("=" * 80)

        # List all generated files
        print("\nGenerated files:")
        for f in sorted(self.output_dir.glob("*.md")):
            print(f"  - {f.name}")
        print()


def main():
    """Main entry point"""

    # Check if running from scripts directory
    script_dir = Path(__file__).parent
    if script_dir.name == "scripts":
        tables_dir = script_dir.parent / "tables"
        output_dir = tables_dir / "presentation"
    else:
        tables_dir = Path("tables")
        output_dir = tables_dir / "presentation"

    # Generate tables
    generator = PresentationTableGenerator(tables_dir=str(tables_dir), output_dir=str(output_dir))

    generator.generate_all_tables()

    return 0


if __name__ == "__main__":
    sys.exit(main())
