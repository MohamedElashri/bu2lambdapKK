import argparse

import numpy as np
import ROOT as rt  # type: ignore
from tqdm import tqdm
from uncertainties import ufloat

# Parse command line arguments
parser = argparse.ArgumentParser(description="Calculate trigger efficiencies")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output for debugging"
)
parser.add_argument(
    "-l0",
    "--l0-breakdown",
    action="store_true",
    help="Show breakdown of L0Global_TIS contributions",
)
args = parser.parse_args()

# Define trigger cuts for each level
L0_trigger_cuts = [
    "Bu_L0Global_TIS > 0",
    "Bu_L0HadronDecision_TOS > 0",
    "(Bu_L0Global_TIS > 0 || Bu_L0HadronDecision_TOS > 0)",
]

# Define individual L0Global_TIS contributing lines
L0_TIS_components = [
    "Bu_L0GlobalDecision_TIS > 0",
    "Bu_L0PhysDecision_TIS > 0",
    "Bu_L0HadronDecision_TIS > 0",
    "Bu_L0MuonDecision_TIS > 0",
    "Bu_L0MuonHighDecision_TIS > 0",
    "Bu_L0DiMuonDecision_TIS > 0",
    "Bu_L0PhotonDecision_TIS > 0",
    "Bu_L0ElectronDecision_TIS > 0",
]

L1_trigger_cuts = [
    "Bu_Hlt1TrackMVADecision_TOS > 0",
    "Bu_Hlt1TwoTrackMVADecision_TOS > 0",
    "(Bu_Hlt1TrackMVADecision_TOS > 0 || Bu_Hlt1TwoTrackMVADecision_TOS > 0)",
]

L2_trigger_cuts = [
    "Bu_Hlt2Topo2BodyDecision_TOS > 0",
    "Bu_Hlt2Topo3BodyDecision_TOS > 0",
    "Bu_Hlt2Topo4BodyDecision_TOS > 0",
    "(Bu_Hlt2Topo2BodyDecision_TOS > 0 || Bu_Hlt2Topo3BodyDecision_TOS > 0 || Bu_Hlt2Topo4BodyDecision_TOS > 0)",
]

# Define fixed truth cut string
truthpkk = """(abs(Bu_TRUEID)==521) &&
                    (abs(L0_TRUEID)==3122) &&
                    (abs(Lp_TRUEID)==2212) &&
                    (abs(Lpi_TRUEID)==211) &&
                    (abs(Lp_MC_MOTHER_ID)==3122) &&
                    (abs(Lpi_MC_MOTHER_ID)==3122) &&
                    (abs(p_TRUEID)==2212) &&
                    (abs(h1_TRUEID)==321) &&
                    (abs(h2_TRUEID)==321) &&
                    (abs(p_MC_MOTHER_ID)==521) &&
                    (abs(L0_MC_MOTHER_ID)==521) &&
                    (((abs(h1_MC_MOTHER_ID)==521) && (abs(h2_MC_MOTHER_ID)==521)) ||
                     ((abs(h2_MC_MOTHER_ID)==521) && (abs(h1_MC_MOTHER_ID)==521)))"""

# Remove all newlines and extra spaces to make it a single-line expression
truthpkk = truthpkk.replace("\n", " ").replace("  ", " ").strip()


def debug_print(*print_args, **kwargs):
    """Print only if verbose mode is enabled"""
    if args.verbose:
        print(*print_args, **kwargs)


def getEntries(tuples, tree, cutstr):
    chain = rt.TChain(tree)  # type: ignore

    for _tuple in tuples:
        try:
            chain.Add(_tuple)
            debug_print(f"Added file to chain: {_tuple}")
        except Exception as e:
            print(f"Error adding file {_tuple}: {e}")

    # Check if chain has entries
    entries_total = chain.GetEntries()
    if entries_total == 0:
        print("Warning: Chain contains no entries!")
        return 0

    debug_print(f"Chain entries: {entries_total}")
    debug_print(f"Applying cut: {cutstr}")

    try:
        entries = chain.GetEntries(cutstr)
        return entries
    except Exception as e:
        print(f"Error applying cut: {e}")
        return 0


def caleff(nPass, nGen):
    if nGen == 0:
        print("Warning: Division by zero. Setting efficiency to 0.")
        return ufloat(0, 0)

    eff = nPass / nGen
    eff_err = np.sqrt(eff * (1 - eff) / nGen)
    return ufloat(eff, eff_err)


def print_table_both_formats(
    caption, label, headers, row_names, data, track_type, is_breakdown=False
):
    """Print table in both LaTeX and Markdown formats

    Args:
        caption: Table caption text
        label: LaTeX label for the table
        headers: List of column headers (e.g., ['2016', '2017', '2018'])
        row_names: List of row labels
        data: Dictionary with structure data[track][year] containing efficiency values
        track_type: Either 'LL' or 'DD'
        is_breakdown: Boolean indicating if this is a breakdown table (affects OR row formatting)
    """

    # Print LaTeX version
    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{{caption}}}")
    print("\\begin{tabular}{l|ccc}")
    print("\\hline")
    print(f"Levels & {' & '.join(headers)} \\\\ \\hline")

    for i, row_name in enumerate(row_names):
        eff_str = ""
        for year_idx, year in enumerate(["16", "17", "18"]):
            if i < len(data[track_type][year]):
                eff = round(data[track_type][year][i].nominal_value * 100, 2)
                eff_err = round(data[track_type][year][i].std_dev * 100, 2)

                # Check if this is an OR row or Total row
                is_or_or_total = (
                    "OR" in row_name or "Total" in row_name or "\\texttt{Total}" in row_name
                )

                if is_or_or_total:
                    if year_idx == 2:  # Last year (2018)
                        eff_str += f" & $\\bm{{{eff}\\pm{eff_err}}}$ \\\\ \\hline"
                    else:
                        eff_str += f" & $\\bm{{{eff}\\pm{eff_err}}}$"
                else:
                    if year_idx == 2:  # Last year (2018)
                        eff_str += f" & ${eff}\\pm{eff_err}$ \\\\"
                    else:
                        eff_str += f" & ${eff}\\pm{eff_err}$"
            else:
                if year_idx == 2:
                    eff_str += " & $0.00\\pm0.00$ \\\\"
                else:
                    eff_str += " & $0.00\\pm0.00$"

        print(f"{row_name} {eff_str} ")

    print("\\hline")
    print("\\end{tabular}")
    print(f"\\label{{{label}}}")
    print("\\end{table}")

    print("\n")

    # Print Markdown version
    print(f"**{caption}**\n")

    # Create markdown table header
    header_row = "| Levels | " + " | ".join(headers) + " |"
    separator_row = "|" + "|".join(["---"] * (len(headers) + 1)) + "|"

    print(header_row)
    print(separator_row)

    for i, row_name in enumerate(row_names):
        # Clean up LaTeX formatting for markdown
        md_row_name = (
            row_name.replace("\\texttt{", "")
            .replace("}", "")
            .replace("\\bm{", "")
            .replace("\\", "")
        )
        row_values = [md_row_name]

        for year in ["16", "17", "18"]:
            if i < len(data[track_type][year]):
                eff = round(data[track_type][year][i].nominal_value * 100, 2)
                eff_err = round(data[track_type][year][i].std_dev * 100, 2)

                # Check if this is an OR row or Total row for bold formatting
                is_or_or_total = (
                    "OR" in row_name or "Total" in row_name or "texttt{Total}" in row_name
                )

                if is_or_or_total:
                    row_values.append(f"**{eff}±{eff_err}**")
                else:
                    row_values.append(f"{eff}±{eff_err}")
            else:
                row_values.append("0.00±0.00")

        print("| " + " | ".join(row_values) + " |")

    print("\n")


efficiency = {}
l0_tis_breakdown = {}  # For storing L0Global_TIS breakdown

# Calculate total trees to process (3 years × 2 polarities × 2 track types = 12 trees)
total_trees = 3 * 2 * 2  # years × polarities × track types
with tqdm(total=total_trees, desc="Processing trees") as pbar:
    for track in ["LL", "DD"]:
        efficiency[track] = {}
        l0_tis_breakdown[track] = {}

        for year in ["16", "17", "18"]:
            efficiency[track][year] = []
            l0_tis_breakdown[track][year] = []

            # Create explicit paths for the files
            base_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/MC/DaVinciTuples/restripped.MC/"

            # Process MU file (contains 1 tree)
            mu_file = f"{base_dir}MC{year}MUBu2L0barPKpKm.root/B2L0barPKpKm_{track}/DecayTree"
            tuple_path = [mu_file]
            debug_print(f"Processing tree: {year} MU {track}")

            # First check if file exists and get events with truth cut
            before_trigger_evts_mu = getEntries(tuple_path, "DecayTree", cutstr=truthpkk)
            debug_print(
                f"MU Tree: {track}, Year: {year}, Before trigger events: {before_trigger_evts_mu}"
            )

            # Update progress bar after processing this tree
            pbar.update(1)

            # Process MD file (contains 1 tree)
            md_file = f"{base_dir}MC{year}MDBu2L0barPKpKm.root/B2L0barPKpKm_{track}/DecayTree"
            tuple_path = [md_file]
            debug_print(f"Processing tree: {year} MD {track}")

            # First check if file exists and get events with truth cut
            before_trigger_evts_md = getEntries(tuple_path, "DecayTree", cutstr=truthpkk)
            debug_print(
                f"MD Tree: {track}, Year: {year}, Before trigger events: {before_trigger_evts_md}"
            )

            # Update progress bar after processing this tree
            pbar.update(1)

            # Combine events from both trees
            before_trigger_evts = before_trigger_evts_mu + before_trigger_evts_md

            # Continue with combined events from both files
            tuple_path = [mu_file, md_file]

            if before_trigger_evts == 0:
                print(f"Warning: No events found for {track} {year} before trigger. Skipping...")
                # Fill with zeros to maintain structure
                efficiency[track][year] = [ufloat(0, 0)] * 11
                l0_tis_breakdown[track][year] = [ufloat(0, 0)] * len(L0_TIS_components)
                continue

            # L0 Efficiencies
            L0_efficiencies = []
            after_L0trigger_evts = 0
            for L0_trigger_cut in L0_trigger_cuts:
                # For L0, combine the standalone truthpkk with the trigger
                cut_string = f"{truthpkk} && {L0_trigger_cut}"
                after_L0trigger_evts = getEntries(tuple_path, "DecayTree", cutstr=cut_string)
                debug_print(f"L0 cut applied, Events: {after_L0trigger_evts}")
                _eff = caleff(after_L0trigger_evts, before_trigger_evts)
                L0_efficiencies.append(_eff)

            # L0 TIS breakdown (if option is enabled)
            if args.l0_breakdown:
                for L0_component in L0_TIS_components:
                    cut_string = f"{truthpkk} && {L0_component}"
                    component_events = getEntries(tuple_path, "DecayTree", cutstr=cut_string)
                    debug_print(f"L0 component {L0_component}: {component_events} events")
                    _eff = caleff(component_events, before_trigger_evts)
                    l0_tis_breakdown[track][year].append(_eff)

            # L1 Efficiencies
            L1_efficiencies = []
            after_L1trigger_evts = 0
            for L1_trigger_cut in L1_trigger_cuts:
                # For L1, combine L0 final cut with L1 cut
                cut_string = f"{truthpkk} && {L0_trigger_cuts[-1]} && {L1_trigger_cut}"
                after_L1trigger_evts = getEntries(tuple_path, "DecayTree", cutstr=cut_string)
                debug_print(f"L1 cut applied, Events: {after_L1trigger_evts}")
                _eff = caleff(
                    after_L1trigger_evts, after_L0trigger_evts or 1
                )  # Avoid division by zero
                L1_efficiencies.append(_eff)

            # L2 Efficiencies
            L2_efficiencies = []
            after_L2trigger_evts = 0
            for L2_trigger_cut in L2_trigger_cuts:
                # For L2, combine L0 final cut, L1 final cut with L2 cut
                cut_string = f"{truthpkk} && {L0_trigger_cuts[-1]} && {L1_trigger_cuts[-1]} && {L2_trigger_cut}"
                after_L2trigger_evts = getEntries(tuple_path, "DecayTree", cutstr=cut_string)
                debug_print(f"L2 cut applied, Events: {after_L2trigger_evts}")
                _eff = caleff(
                    after_L2trigger_evts, after_L1trigger_evts or 1
                )  # Avoid division by zero
                L2_efficiencies.append(_eff)

            efficiency[track][year].extend(L0_efficiencies)
            efficiency[track][year].extend(L1_efficiencies)
            efficiency[track][year].extend(L2_efficiencies)

            L012_eff = [L0_efficiencies[-1], L1_efficiencies[-1], L2_efficiencies[-1]]
            # Total trigger efficiency
            efficiency[track][year].append(np.prod(L012_eff))

# Define triggers
triggers = []
triggers.append("\\texttt{Bu\\_L0Global\\_TIS}")
triggers.append("\\texttt{Bu\\_L0HadronDecision\\_TOS}")
triggers.append("\\texttt{OR}")
triggers.append("\\texttt{Bu\\_Hlt1TrackMVADecision\\_TOS}")
triggers.append("\\texttt{Bu\\_Hlt1TwoTrackMVADecision\\_TOS}")
triggers.append("\\texttt{OR}")
triggers.append("\\texttt{Bu\\_Hlt2Topo2BodyDecision\\_TOS}")
triggers.append("\\texttt{Bu\\_Hlt2Topo3BodyDecision\\_TOS}")
triggers.append("\\texttt{Bu\\_Hlt2Topo4BodyDecision\\_TOS}")
triggers.append("\\texttt{OR}")
triggers.append("Total")

# Define L0 TIS component names for table
l0_tis_component_names = []
l0_tis_component_names.append("\\texttt{Bu\\_L0GlobalDecision\\_TIS}")
l0_tis_component_names.append("\\texttt{Bu\\_L0PhysDecision\\_TIS}")
l0_tis_component_names.append("\\texttt{Bu\\_L0HadronDecision\\_TIS}")
l0_tis_component_names.append("\\texttt{Bu\\_L0MuonDecision\\_TIS}")
l0_tis_component_names.append("\\texttt{Bu\\_L0MuonHighDecision\\_TIS}")
l0_tis_component_names.append("\\texttt{Bu\\_L0DiMuonDecision\\_TIS}")
l0_tis_component_names.append("\\texttt{Bu\\_L0PhotonDecision\\_TIS}")
l0_tis_component_names.append("\\texttt{Bu\\_L0ElectronDecision\\_TIS}")

# Print main trigger efficiency tables
print_table_both_formats(
    caption="Trigger efficiencies for $B^+ \\to \\bar{\\Lambda}^0_{\\text{LL}} p K^+ K^-$ selection (\\%)",
    label="tab:trigger_eff_LL",
    headers=["2016", "2017", "2018"],
    row_names=triggers,
    data=efficiency,
    track_type="LL",
)

print("\n\n")

print_table_both_formats(
    caption="Trigger efficiencies for $B^+ \\to \\bar{\\Lambda}^0_{\\text{DD}} p K^+ K^-$ selection (\\%)",
    label="tab:trigger_eff_DD",
    headers=["2016", "2017", "2018"],
    row_names=triggers,
    data=efficiency,
    track_type="DD",
)

print("\n\n")

# Print L0 TIS breakdown tables if the option is enabled
if args.l0_breakdown:
    # Add Total row data to breakdown
    l0_tis_with_total = l0_tis_component_names + ["\\texttt{Total}"]

    # Create a modified breakdown dict that includes the Total row
    l0_tis_breakdown_with_total = {}
    for track in ["LL", "DD"]:
        l0_tis_breakdown_with_total[track] = {}
        for year in ["16", "17", "18"]:
            l0_tis_breakdown_with_total[track][year] = l0_tis_breakdown[track][year] + [
                efficiency[track][year][0]
            ]

    print_table_both_formats(
        caption="$L0$ contributions for $B^+ \\to \\bar{\\Lambda}^0_{\\text{LL}} p K^+ K^-$ selection (\\%)",
        label="tab:l0_tis_breakdown_LL",
        headers=["2016", "2017", "2018"],
        row_names=l0_tis_with_total,
        data=l0_tis_breakdown_with_total,
        track_type="LL",
        is_breakdown=True,
    )

    print("\n\n")

    print_table_both_formats(
        caption="$L0$ contributions for $B^+ \\to \\bar{\\Lambda}^0_{\\text{DD}} p K^+ K^-$ selection (\\%)",
        label="tab:l0_tis_breakdown_DD",
        headers=["2016", "2017", "2018"],
        row_names=l0_tis_with_total,
        data=l0_tis_breakdown_with_total,
        track_type="DD",
        is_breakdown=True,
    )
