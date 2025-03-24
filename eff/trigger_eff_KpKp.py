import ROOT as rt
import sys
import os
import argparse
import numpy as np
from uncertainties import ufloat
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate trigger efficiencies')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output for debugging')
args = parser.parse_args()

# Define trigger cuts for each level
L0_trigger_cuts = [
    "Bu_L0Global_TIS > 0",
    "Bu_L0HadronDecision_TOS > 0",
    "(Bu_L0Global_TIS > 0 || Bu_L0HadronDecision_TOS > 0)"
]

L1_trigger_cuts = [
    "Bu_Hlt1TrackMVADecision_TOS > 0",
    "Bu_Hlt1TwoTrackMVADecision_TOS > 0",
    "(Bu_Hlt1TrackMVADecision_TOS > 0 || Bu_Hlt1TwoTrackMVADecision_TOS > 0)"
]

L2_trigger_cuts = [
    "Bu_Hlt2Topo2BodyDecision_TOS > 0",
    "Bu_Hlt2Topo3BodyDecision_TOS > 0",
    "Bu_Hlt2Topo4BodyDecision_TOS > 0",
    "(Bu_Hlt2Topo2BodyDecision_TOS > 0 || Bu_Hlt2Topo3BodyDecision_TOS > 0 || Bu_Hlt2Topo4BodyDecision_TOS > 0)"
]

# Define fixed truth cut string for B+ → Λ⁰ p̄ K+ K+
truthpkk_standalone = """(abs(Bu_TRUEID)==521) && 
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
truthpkk_standalone = truthpkk_standalone.replace('\n', ' ').replace('  ', ' ').strip()

def debug_print(*print_args, **kwargs):
    """Print only if verbose mode is enabled"""
    if args.verbose:
        print(*print_args, **kwargs)

def getEntries(tuples, tree, cutstr):
    chain = rt.TChain(tree)
    
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
    eff_err = np.sqrt(eff * (1-eff) / nGen)
    return ufloat(eff, eff_err)

efficiency = {}
# Calculate total iterations for progress bar (2 tracks × 3 years)
total_iterations = 2 * 3
with tqdm(total=total_iterations, desc="Processing files") as pbar:
    for track in ["LL", "DD"]:
        efficiency[track] = {}
        for year in ["16", "17", "18"]:
            efficiency[track][year] = []
            
            # Create explicit paths for the files - for B+ → Λ⁰ p̄ K+ K+
            base_dir = '/share/lazy/Mohamed/Bu2LambdaPPP/MC/DaVinciTuples/restripped.MC/'
            mu_file = f'{base_dir}MC{year}MUBu2L0PbarKpKp.root/B2L0PbarKpKp_{track}/DecayTree'
            md_file = f'{base_dir}MC{year}MDBu2L0PbarKpKp.root/B2L0PbarKpKp_{track}/DecayTree'
            
            # Add both files to the path list
            tuple_path = [mu_file, md_file]
            
            debug_print(f"Looking for files: {mu_file} and {md_file}")
            
            # First check if files exist and get events with truth cut
            before_trigger_evts = getEntries(tuple_path, 'DecayTree', cutstr=truthpkk_standalone)
            
            debug_print(f"Track: {track}, Year: {year}, Before trigger events: {before_trigger_evts}")
            
            if before_trigger_evts == 0:
                print(f"Warning: No events found for {track} {year} before trigger. Skipping...")
                # Fill with zeros to maintain structure
                efficiency[track][year] = [ufloat(0, 0)] * 11
                pbar.update(1)
                continue

            # L0 Efficiencies
            L0_efficiencies = []
            after_L0trigger_evts = 0
            for L0_trigger_cut in L0_trigger_cuts:
                # For L0, combine the standalone truthpkk with the trigger
                cut_string = f"{truthpkk_standalone} && {L0_trigger_cut}"
                after_L0trigger_evts = getEntries(tuple_path, 'DecayTree', cutstr=cut_string)
                debug_print(f"L0 cut applied, Events: {after_L0trigger_evts}")
                _eff = caleff(after_L0trigger_evts, before_trigger_evts)
                L0_efficiencies.append(_eff)

            # L1 Efficiencies
            L1_efficiencies = []
            after_L1trigger_evts = 0
            for L1_trigger_cut in L1_trigger_cuts:
                # For L1, combine L0 final cut with L1 cut
                cut_string = f"{truthpkk_standalone} && {L0_trigger_cuts[-1]} && {L1_trigger_cut}"
                after_L1trigger_evts = getEntries(tuple_path, 'DecayTree', cutstr=cut_string)
                debug_print(f"L1 cut applied, Events: {after_L1trigger_evts}")
                _eff = caleff(after_L1trigger_evts, after_L0trigger_evts or 1)  # Avoid division by zero
                L1_efficiencies.append(_eff)

            # L2 Efficiencies
            L2_efficiencies = []
            after_L2trigger_evts = 0
            for L2_trigger_cut in L2_trigger_cuts:
                # For L2, combine L0 final cut, L1 final cut with L2 cut
                cut_string = f"{truthpkk_standalone} && {L0_trigger_cuts[-1]} && {L1_trigger_cuts[-1]} && {L2_trigger_cut}"
                after_L2trigger_evts = getEntries(tuple_path, 'DecayTree', cutstr=cut_string)
                debug_print(f"L2 cut applied, Events: {after_L2trigger_evts}")
                _eff = caleff(after_L2trigger_evts, after_L1trigger_evts or 1)  # Avoid division by zero
                L2_efficiencies.append(_eff)

            efficiency[track][year].extend(L0_efficiencies)
            efficiency[track][year].extend(L1_efficiencies)
            efficiency[track][year].extend(L2_efficiencies)

            L012_eff = [L0_efficiencies[-1], L1_efficiencies[-1], L2_efficiencies[-1]]
            # Total trigger efficiency
            efficiency[track][year].append(np.prod(L012_eff))
            
            # Update progress bar
            pbar.update(1)

# Define triggers
triggers = []
triggers.append("\\texttt{Bu\_L0Global\_TIS}")
triggers.append("\\texttt{Bu\_L0HadronDecision\_TOS}")
triggers.append("\\texttt{OR}")
triggers.append("\\texttt{Bu\_Hlt1TrackMVADecision\_TOS}")
triggers.append("\\texttt{Bu\_Hlt1TwoTrackMVADecision\_TOS}")
triggers.append("\\texttt{OR}")
triggers.append("\\texttt{Bu\_Hlt2Topo2BodyDecision\_TOS}")
triggers.append("\\texttt{Bu\_Hlt2Topo3BodyDecision\_TOS}")
triggers.append("\\texttt{Bu\_Hlt2Topo4BodyDecision\_TOS}")
triggers.append("\\texttt{OR}")
triggers.append("Total")

print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Trigger efficiencies for $B^+ \\to \\Lambda^0_{\\text{LL}} \\bar{p} K^+ K^+$ selection (\\%)}")
print("\\begin{tabular}{l|ccc}")
print("\\hline")
print("Levels & 2016 & 2017 & 2018 \\\\ \\hline")

for i in range(11):
    eff_str = ""
    for track in ["LL"]:
        for year in ["16", "17", "18"]:
            if i < len(efficiency[track][year]):
                eff = round(efficiency[track][year][i].nominal_value*100, 2)
                eff_err = round(efficiency[track][year][i].std_dev*100, 2)
                if "OR" in triggers[i]:
                    if year == "18":
                        eff_str += f" & $\\bm{{{eff}\\pm{eff_err}}}$ \\\\ \\hline"
                    else:
                        eff_str += f" & $\\bm{{{eff}\\pm{eff_err}}}$"
                else:
                    if year == "18":
                        eff_str += f" & ${eff}\\pm{eff_err}$ \\\\"
                    else:
                        eff_str += f" & ${eff}\\pm{eff_err}$"
            else:
                if year == "18":
                    eff_str += f" & $0.00\\pm0.00$ \\\\"
                else:
                    eff_str += f" & $0.00\\pm0.00$"

    print(f"{triggers[i]} {eff_str} ")

print("\\hline")
print("\\end{tabular}")
print("\\label{tab:trigger_eff_LL_pbar}")
print("\\end{table}")

print("\n\n")

print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Trigger efficiencies for $B^+ \\to \\Lambda^0_{\\text{DD}} \\bar{p} K^+ K^+$ selection (\\%)}")
print("\\begin{tabular}{l|ccc}")
print("\\hline")
print("Levels & 2016 & 2017 & 2018 \\\\ \\hline")

for i in range(11):
    eff_str = ""
    for track in ["DD"]:
        for year in ["16", "17", "18"]:
            if i < len(efficiency[track][year]):
                eff = round(efficiency[track][year][i].nominal_value*100, 2)
                eff_err = round(efficiency[track][year][i].std_dev*100, 2)
                if "OR" in triggers[i]:
                    if year == "18":
                        eff_str += f" & $\\bm{{{eff}\\pm{eff_err}}}$ \\\\ \\hline"
                    else:
                        eff_str += f" & $\\bm{{{eff}\\pm{eff_err}}}$"
                else:
                    if year == "18":
                        eff_str += f" & ${eff}\\pm{eff_err}$ \\\\"
                    else:
                        eff_str += f" & ${eff}\\pm{eff_err}$"
            else:
                if year == "18":
                    eff_str += f" & $0.00\\pm0.00$ \\\\"
                else:
                    eff_str += f" & $0.00\\pm0.00$"

    print(f"{triggers[i]} {eff_str} ")

print("\\hline")
print("\\end{tabular}")
print("\\label{tab:trigger_eff_DD_pbar}")
print("\\end{table}")

print("\n\n")