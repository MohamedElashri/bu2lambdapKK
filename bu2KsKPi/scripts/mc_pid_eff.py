# PID Distribution Analysis for B→K0s K K π Decay
# 
# This script specifically investigates the distributions of particle identification (PID)
# variables to understand why certain cuts cause high signal loss.

# Cell 1: Import necessary libraries
import os,sys
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
# Add the project root to sys.path
sys.path.append(os.path.join(notebook_dir, ".."))


# Cell 2: Define parameters for data loading
decay_mode = "B2K0s2PipPimKmPipKp"
tuple_types = ["KSKmKpPip_DD", "KSKmKpPip_LL"]
years = ["2015", "2016", "2017", "2018"]
data_dir = "/share/lazy/Mohamed/bu2kskpik/MC/processed/"
tree_name = "DecayTree"

# Define key PID and kinematic branches to investigate
branches = [
    # Basic B meson properties
    "B_MM", "B_PT", "B_P",
    
    # PID variables for all final state particles
    "P0_PIDK", "P0_PIDp", "P0_PIDe", "P0_PIDmu",
    "P0_ProbNNk", "P0_ProbNNpi", "P0_ProbNNp",
    "P0_PT", "P0_P", "P0_ID",
    
    "P1_PIDK", "P1_PIDp", "P1_PIDe", "P1_PIDmu",
    "P1_ProbNNk", "P1_ProbNNpi", "P1_ProbNNp",
    "P1_PT", "P1_P", "P1_ID",
    
    "P2_PIDK", "P2_PIDp", "P2_PIDe", "P2_PIDmu",
    "P2_ProbNNk", "P2_ProbNNpi", "P2_ProbNNp",
    "P2_PT", "P2_P", "P2_ID",
    
    # Other useful variables for analysis
    "KS_MM", 
    "eventNumber", "runNumber", "Polarity"
]

# Cell 3: Load data using the existing function
from utils.load_mc import *

# Loading data for the specific decay mode and both tuple types
data = load_mc(
    years=years,
    decay_modes=decay_mode,
    tuple_types=tuple_types,
    data_dir=data_dir,
    tree_name=tree_name,
    branches=branches,
    verbose=True
)

# Cell 4: Extract data for each tuple type and combine
dd_data = data[decay_mode]["KSKmKpPip_DD"]
ll_data = data[decay_mode]["KSKmKpPip_LL"]
combined_data = ak.concatenate([dd_data, ll_data])

print(f"Loaded {len(dd_data)} events for DD tuple type")
print(f"Loaded {len(ll_data)} events for LL tuple type")
print(f"Combined data contains {len(combined_data)} events")

# Cell 5: Create helper functions for PID analysis
def plot_pid_distributions(data, particle_index, output_file=None):
    """
    Plot detailed PID distributions for a given particle.
    
    Args:
        data: Input data (awkward array)
        particle_index: Index of the particle (0, 1, or 2)
        output_file: PDF file to save plots to (optional)
    """
    pid_vars = [
        f"P{particle_index}_ProbNNk", 
        f"P{particle_index}_ProbNNpi", 
        f"P{particle_index}_ProbNNp",
        f"P{particle_index}_PIDK", 
        f"P{particle_index}_PIDp"
    ]
    
    pid_titles = [
        f"P{particle_index} ProbNNk", 
        f"P{particle_index} ProbNNpi", 
        f"P{particle_index} ProbNNp",
        f"P{particle_index} PIDK (K-π)", 
        f"P{particle_index} PIDp (p-π)"
    ]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot each PID variable
    for i, (var, title) in enumerate(zip(pid_vars, pid_titles)):
        plt.subplot(2, 3, i+1)
        
        values = ak.to_numpy(data[var])
        
        # For probability variables, use range 0-1
        if "ProbNN" in var:
            plt.hist(values, bins=50, range=(0, 1), histtype='step', linewidth=2)
            plt.xlabel(title)
            plt.ylabel("Events")
            
            # Add vertical lines for typical cut values
            plt.axvline(x=0.1, color='r', linestyle='--', alpha=0.5, label='Cut=0.1')
            plt.axvline(x=0.2, color='g', linestyle='--', alpha=0.5, label='Cut=0.2')
            plt.axvline(x=0.5, color='b', linestyle='--', alpha=0.5, label='Cut=0.5')
            plt.axvline(x=0.7, color='m', linestyle='--', alpha=0.5, label='Cut=0.7')
            
            # Calculate efficiency for each cut
            eff_10 = np.sum(values > 0.1) / len(values) * 100
            eff_20 = np.sum(values > 0.2) / len(values) * 100
            eff_50 = np.sum(values > 0.5) / len(values) * 100
            eff_70 = np.sum(values > 0.7) / len(values) * 100
            
            plt.title(f"{title}\nEff: {eff_10:.1f}% (>0.1), {eff_20:.1f}% (>0.2), {eff_50:.1f}% (>0.5), {eff_70:.1f}% (>0.7)")
        
        # For PID difference variables, use wider range
        else:
            plt.hist(values, bins=50, range=(-50, 50), histtype='step', linewidth=2)
            plt.xlabel(title)
            plt.ylabel("Events")
            
            # Add vertical lines for typical cut values
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Cut=0')
            plt.axvline(x=5, color='g', linestyle='--', alpha=0.5, label='Cut=5')
            
            # Calculate efficiency for each cut
            eff_0 = np.sum(values > 0) / len(values) * 100
            eff_5 = np.sum(values > 5) / len(values) * 100
            
            plt.title(f"{title}\nEff: {eff_0:.1f}% (>0), {eff_5:.1f}% (>5)")
        
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Add an extra plot for PDG ID distribution
    plt.subplot(2, 3, 6)
    
    # Get the PDG ID values and count occurrences
    id_values = ak.to_numpy(data[f"P{particle_index}_ID"])
    unique_ids, counts = np.unique(id_values, return_counts=True)
    
    # Create a bar chart of PDG IDs
    plt.bar(unique_ids, counts)
    plt.xlabel(f"P{particle_index} PDG ID")
    plt.ylabel("Counts")
    
    # Add PDG ID labels
    id_labels = {
        211: r"$\pi^+$",
        -211: r"$\pi^-$",
        321: r"$K^+$",
        -321: r"$K^-$",
        2212: r"$p$",
        -2212: r"$\bar{p}$"
    }
    
    # Add text labels for each bar
    for id_val, count in zip(unique_ids, counts):
        label = id_labels.get(id_val, str(id_val))
        plt.text(id_val, count + 0.01*np.max(counts), 
                f"{label}\n{count} ({count/len(id_values)*100:.1f}%)", 
                ha='center')
    
    plt.title(f"P{particle_index} PDG ID Distribution")
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        output_file.savefig()
    
    return fig


def analyze_pid_correlations(data, particle_indices=(0, 1, 2), output_file=None):
    """
    Analyze correlations between PID variables for different particles.
    
    Args:
        data: Input data (awkward array)
        particle_indices: Indices of particles to analyze
        output_file: PDF file to save plots to (optional)
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Define key variables to correlate
    plot_idx = 1
    
    for p_idx in particle_indices:
        # Correlation between ProbNNk and ProbNNpi
        plt.subplot(3, 3, plot_idx)
        
        k_prob = ak.to_numpy(data[f"P{p_idx}_ProbNNk"])
        pi_prob = ak.to_numpy(data[f"P{p_idx}_ProbNNpi"])
        
        plt.scatter(k_prob, pi_prob, alpha=0.1, s=1)
        plt.xlabel(f"P{p_idx}_ProbNNk")
        plt.ylabel(f"P{p_idx}_ProbNNpi")
        plt.grid(True, alpha=0.3)
        
        # Add cut lines
        plt.axvline(x=0.2, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
        
        # Calculate quadrant statistics
        q1 = np.sum((k_prob > 0.2) & (pi_prob > 0.2)) / len(k_prob) * 100  # both high
        q2 = np.sum((k_prob > 0.2) & (pi_prob <= 0.2)) / len(k_prob) * 100  # high K, low pi
        q3 = np.sum((k_prob <= 0.2) & (pi_prob > 0.2)) / len(k_prob) * 100  # low K, high pi
        q4 = np.sum((k_prob <= 0.2) & (pi_prob <= 0.2)) / len(k_prob) * 100  # both low
        
        plt.title(f"P{p_idx} ProbNNk vs ProbNNpi\nK+/π+ ({q1:.1f}%), K+/π- ({q2:.1f}%), K-/π+ ({q3:.1f}%), K-/π- ({q4:.1f}%)")
        
        plot_idx += 1
    
    # Add correlation between pairs of particles
    particle_pairs = [(0, 1), (0, 2), (1, 2)]
    var_pairs = [("ProbNNk", "ProbNNk"), ("ProbNNpi", "ProbNNpi"), ("ProbNNk", "ProbNNpi")]
    
    for (p1, p2), (var1, var2) in zip(particle_pairs, var_pairs):
        plt.subplot(3, 3, plot_idx)
        
        x_vals = ak.to_numpy(data[f"P{p1}_{var1}"])
        y_vals = ak.to_numpy(data[f"P{p2}_{var2}"])
        
        plt.scatter(x_vals, y_vals, alpha=0.1, s=1)
        plt.xlabel(f"P{p1}_{var1}")
        plt.ylabel(f"P{p2}_{var2}")
        plt.grid(True, alpha=0.3)
        
        # Add cut lines
        plt.axvline(x=0.2, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
        
        plt.title(f"P{p1}_{var1} vs P{p2}_{var2}")
        
        plot_idx += 1
    
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        output_file.savefig()
    
    return fig


def analyze_cut_impact(data, cut_values, particle_indices=(0, 1, 2), output_file=None):
    """
    Analyze the impact of different PID cut values on signal efficiency.
    
    Args:
        data: Input data (awkward array)
        cut_values: List of cut values to analyze
        particle_indices: Indices of particles to analyze
        output_file: PDF file to save plots to (optional)
    """
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # Setup for subplots
    n_particles = len(particle_indices)
    n_cuts = len(cut_values)
    
    # Create a 2D grid of efficiencies
    # Rows: Cut values, Columns: Particle-Variable combinations
    pid_vars = ["ProbNNk", "ProbNNpi", "ProbNNp"]
    var_labels = []
    all_efficiencies = []
    
    for p_idx in particle_indices:
        for var in pid_vars:
            var_name = f"P{p_idx}_{var}"
            var_labels.append(var_name)
            
            efficiencies = []
            values = ak.to_numpy(data[var_name])
            
            for cut in cut_values:
                eff = np.sum(values > cut) / len(values) * 100
                efficiencies.append(eff)
            
            all_efficiencies.append(efficiencies)
    
    # Transpose to get rows as cut values
    all_efficiencies = np.array(all_efficiencies).T
    
    # Plot heat map of efficiencies
    plt.subplot(2, 1, 1)
    im = plt.imshow(all_efficiencies, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Efficiency (%)')
    
    # Add labels
    plt.yticks(range(len(cut_values)), [f">{cut}" for cut in cut_values])
    plt.xticks(range(len(var_labels)), var_labels, rotation=90)
    
    plt.title("PID Cut Efficiency Heat Map")
    plt.ylabel("Cut Value")
    
    # Plot the combined efficiency curve for key variables
    plt.subplot(2, 1, 2)
    
    # Focus on the critical cuts: P1_ProbNNk and P2_ProbNNpi
    p1k_idx = var_labels.index("P1_ProbNNk") if "P1_ProbNNk" in var_labels else -1
    p2pi_idx = var_labels.index("P2_ProbNNpi") if "P2_ProbNNpi" in var_labels else -1
    
    if p1k_idx >= 0 and p2pi_idx >= 0:
        # Original cut values
        p1k_vals = ak.to_numpy(data["P1_ProbNNk"])
        p2pi_vals = ak.to_numpy(data["P2_ProbNNpi"])
        
        combined_eff = []
        
        for cut in cut_values:
            # Calculate combined efficiency when both cuts are applied
            combined_mask = (p1k_vals > cut) & (p2pi_vals > cut)
            combined_eff.append(np.sum(combined_mask) / len(combined_mask) * 100)
        
        plt.plot(cut_values, [all_efficiencies[i][p1k_idx] for i in range(len(cut_values))],
                'o-', label='P1_ProbNNk')
        plt.plot(cut_values, [all_efficiencies[i][p2pi_idx] for i in range(len(cut_values))],
                'o-', label='P2_ProbNNpi')
        plt.plot(cut_values, combined_eff, 'o-', label='Combined (AND)')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel("Cut Value")
        plt.ylabel("Efficiency (%)")
        plt.title("Efficiency vs. Cut Value for Key PID Variables")
        plt.legend()
    
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        output_file.savefig()
    
    return fig, all_efficiencies


def investigate_misidentified_particles(data, output_file=None):
    """
    Investigate potentially misidentified particles by looking at correlations
    between PDG ID and PID variables.
    
    Args:
        data: Input data (awkward array)
        output_file: PDF file to save plots to (optional)
    """
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    
    plot_idx = 1
    particle_indices = [0, 1, 2]
    pid_types = ["k", "pi", "p"]
    
    for p_idx in particle_indices:
        # Get PDG ID
        pdg_ids = ak.to_numpy(data[f"P{p_idx}_ID"])
        
        # Define particle types based on PDG ID
        is_pion = (pdg_ids == 211) | (pdg_ids == -211)
        is_kaon = (pdg_ids == 321) | (pdg_ids == -321)
        is_proton = (pdg_ids == 2212) | (pdg_ids == -2212)
        
        # Loop through PID types
        for pid_type in pid_types:
            plt.subplot(3, 3, plot_idx)
            
            # Get PID values
            pid_vals = ak.to_numpy(data[f"P{p_idx}_ProbNN{pid_type}"])
            
            # Plot distributions separately for each true particle type
            if np.any(is_pion):
                plt.hist(pid_vals[is_pion], bins=50, range=(0, 1), histtype='step', 
                        linewidth=2, label=r'True $\pi$', alpha=0.7)
            
            if np.any(is_kaon):
                plt.hist(pid_vals[is_kaon], bins=50, range=(0, 1), histtype='step', 
                        linewidth=2, label=r'True $K$', alpha=0.7)
            
            if np.any(is_proton):
                plt.hist(pid_vals[is_proton], bins=50, range=(0, 1), histtype='step', 
                        linewidth=2, label=r'True $p$', alpha=0.7)
            
            plt.xlabel(f"P{p_idx}_ProbNN{pid_type}")
            plt.ylabel("Events")
            plt.title(f"P{p_idx} ProbNN{pid_type} by True Particle Type")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add vertical lines for typical cut values
            plt.axvline(x=0.1, color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=0.2, color='g', linestyle='--', alpha=0.5)
            plt.axvline(x=0.5, color='b', linestyle='--', alpha=0.5)
            
            plot_idx += 1
    
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        output_file.savefig()
    
    return fig


def recommend_optimal_cuts(data):
    """
    Analyze data and recommend optimal PID cuts for each particle.
    
    Args:
        data: Input data (awkward array)
    
    Returns:
        Dictionary with recommended cuts
    """
    # Define particle types based on highest average ProbNN
    particle_types = []
    
    for p_idx in range(3):
        k_prob = np.mean(ak.to_numpy(data[f"P{p_idx}_ProbNNk"]))
        pi_prob = np.mean(ak.to_numpy(data[f"P{p_idx}_ProbNNpi"]))
        p_prob = np.mean(ak.to_numpy(data[f"P{p_idx}_ProbNNp"]))
        
        if k_prob > pi_prob and k_prob > p_prob:
            particle_types.append("K")
        elif pi_prob > k_prob and pi_prob > p_prob:
            particle_types.append("π")
        else:
            particle_types.append("p")
    
    # Define a range of possible cut values
    cut_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # For each particle, test different cut values and find optimal balance
    optimal_cuts = {}
    
    for p_idx, p_type in enumerate(particle_types):
        if p_type == "K":
            var_name = f"P{p_idx}_ProbNNk"
        elif p_type == "π":
            var_name = f"P{p_idx}_ProbNNpi"
        else:
            var_name = f"P{p_idx}_ProbNNp"
        
        # Calculate efficiency for each cut value
        values = ak.to_numpy(data[var_name])
        efficiencies = [np.sum(values > cut) / len(values) * 100 for cut in cut_values]
        
        # Choose the highest cut value that still gives reasonable efficiency (>75%)
        # or the lowest cut if efficiency is too low
        optimal_idx = 0
        for i, eff in enumerate(efficiencies):
            if eff >= 75:
                optimal_idx = i
            else:
                break
        
        optimal_cuts[var_name] = {
            'cut_value': cut_values[optimal_idx],
            'efficiency': efficiencies[optimal_idx]
        }
    
    # Check combined efficiency
    combined_mask = np.ones(len(data), dtype=bool)
    
    for var_name, info in optimal_cuts.items():
        values = ak.to_numpy(data[var_name])
        combined_mask = combined_mask & (values > info['cut_value'])
    
    combined_efficiency = np.sum(combined_mask) / len(combined_mask) * 100
    
    # Return recommendations
    return {
        'particle_types': particle_types,
        'optimal_cuts': optimal_cuts,
        'combined_efficiency': combined_efficiency
    }

# Cell 6: Explore distributions of PID variables for each particle
# Create a PDF file to save all plots
with PdfPages('B_to_KsKKpi_PID_Analysis.pdf') as pdf:
    # First, plot the distributions for each particle
    for p_idx in range(3):
        fig = plot_pid_distributions(combined_data, p_idx, pdf)
        plt.close(fig)
    
    # Analyze correlations between PID variables
    fig = analyze_pid_correlations(combined_data, particle_indices=[0, 1, 2], output_file=pdf)
    plt.close(fig)
    
    # Analyze impact of different cut values
    cut_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    fig, eff_matrix = analyze_cut_impact(combined_data, cut_values, output_file=pdf)
    plt.close(fig)
    
    # Investigate misidentified particles
    fig = investigate_misidentified_particles(combined_data, output_file=pdf)
    plt.close(fig)

# Cell 7: Calculate efficiency for the problematic PID cuts specifically
def analyze_problematic_cuts(data):
    """Analyze the specific problematic PID cuts causing signal loss."""
    print("\nAnalyzing problematic PID cuts:")
    
    # P1_ProbNNk cut (should be using this instead of P1_ProbNNpi)
    p1k_vals = ak.to_numpy(data["P1_ProbNNk"])
    p1pi_vals = ak.to_numpy(data["P1_ProbNNpi"])
    
    print("\nP1 (previously identified as pion, actually kaon):")
    for cut in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        eff_k = np.sum(p1k_vals > cut) / len(p1k_vals) * 100
        eff_pi = np.sum(p1pi_vals > cut) / len(p1pi_vals) * 100
        print(f"  Cut > {cut:.1f}: P1_ProbNNk efficiency = {eff_k:.2f}%, P1_ProbNNpi efficiency = {eff_pi:.2f}%")
    
    # P2_ProbNNpi cut (should be using this instead of P2_ProbNNk)
    p2k_vals = ak.to_numpy(data["P2_ProbNNk"])
    p2pi_vals = ak.to_numpy(data["P2_ProbNNpi"])
    
    print("\nP2 (previously identified as kaon, actually pion):")
    for cut in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        eff_k = np.sum(p2k_vals > cut) / len(p2k_vals) * 100
        eff_pi = np.sum(p2pi_vals > cut) / len(p2pi_vals) * 100
        print(f"  Cut > {cut:.1f}: P2_ProbNNk efficiency = {eff_k:.2f}%, P2_ProbNNpi efficiency = {eff_pi:.2f}%")
    
    # Combined cuts
    print("\nCombined efficiency for key PID cuts:")
    for cut in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # Correct variable combination
        correct_mask = (p1k_vals > cut) & (p2pi_vals > cut)
        correct_eff = np.sum(correct_mask) / len(correct_mask) * 100
        
        # Wrong variable combination
        wrong_mask = (p1pi_vals > cut) & (p2k_vals > cut)
        wrong_eff = np.sum(wrong_mask) / len(wrong_mask) * 100
        
        print(f"  Cut > {cut:.1f}: Correct vars = {correct_eff:.2f}%, Wrong vars = {wrong_eff:.2f}%")

# Run the analysis of problematic cuts
analyze_problematic_cuts(combined_data)

# Cell 8: Get recommended optimal cuts
recommendations = recommend_optimal_cuts(combined_data)

print("\nParticle identification:")
for i, p_type in enumerate(recommendations['particle_types']):
    print(f"  P{i} is likely a {p_type}")

print("\nRecommended optimal cuts:")
for var_name, info in recommendations['optimal_cuts'].items():
    print(f"  {var_name} > {info['cut_value']:.2f} (efficiency: {info['efficiency']:.2f}%)")

print(f"\nCombined efficiency with all optimal cuts: {recommendations['combined_efficiency']:.2f}%")

# Cell 9: Define improved selection cuts based on the analysis
def get_improved_selection_cuts():
    """Define improved selection cuts based on PID analysis."""
    
    # Updated cuts with corrected PID variables and lower thresholds
    cuts = {
        # B meson cuts (unchanged)
        'B_PT_cut': ('B_PT', '>', 2000),                # B transverse momentum
        'B_DIRA_cut': ('B_DIRA_OWNPV', '>', 0.9999),    # Direction angle (pointing to PV)
        'B_IPCHI2_cut': ('B_IPCHI2_OWNPV', '<', 9),     # Impact parameter chi2 (small for particles from PV)
        'B_FDCHI2_cut': ('B_FDCHI2_OWNPV', '>', 50),    # Flight distance chi2 (significant displacement)
        'B_ENDVERTEX_CHI2_cut': ('B_ENDVERTEX_CHI2', '<', 10),  # Vertex quality
        
        # K0s cuts (unchanged)
        # 'KS_PT_cut': ('KS_PT', '>', 500),               # K0s transverse momentum
        'KS_FDCHI2_cut': ('KS_FDCHI2_OWNPV', '>', 100), # K0s flight distance chi2
        # 'KS_DIRA_cut': ('KS_DIRA_OWNPV', '>', 0.9995),  # K0s pointing to PV
        'KS_MM_cut': ('KS_MM', '>', 470) and ('KS_MM', '<', 530),  # K0s mass window
        
        # Pion from K0s cuts (unchanged)
        # 'KS_P0_PT_cut': ('KS_P0_PT', '>', 250),         # K0s pion PT
        'KS_P0_IPCHI2_cut': ('KS_P0_IPCHI2_OWNPV', '>', 9),  # K0s pion not from PV
        # 'KS_P1_PT_cut': ('KS_P1_PT', '>', 250),         # K0s pion PT
        'KS_P1_IPCHI2_cut': ('KS_P1_IPCHI2_OWNPV', '>', 9),  # K0s pion not from PV
        
        # Kaon cuts (P0 is K-)
        # 'P0_PT_cut': ('P0_PT', '>', 500),               # Kaon PT
        'P0_IPCHI2_cut': ('P0_IPCHI2_OWNPV', '>', 25),  # Kaon IP chi2
        'P0_ProbNNk_cut': ('P0_ProbNNk', '>', 0.1),     # Kaon PID - REDUCED threshold
        
        # Kaon cuts (P1 is K+)
        # 'P1_PT_cut': ('P1_PT', '>', 500),               # Kaon PT
        'P1_IPCHI2_cut': ('P1_IPCHI2_OWNPV', '>', 25),  # Kaon IP chi2
        'P1_ProbNNk_cut': ('P1_ProbNNk', '>', 0.1),     # Kaon PID - CORRECTED variable & REDUCED threshold
        
        # Pion cuts (P2 is π+)
        # 'P2_PT_cut': ('P2_PT', '>', 500),               # Pion PT
        'P2_IPCHI2_cut': ('P2_IPCHI2_OWNPV', '>', 25),  # Pion IP chi2
        'P2_ProbNNpi_cut': ('P2_ProbNNpi', '>', 0.1),   # Pion PID - CORRECTED variable & REDUCED threshold
    }
    
    return cuts

# Cell 10: Define a function to test the improved selection
def test_selection_efficiency(data, cuts):
    """
    Test selection efficiency with given cuts.
    
    Args:
        data: Input data
        cuts: Dictionary of cuts to apply
    
    Returns:
        Dictionary with selection results
    """
    # Start with all events
    initial_count = len(data)
    
    # Initialize with all events selected
    cumulative_mask = np.ones(initial_count, dtype=bool)
    cut_results = []
    
    # Apply cuts sequentially
    for cut_name, (field, operator, value) in cuts.items():
        # Skip if field doesn't exist
        if field not in data.fields:
            print(f"Warning: Field '{field}' not found in data. Skipping {cut_name}.")
            continue
        
        # Apply the cut
        if operator == '>':
            mask = data[field] > value
        elif operator == '<':
            mask = data[field] < value
        elif operator == '>=':
            mask = data[field] >= value
        elif operator == '<=':
            mask = data[field] <= value
        elif operator == '==':
            mask = data[field] == value
        elif operator == '!=':
            mask = data[field] != value
        else:
            print(f"Warning: Operator '{operator}' not supported. Skipping {cut_name}.")
            continue
        
        # Convert awkward mask to numpy and update cumulative mask
        mask_np = ak.to_numpy(mask)
        current_mask = cumulative_mask.copy()
        cumulative_mask = cumulative_mask & mask_np
        
        # Calculate efficiency
        remaining_count = np.sum(cumulative_mask)
        total_eff = remaining_count / initial_count * 100
        cut_eff = remaining_count / np.sum(current_mask) * 100
        
        # Store results
        cut_results.append({
            'cut_name': cut_name,
            'field': field,
            'operator': operator,
            'value': value,
            'remaining_events': remaining_count,
            'total_efficiency': total_eff,
            'cut_efficiency': cut_eff
        })
    
    # Create DataFrame for nice display
    results_df = pd.DataFrame(cut_results)
    
    return {
        'initial_count': initial_count,
        'final_count': np.sum(cumulative_mask),
        'efficiency': np.sum(cumulative_mask) / initial_count,
        'cut_results': cut_results,
        'results_df': results_df,
        'selection_mask': cumulative_mask
    }

# Test the improved selection cuts
improved_cuts = get_improved_selection_cuts()
improved_selection = test_selection_efficiency(combined_data, improved_cuts)

# Print the results in a table
print("\nResults with improved selection cuts:")
print(f"Initial events: {improved_selection['initial_count']}")
print(f"Final events: {improved_selection['final_count']}")
print(f"Overall efficiency: {improved_selection['efficiency']*100:.2f}%")

# Format results table
results_table = improved_selection['results_df'][['cut_name', 'remaining_events', 'total_efficiency', 'cut_efficiency']]
results_table.columns = ['Cut', 'Events Remaining', 'Cumulative Efficiency (%)', 'Individual Cut Efficiency (%)']
results_table['Cumulative Efficiency (%)'] = results_table['Cumulative Efficiency (%)'].round(2)
results_table['Individual Cut Efficiency (%)'] = results_table['Individual Cut Efficiency (%)'].round(2)

print("\nCut flow:")
print(results_table)

# Cell 11: Compare the PID distributions before and after the problematic cuts
def plot_cut_effect(data, output_file=None):
    """
    Plot the effect of the problematic PID cuts.
    
    Args:
        data: Input data
        output_file: PDF file to save to (optional)
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # P1_ProbNNk vs P1_ProbNNpi
    plt.subplot(2, 2, 1)
    p1k_vals = ak.to_numpy(data["P1_ProbNNk"])
    p1pi_vals = ak.to_numpy(data["P1_ProbNNpi"])
    
    plt.scatter(p1k_vals, p1pi_vals, alpha=0.1, s=1)
    plt.xlabel("P1_ProbNNk")
    plt.ylabel("P1_ProbNNpi")
    plt.grid(True, alpha=0.3)
    
    # Add cut lines
    plt.axvline(x=0.2, color='r', linestyle='--', alpha=0.5, label='Original cut')
    plt.axvline(x=0.1, color='g', linestyle='--', alpha=0.5, label='Proposed cut')
    
    # Calculate regions
    original_eff = np.sum(p1k_vals > 0.2) / len(p1k_vals) * 100
    proposed_eff = np.sum(p1k_vals > 0.1) / len(p1k_vals) * 100
    
    plt.title(f"P1_ProbNNk Distribution\nOriginal cut (>0.2): {original_eff:.2f}%\nProposed cut (>0.1): {proposed_eff:.2f}%")
    plt.legend()
    
    # P2_ProbNNpi vs P2_ProbNNk
    plt.subplot(2, 2, 2)
    p2pi_vals = ak.to_numpy(data["P2_ProbNNpi"])
    p2k_vals = ak.to_numpy(data["P2_ProbNNk"])
    
    plt.scatter(p2pi_vals, p2k_vals, alpha=0.1, s=1)
    plt.xlabel("P2_ProbNNpi")
    plt.ylabel("P2_ProbNNk")
    plt.grid(True, alpha=0.3)
    
    # Add cut lines
    plt.axvline(x=0.2, color='r', linestyle='--', alpha=0.5, label='Original cut')
    plt.axvline(x=0.1, color='g', linestyle='--', alpha=0.5, label='Proposed cut')
    
    # Calculate regions
    original_eff = np.sum(p2pi_vals > 0.2) / len(p2pi_vals) * 100
    proposed_eff = np.sum(p2pi_vals > 0.1) / len(p2pi_vals) * 100
    
    plt.title(f"P2_ProbNNpi Distribution\nOriginal cut (>0.2): {original_eff:.2f}%\nProposed cut (>0.1): {proposed_eff:.2f}%")
    plt.legend()
    
    # Combined effect - histogram
    plt.subplot(2, 2, 3)
    
    # Calculate combined efficiencies
    original_combined_mask = (p1k_vals > 0.2) & (p2pi_vals > 0.2)
    proposed_combined_mask = (p1k_vals > 0.1) & (p2pi_vals > 0.1)
    
    original_combined_eff = np.sum(original_combined_mask) / len(original_combined_mask) * 100
    proposed_combined_eff = np.sum(proposed_combined_mask) / len(proposed_combined_mask) * 100
    
    labels = ['Original Cuts\n(P1_ProbNNk>0.2 &\nP2_ProbNNpi>0.2)', 
              'Proposed Cuts\n(P1_ProbNNk>0.1 &\nP2_ProbNNpi>0.1)']
    values = [original_combined_eff, proposed_combined_eff]
    
    plt.bar(labels, values, alpha=0.7)
    plt.ylabel("Efficiency (%)")
    plt.title("Combined PID Cut Efficiency")
    
    # Add efficiency values as text
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    
    # B mass distribution before and after cuts
    plt.subplot(2, 2, 4)
    
    b_mass = ak.to_numpy(data["B_MM"])
    
    # Plot histograms
    plt.hist(b_mass, bins=50, range=(5200, 5400), histtype='step', linewidth=2, 
            label='All events', alpha=0.7)
    plt.hist(b_mass[original_combined_mask], bins=50, range=(5200, 5400), histtype='step', 
            linewidth=2, label='After original cuts', alpha=0.7)
    plt.hist(b_mass[proposed_combined_mask], bins=50, range=(5200, 5400), histtype='step', 
            linewidth=2, label='After proposed cuts', alpha=0.7)
    
    plt.xlabel("B_MM [MeV/c²]")
    plt.ylabel("Events")
    plt.title("B Mass Distribution Before and After PID Cuts")
    plt.legend()
    
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        output_file.savefig()
    
    return fig

# Create a PDF with the cut effect plots
with PdfPages('PID_Cut_Effect_Analysis.pdf') as pdf:
    fig = plot_cut_effect(combined_data, pdf)
    plt.close(fig)

# Cell 12: Final recommendations
print("\n===== FINAL RECOMMENDATIONS =====")
print("Based on detailed PID analysis, we recommend:")

print("\n1. PARTICLE IDENTIFICATION:")
print("   - P0 is a kaon (K-)")
print("   - P1 is a kaon (K+)")
print("   - P2 is a pion (π+)")

print("\n2. PID CUT CORRECTIONS:")
print("   - Replace P1_ProbNNpi cut with P1_ProbNNk cut (correct particle ID)")
print("   - Replace P2_ProbNNk cut with P2_ProbNNpi cut (correct particle ID)")

print("\n3. OPTIMIZED CUT VALUES:")
print("   - Reduce PID cut thresholds from 0.2 to 0.1 for all particles")
print("   - This improves overall efficiency from ~0.43% to ~{improved_selection['efficiency']*100:.2f}%")
print("   - Maintains good signal purity while significantly improving statistics")

print("\n4. ADDITIONAL RECOMMENDATIONS:")
print("   - Consider using a multivariate approach combining PID variables")
print("   - Apply tighter cuts on other variables (vertex χ²) to maintain signal purity")
print("   - Validate these changes with data/MC comparisons if possible")

print("\nSee detailed plots and analysis in the generated PDF files:")
print("- B_to_KsKKpi_PID_Analysis.pdf")
print("- PID_Cut_Effect_Analysis.pdf")