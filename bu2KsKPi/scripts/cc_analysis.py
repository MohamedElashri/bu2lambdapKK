"""
Correlated Channels (CC) Analysis for B -> K0s K- π+ K+ decay.

This script focuses on analyzing the K0s K- π+ system for potential charmonium
contributions and other resonances in the three-body combinations.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import ROOT
from ROOT import TCanvas, TH1F, TF1, gStyle, kRed, kBlue, kGreen, TLegend

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import loading function from utils
from utils.load_mc import load_mc, calculate_invariant_mass
# Import plotting utilities
from utils.plot import (
    create_lhcb_figure, finalize_lhcb_figure, plot_histogram, 
    plot_multiple_histograms, plot_2d_histogram, fit_histogram_with_pyroot,
    fit_with_roofit, plot_cc_regions
)

# Configure decay modes and data loading parameters
DECAY_MODES = {
    "B2K0s2PipPimKmPipKp": "B+ → (K0_S → π+π-)K-π+K+",
    "B2Jpsi2K0s2PipPimKmPipKp": "B+ → (J/ψ → (K0_S → π+π-)K-π+)K+",
    "B2Etac2K0s2PipPimKmPipKp": "B+ → (ηc → (K0_S → π+π-)K-π+)K+",
    "B2Etac2S2K0s2PipPimKmPipKp": "B+ → (ηc(2S) → (K0_S → π+π-)K-π+)K+",
    "B2Chic12K0s2PipPimKmPipKp": "B+ → (χc1 → (K0_S → π+π-)K-π+)K+"
}

# Define resonance regions
RESONANCES = {
    "J/ψ": {
        "mass": 3097,    # MeV/c²
        "width": 0.093,  # MeV/c²
        "spin": 1,
        "window": (3050, 3150),
        "color": "red"
    },
    "ηc": {
        "mass": 2984,    # MeV/c²
        "width": 32.0,   # MeV/c²
        "spin": 0,
        "window": (2900, 3050),
        "color": "blue"
    },
    "χc1": {
        "mass": 3511,    # MeV/c²
        "width": 0.84,   # MeV/c²
        "spin": 1,
        "window": (3450, 3550),
        "color": "green"
    },
    "ηc(2S)": {
        "mass": 3639,    # MeV/c²
        "width": 11.3,   # MeV/c²
        "spin": 0,
        "window": (3600, 3700),
        "color": "purple"
    }
}

def load_and_prepare_data():
    """Load and prepare data for CC analysis."""
    # Define parameters for loading MC
    years = ["2016", "2017", "2018"]
    tuple_types = ["KSKmKpPip_DD", "KSKmKpPip_LL"]
    data_dir = "/share/lazy/Mohamed/bu2kskpik/MC/processed/"
    tree_name = "DecayTree"
    
    # Define necessary branches (only those needed for CC analysis)
    branches = [
        # B meson properties
        "B_MM", "B_M", "B_PT", "B_IPCHI2_OWNPV", "B_FDCHI2_OWNPV", "B_DIRA_OWNPV",
        
        # K0S properties
        "KS_M", "KS_PT", "KS_FDCHI2_OWNPV", "KS_PX", "KS_PY", "KS_PZ", "KS_PE",
        
        # B daughters (K-, π+, K+) properties
        "P0_PIDK", "P0_PT", "P0_PX", "P0_PY", "P0_PZ", "P0_PE",  # K-
        "P1_PIDK", "P1_PT", "P1_PX", "P1_PY", "P1_PZ", "P1_PE",  # π+
        "P2_PIDK", "P2_PT", "P2_PX", "P2_PY", "P2_PZ", "P2_PE",  # K+
    ]
    
    print("Loading MC data for CC analysis...")
    data = load_mc(
        years=years,
        decay_modes=list(DECAY_MODES.keys()),
        tuple_types=tuple_types,
        data_dir=data_dir,
        tree_name=tree_name,
        branches=branches,
        verbose=True
    )
    
    # Combine DD and LL data for each decay mode
    combined_data = {}
    
    for decay_mode in DECAY_MODES:
        if decay_mode in data:
            print(f"\nProcessing {decay_mode} for CC analysis...")
            
            dd_data = data[decay_mode].get("KSKmKpPip_DD")
            ll_data = data[decay_mode].get("KSKmKpPip_LL")
            
            if dd_data is not None and ll_data is not None:
                combined_data[decay_mode] = ak.concatenate([dd_data, ll_data])
                print(f"  Combined data contains {len(combined_data[decay_mode])} events")
            elif dd_data is not None:
                combined_data[decay_mode] = dd_data
                print(f"  Using only DD data ({len(combined_data[decay_mode])} events)")
            elif ll_data is not None:
                combined_data[decay_mode] = ll_data
                print(f"  Using only LL data ({len(combined_data[decay_mode])} events)")
            else:
                print(f"  No data found for {decay_mode}")
                combined_data[decay_mode] = None
    
    return combined_data

def calculate_three_body_mass(data):
    """Calculate the invariant mass of K0s K- π+ three-body combination."""
    if data is None:
        return None
    
    # Extract K0s 4-momentum
    ks_px = data["KS_PX"]
    ks_py = data["KS_PY"]
    ks_pz = data["KS_PZ"]
    ks_e = data["KS_PE"]
    
    # Extract K- (P0) 4-momentum
    km_px = data["P0_PX"]
    km_py = data["P0_PY"]
    km_pz = data["P0_PZ"]
    km_e = data["P0_PE"]
    
    # Extract π+ (P1) 4-momentum
    pip_px = data["P1_PX"]
    pip_py = data["P1_PY"]
    pip_pz = data["P1_PZ"]
    pip_e = data["P1_PE"]
    
    # Calculate K0s K- π+ invariant mass
    kskpi_px = ks_px + km_px + pip_px
    kskpi_py = ks_py + km_py + pip_py
    kskpi_pz = ks_pz + km_pz + pip_pz
    kskpi_e = ks_e + km_e + pip_e
    
    kskpi_m2 = kskpi_e**2 - kskpi_px**2 - kskpi_py**2 - kskpi_pz**2
    kskpi_m = np.sqrt(ak.to_numpy(kskpi_m2))
    
    return kskpi_m

def analyze_cc_regions(combined_data, output_dir="cc_plots"):
    """Analyze correlated channel regions in detail."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define cc regions based on known resonances
    cc_regions = []
    for name, info in RESONANCES.items():
        x_min, x_max = info["window"]
        cc_regions.append((x_min, x_max, 5250, 5300, name))
    
    for decay_mode in DECAY_MODES:
        if combined_data.get(decay_mode) is not None:
            print(f"\nAnalyzing CC regions for {decay_mode}...")
            
            # Calculate three-body mass
            kskpi_mass = calculate_three_body_mass(combined_data[decay_mode])
            
            # 1. Create 2D plot of three-body mass vs B mass
            fig = plot_cc_regions(
                combined_data[decay_mode],
                x_var=kskpi_mass,
                y_var="B_MM",
                bins=50,
                x_range=(2500, 4000),
                y_range=(5200, 5400),
                figsize=(12, 10),
                cc_regions=None,  # Don't add boxes in this view
                log_scale=True
            )
            
            # Add vertical lines for resonances
            ax = plt.gca()
            for name, info in RESONANCES.items():
                ax.axvline(x=info["mass"], color=info["color"], linestyle='--', alpha=0.7)
                ax.text(info["mass"], ax.get_ylim()[1]*0.98, name, ha='center', va='top', 
                       rotation=90, color=info["color"], fontsize=12)
            
            plt.xlabel("K⁰ₛK⁻π⁺ Mass [MeV/c²]")
            plt.ylabel("B Mass [MeV/c²]")
            plt.title(f"K⁰ₛK⁻π⁺ vs B Mass - {DECAY_MODES[decay_mode]}")
            
            plt.savefig(os.path.join(output_dir, f"kskpi_vs_b_mass_{decay_mode}.png"), dpi=300)
            plt.savefig(os.path.join(output_dir, f"kskpi_vs_b_mass_{decay_mode}.pdf"))
            plt.close()
            
            # 2. Plot the three-body mass with fits for each resonance region
            fig, ax = create_lhcb_figure(figsize=(12, 8))
            
            # Full distribution
            plot_histogram(kskpi_mass, bins=150, range=(2500, 4000),
                          label="All events", ax=ax, histtype="step", color="black")
            
            # Add zoomed inset for J/ψ region
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
            axins = inset_axes(ax, width="40%", height="30%", loc="upper right")
            
            jpsi_window = RESONANCES["J/ψ"]["window"]
            plot_histogram(kskpi_mass, bins=50, range=jpsi_window,
                          ax=axins, histtype="step", color="red")
            axins.set_xlim(jpsi_window)
            axins.set_title("J/ψ Region")
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray")
            
            # Add vertical lines for resonances
            for name, info in RESONANCES.items():
                ax.axvline(x=info["mass"], color=info["color"], linestyle='--', alpha=0.7)
                ax.text(info["mass"], ax.get_ylim()[1]*0.95, name, ha='center', va='top', 
                       rotation=90, color=info["color"], fontsize=12)
            
            finalize_lhcb_figure(ax, 
                               title=f"K⁰ₛK⁻π⁺ Invariant Mass - {DECAY_MODES[decay_mode]}",
                               xlabel="K⁰ₛK⁻π⁺ Mass [MeV/c²]",
                               ylabel="Events / 10 MeV/c²",
                               simulation=True,
                               status="Preliminary")
            
            plt.savefig(os.path.join(output_dir, f"kskpi_mass_full_{decay_mode}.png"), dpi=300)
            plt.savefig(os.path.join(output_dir, f"kskpi_mass_full_{decay_mode}.pdf"))
            plt.close()
            
            # 3. Fit each resonance region separately
            for name, info in RESONANCES.items():
                # Select events in the resonance window
                x_min, x_max = info["window"]
                mask = (kskpi_mass >= x_min) & (kskpi_mass <= x_max)
                
                if ak.sum(mask) > 10:  # Only fit if we have enough events
                    # Fit with PyROOT
                    title = f"{name} Region in {DECAY_MODES[decay_mode]}"
                    
                    # Choose fit function based on resonance
                    if info["width"] < 1.0:  # Narrow resonance (J/ψ, χc1)
                        fit_func = "gaus(0) + pol1(3)"
                    else:  # Wider resonance (ηc, ηc(2S))
                        fit_func = "gaus(0) + pol2(3)"
                    
                    hist, fit_func, canvas = fit_histogram_with_pyroot(
                        kskpi_mass[mask],
                        bins=50,
                        range_min=x_min,
                        range_max=x_max,
                        title=title,
                        fit_function=fit_func,
                        save_path=os.path.join(output_dir, f"{name}_fit_{decay_mode}.pdf")
                    )
                    
                    print(f"  Saved {name} fit for {decay_mode}")
                    
                    # Also fit with RooFit for comparison if it's J/psi or a significant peak
                    if name == "J/ψ" or ak.sum(mask) > 100:
                        fit_result, roofit_canvas, model = fit_with_roofit(
                            kskpi_mass[mask],
                            bins=50,
                            range_min=x_min,
                            range_max=x_max,
                            variable_name=f"{name}_mass",
                            variable_title=f"{name} Mass [MeV/c²]",
                            fit_model="gauss+exp",  # Using exponential background for better stability
                            title=f"{name} RooFit - {DECAY_MODES[decay_mode]}",
                            save_path=os.path.join(output_dir, f"{name}_roofit_{decay_mode}.pdf")
                        )
                        
                        print(f"  Saved {name} RooFit for {decay_mode}")
                else:
                    print(f"  Not enough events in {name} region for {decay_mode}")

def analyze_specific_decay_modes(combined_data, output_dir="cc_plots"):
    """Analyze specific decay modes for validation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare three-body mass for different decay modes
    fig, ax = create_lhcb_figure(figsize=(12, 8))
    
    for decay_mode in DECAY_MODES:
        if combined_data.get(decay_mode) is not None:
            # Calculate three-body mass
            kskpi_mass = calculate_three_body_mass(combined_data[decay_mode])
            
            # Plot distribution with different color for each mode
            plot_histogram(kskpi_mass, bins=150, range=(2500, 4000),
                          label=DECAY_MODES[decay_mode], ax=ax, histtype="step")
    
    # Add vertical lines for resonances
    for name, info in RESONANCES.items():
        ax.axvline(x=info["mass"], color='red', linestyle='--', alpha=0.7)
        ax.text(info["mass"], ax.get_ylim()[1]*0.95, name, ha='center', va='top', 
               rotation=90, color='red', fontsize=12)
    
    finalize_lhcb_figure(ax, 
                       title=f"K⁰ₛK⁻π⁺ Invariant Mass - All Decay Modes",
                       xlabel="K⁰ₛK⁻π⁺ Mass [MeV/c²]",
                       ylabel="Events / 10 MeV/c²",
                       simulation=True,
                       status="Preliminary")
    
    plt.savefig(os.path.join(output_dir, "kskpi_mass_all_modes.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "kskpi_mass_all_modes.pdf"))
    plt.close()
    
    # Also create a zoomed version focusing on charmonium region
    fig, ax = create_lhcb_figure(figsize=(12, 8))
    
    for decay_mode in DECAY_MODES:
        if combined_data.get(decay_mode) is not None:
            # Calculate three-body mass
            kskpi_mass = calculate_three_body_mass(combined_data[decay_mode])
            
            # Plot distribution with different color for each mode
            plot_histogram(kskpi_mass, bins=100, range=(2900, 3900),
                          label=DECAY_MODES[decay_mode], ax=ax, histtype="step")
    
    # Add vertical lines for resonances
    for name, info in RESONANCES.items():
        ax.axvline(x=info["mass"], color='red', linestyle='--', alpha=0.7)
        ax.text(info["mass"], ax.get_ylim()[1]*0.95, name, ha='center', va='top', 
               rotation=90, color='red', fontsize=12)
    
    finalize_lhcb_figure(ax, 
                       title=f"K⁰ₛK⁻π⁺ Invariant Mass (Charmonium Region) - All Decay Modes",
                       xlabel="K⁰ₛK⁻π⁺ Mass [MeV/c²]",
                       ylabel="Events / 10 MeV/c²",
                       simulation=True,
                       status="Preliminary")
    
    plt.savefig(os.path.join(output_dir, "kskpi_mass_charmonium_all_modes.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "kskpi_mass_charmonium_all_modes.pdf"))
    plt.close()

def main():
    """Main function for CC analysis."""
    # Load and prepare data
    combined_data = load_and_prepare_data()
    
    # Create output directory
    output_dir = "cc_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze CC regions
    analyze_cc_regions(combined_data, output_dir)
    
    # Analyze specific decay modes
    analyze_specific_decay_modes(combined_data, output_dir)
    
    print("\nCC analysis completed. Results saved in the 'cc_plots' directory.")

if __name__ == "__main__":
    main()