"""
Analysis script for B -> K0s K- pi+ K+ decay MC data.
This script loads MC for multiple decay modes,
performs basic analysis, creates plots, and applies selection cuts.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Turn off interactive plotting to avoid displaying plots over SSH
plt.ioff()
import awkward as ak
import ROOT
from ROOT import TCanvas, TH1F, TF1, gStyle, kRed, kBlue, kGreen, TLegend
# Disable ROOT GUI
ROOT.gROOT.SetBatch(True)

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import loading function from utils
from utils.load_mc import load_mc
# Import plotting utilities
from utils.plot import (
    create_lhcb_figure, finalize_lhcb_figure, plot_histogram, 
    plot_multiple_histograms, plot_2d_histogram, fit_histogram_with_pyroot,
    fit_with_roofit, create_summary_table, plot_pid_distributions,
    plot_cc_regions, apply_cuts_and_plot
)

# Configure decay modes and data loading parameters
DECAY_MODES = {
    "B2K0s2PipPimKmPipKp": "B^{+} #rightarrow (K^{0}_{S} #rightarrow #pi^{+}#pi^{-})K^{-}#pi^{+}K^{+}",
    "B2Jpsi2K0s2PipPimKmPipKp": "B^{+} #rightarrow (J/#psi #rightarrow (K^{0}_{S} #rightarrow #pi^{+}#pi^{-})K^{-}#pi^{+})K^{+}",
    "B2Etac2K0s2PipPimKmPipKp": "B^{+} #rightarrow (#eta_{c} #rightarrow (K^{0}_{S} #rightarrow #pi^{+}#pi^{-})K^{-}#pi^{+})K^{+}",
    "B2Etac2S2K0s2PipPimKmPipKp": "B^{+} #rightarrow (#eta_{c}(2S) #rightarrow (K^{0}_{S} #rightarrow #pi^{+}#pi^{-})K^{-}#pi^{+})K^{+}",
    "B2Chic12K0s2PipPimKmPipKp": "B^{+} #rightarrow (#chi_{c1} #rightarrow (K^{0}_{S} #rightarrow #pi^{+}#pi^{-})K^{-}#pi^{+})K^{+}"
}

# Define tuples and years for the analysis
TUPLE_TYPES = ["KSKmKpPip_DD", "KSKmKpPip_LL"]
YEARS = ["2016", "2017", "2018"]  # Excluding 2015 as mentioned
DATA_DIR = "/share/lazy/Mohamed/bu2kskpik/MC/processed/"
TREE_NAME = "DecayTree"

# Define necessary branches
BRANCHES = [
    # B meson properties
    "B_ENDVERTEX_X", "B_ENDVERTEX_Y", "B_ENDVERTEX_Z", "B_ENDVERTEX_CHI2",
    "B_OWNPV_X", "B_OWNPV_Y", "B_OWNPV_Z",
    "B_IP_OWNPV", "B_IPCHI2_OWNPV", "B_FD_OWNPV", "B_FDCHI2_OWNPV", "B_DIRA_OWNPV",
    "B_P", "B_PT", "B_PE", "B_PX", "B_PY", "B_PZ", "B_MM", "B_MMERR", "B_M",
    "B_ID", "B_TAU", "B_TAUERR", "B_TAUCHI2",
    
    # K0S properties
    "KS_ENDVERTEX_X", "KS_ENDVERTEX_Y", "KS_ENDVERTEX_Z", "KS_ENDVERTEX_CHI2",
    "KS_OWNPV_X", "KS_OWNPV_Y", "KS_OWNPV_Z",
    "KS_IP_OWNPV", "KS_IPCHI2_OWNPV", "KS_FD_OWNPV", "KS_FDCHI2_OWNPV", "KS_DIRA_OWNPV",
    "KS_P", "KS_PT", "KS_PE", "KS_PX", "KS_PY", "KS_PZ", "KS_MM", "KS_MMERR", "KS_M",
    "KS_TAU", "KS_TAUERR", "KS_TAUCHI2",
    
    # K0S daughters (pions) properties
    "KS_P0_P", "KS_P0_PT", "KS_P0_PE", "KS_P0_PX", "KS_P0_PY", "KS_P0_PZ", "KS_P0_M", "KS_P0_ID",
    "KS_P0_IP_OWNPV", "KS_P0_IPCHI2_OWNPV",
    "KS_P0_PIDK", "KS_P0_PIDp", "KS_P0_PIDe", "KS_P0_PIDmu",
    "KS_P0_ProbNNk", "KS_P0_ProbNNpi", "KS_P0_ProbNNp",
    
    "KS_P1_P", "KS_P1_PT", "KS_P1_PE", "KS_P1_PX", "KS_P1_PY", "KS_P1_PZ", "KS_P1_M", "KS_P1_ID",
    "KS_P1_IP_OWNPV", "KS_P1_IPCHI2_OWNPV",
    "KS_P1_PIDK", "KS_P1_PIDp", "KS_P1_PIDe", "KS_P1_PIDmu",
    "KS_P1_ProbNNk", "KS_P1_ProbNNpi", "KS_P1_ProbNNp",
    
    # B daughters (K-, pi+, K+) properties
    "P0_P", "P0_PT", "P0_PE", "P0_PX", "P0_PY", "P0_PZ", "P0_M", "P0_ID",
    "P0_IP_OWNPV", "P0_IPCHI2_OWNPV",
    "P0_PIDK", "P0_PIDp", "P0_PIDe", "P0_PIDmu",
    "P0_ProbNNk", "P0_ProbNNpi", "P0_ProbNNp",
    
    "P1_P", "P1_PT", "P1_PE", "P1_PX", "P1_PY", "P1_PZ", "P1_M", "P1_ID",
    "P1_IP_OWNPV", "P1_IPCHI2_OWNPV",
    "P1_PIDK", "P1_PIDp", "P1_PIDe", "P1_PIDmu",
    "P1_ProbNNk", "P1_ProbNNpi", "P1_ProbNNp",
    
    "P2_P", "P2_PT", "P2_PE", "P2_PX", "P2_PY", "P2_PZ", "P2_M", "P2_ID",
    "P2_IP_OWNPV", "P2_IPCHI2_OWNPV",
    "P2_PIDK", "P2_PIDp", "P2_PIDe", "P2_PIDmu",
    "P2_ProbNNk", "P2_ProbNNpi", "P2_ProbNNp",
    
    # Event information
    "eventNumber", "runNumber", "Polarity", "nCandidate"
]

def load_mc_data():
    """Load MC data for all decay modes and tuple types."""
    print("Loading MC data for analysis...")
    data = load_mc(
        years=YEARS,
        decay_modes=list(DECAY_MODES.keys()),
        tuple_types=TUPLE_TYPES,
        data_dir=DATA_DIR,
        tree_name=TREE_NAME,
        branches=BRANCHES,
        verbose=True
    )
    return data

def process_and_combine_data(data):
    """Process and combine data for each decay mode."""
    combined_data = {}
    dd_data = {}
    ll_data = {}
    
    for decay_mode in DECAY_MODES:
        if decay_mode in data:
            print(f"\nProcessing {decay_mode}...")
            
            # Get DD and LL data
            if "KSKmKpPip_DD" in data[decay_mode]:
                dd_data[decay_mode] = data[decay_mode]["KSKmKpPip_DD"]
                print(f"  Loaded {len(dd_data[decay_mode])} events for DD tuple type")
            else:
                print(f"  No DD data found for {decay_mode}")
                dd_data[decay_mode] = None
            
            if "KSKmKpPip_LL" in data[decay_mode]:
                ll_data[decay_mode] = data[decay_mode]["KSKmKpPip_LL"]
                print(f"  Loaded {len(ll_data[decay_mode])} events for LL tuple type")
            else:
                print(f"  No LL data found for {decay_mode}")
                ll_data[decay_mode] = None
            
            # Combine DD and LL data
            if dd_data[decay_mode] is not None and ll_data[decay_mode] is not None:
                combined_data[decay_mode] = ak.concatenate([dd_data[decay_mode], ll_data[decay_mode]])
                print(f"  Combined data contains {len(combined_data[decay_mode])} events")
            elif dd_data[decay_mode] is not None:
                combined_data[decay_mode] = dd_data[decay_mode]
                print(f"  Using only DD data ({len(combined_data[decay_mode])} events)")
            elif ll_data[decay_mode] is not None:
                combined_data[decay_mode] = ll_data[decay_mode]
                print(f"  Using only LL data ({len(combined_data[decay_mode])} events)")
            else:
                print(f"  No data found for {decay_mode}")
                combined_data[decay_mode] = None
    
    return combined_data, dd_data, ll_data

def calculate_statistics(data, field="B_MM"):
    """Calculate basic statistics for a field in the data."""
    if data is None:
        return None
    
    values = ak.to_numpy(data[field])
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "count": len(values)
    }

def print_statistics(data_dict, field="B_MM"):
    """Print statistics for all decay modes."""
    print(f"\nStatistics for {field}:")
    for decay_mode, data in data_dict.items():
        if data is not None:
            stats = calculate_statistics(data, field)
            print(f"  {DECAY_MODES[decay_mode]}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Median: {stats['median']:.2f}")
            print(f"    Std Dev: {stats['std']:.2f}")
            print(f"    Min: {stats['min']:.2f}")
            print(f"    Max: {stats['max']:.2f}")
            print(f"    Count: {stats['count']}")

def plot_all_b_mass_distributions(combined_data, dd_data, ll_data, output_dir="plots"):
    """Plot B mass distributions for all decay modes."""
    os.makedirs(output_dir, exist_ok=True)
    
    for decay_mode in DECAY_MODES:
        if combined_data.get(decay_mode) is not None:
            # Create figure for combined, DD, and LL distributions
            fig, ax = create_lhcb_figure(figsize=(10, 8))
            
            # Plot DD data if available
            if dd_data.get(decay_mode) is not None:
                plot_histogram(dd_data[decay_mode]["B_MM"], bins=100, range=(5200, 5400),
                              label="DD", ax=ax, histtype="step", color="blue", alpha=0.7)
            
            # Plot LL data if available
            if ll_data.get(decay_mode) is not None:
                plot_histogram(ll_data[decay_mode]["B_MM"], bins=100, range=(5200, 5400),
                              label="LL", ax=ax, histtype="step", color="green", alpha=0.7)
            
            # Plot combined data
            plot_histogram(combined_data[decay_mode]["B_MM"], bins=100, range=(5200, 5400),
                          label="Combined", ax=ax, histtype="step", color="red", alpha=0.7)
            
            # Finalize plot
            finalize_lhcb_figure(ax, 
                              title=f"B Mass Distribution - {DECAY_MODES[decay_mode]}",
                              xlabel="B Mass [MeV/c^{2}]",
                              ylabel="Events / 2 MeV/c^{2}",
                              simulation=False,
                              status="Preliminary",
                              data_label="Simulation")
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"b_mass_{decay_mode}.pdf"))
            plt.close()
            
            print(f"  Saved B mass plot for {decay_mode}")
            
            # Fit the B mass distribution with PyROOT
            hist, fit_func, canvas = fit_histogram_with_pyroot(
                combined_data[decay_mode]["B_MM"],
                bins=100,
                range_min=5200,
                range_max=5400,
                title=f"B Mass Fit - {DECAY_MODES[decay_mode]}",
                fit_function="gaus(0) + pol1(3)",
                save_path=os.path.join(output_dir, f"b_mass_fit_{decay_mode}.pdf")
            )
            
            # Also fit with RooFit for comparison
            fit_result, roofit_canvas, model = fit_with_roofit(
                combined_data[decay_mode]["B_MM"],
                bins=100,
                range_min=5200,
                range_max=5400,
                variable_name="B_mass",
                variable_title="B Mass [MeV/c^{2}]",
                fit_model="gauss+exp",  # Using exponential background instead of polynomial
                title=f"B Mass RooFit - {DECAY_MODES[decay_mode]}",
                save_path=os.path.join(output_dir, f"b_mass_roofit_{decay_mode}.pdf")
            )
            
            print(f"  Saved B mass fits for {decay_mode}")

def plot_pid_variables(combined_data, output_dir="plots"):
    """Plot PID variables for all decay modes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define particles and PID variables
    particles = ["P0", "P1", "P2", "KS_P0", "KS_P1"]
    pid_vars = ["PIDK", "PIDp", "PIDe", "PIDmu", "ProbNNk", "ProbNNpi", "ProbNNp"]
    
    for decay_mode in DECAY_MODES:
        if combined_data.get(decay_mode) is not None:
            print(f"\nPlotting PID variables for {decay_mode}...")
            
            # Plot PID variables for each particle
            for particle in particles:
                # Create figure with multiple PID plots
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                axes = axes.flatten()
                
                # Plot each PID variable
                for i, var in enumerate(pid_vars):
                    if i < len(axes):
                        pid_var = f"{particle}_{var}"
                        if pid_var in combined_data[decay_mode].fields:
                            plot_histogram(combined_data[decay_mode][pid_var], 
                                          bins=50, ax=axes[i], histtype="step")
                            axes[i].set_xlabel(var)
                            axes[i].set_ylabel("Events")
                            axes[i].set_title(f"{pid_var}")
                            finalize_lhcb_figure(axes[i], simulation=False, legend=False,
                                              status="Preliminary", tight_layout=False)
                        else:
                            axes[i].text(0.5, 0.5, f"Variable {pid_var} not found", 
                                      ha='center', va='center', transform=axes[i].transAxes)
                
                # Remove empty subplots
                for i in range(len(pid_vars), len(axes)):
                    fig.delaxes(axes[i])
                
                plt.tight_layout()
                title = f"PID Variables - {particle} - {DECAY_MODES[decay_mode]}"
                fig.suptitle(title, fontsize=16, y=1.02)
                
                # Save figure
                plt.savefig(os.path.join(output_dir, f"pid_{particle}_{decay_mode}.pdf"),
                           bbox_inches='tight')
                plt.close()
                
                print(f"  Saved PID plots for {particle}")

def plot_correlated_channels(combined_data, output_dir="plots"):
    """Plot correlated channels (CC) regions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define CC regions based on known resonances
    # Format: (x_min, x_max, y_min, y_max, label)
    cc_regions = [
        # J/ψ region
        (3050, 3150, 5250, 5300, "J/#psi #rightarrow K^{0}_{S}K^{-}#pi^{+}"),
        # ηc region
        (2900, 3000, 5250, 5300, "#eta_{c} #rightarrow K^{0}_{S}K^{-}#pi^{+}"),
        # χc1 region
        (3450, 3550, 5250, 5300, "#chi_{c1} #rightarrow K^{0}_{S}K^{-}#pi^{+}"),
        # ηc(2S) region
        (3600, 3700, 5250, 5300, "#eta_{c}(2S) #rightarrow K^{0}_{S}K^{-}#pi^{+}")
    ]
    
    for decay_mode in DECAY_MODES:
        if combined_data.get(decay_mode) is not None:
            print(f"\nPlotting CC regions for {decay_mode}...")
            
            # Calculate invariant mass of K0s K- pi+ (three-body combination)
            # We need to compute this from momentum components
            
            # Extract K0s 4-momentum
            ks_px = combined_data[decay_mode]["KS_PX"]
            ks_py = combined_data[decay_mode]["KS_PY"]
            ks_pz = combined_data[decay_mode]["KS_PZ"]
            ks_e = combined_data[decay_mode]["KS_PE"]
            
            # Extract K- (P0) 4-momentum
            km_px = combined_data[decay_mode]["P0_PX"]
            km_py = combined_data[decay_mode]["P0_PY"]
            km_pz = combined_data[decay_mode]["P0_PZ"]
            km_e = combined_data[decay_mode]["P0_PE"]
            
            # Extract π+ (P1) 4-momentum
            pip_px = combined_data[decay_mode]["P1_PX"]
            pip_py = combined_data[decay_mode]["P1_PY"]
            pip_pz = combined_data[decay_mode]["P1_PZ"]
            pip_e = combined_data[decay_mode]["P1_PE"]
            
            # Calculate K0s K- π+ invariant mass
            kskpi_px = ks_px + km_px + pip_px
            kskpi_py = ks_py + km_py + pip_py
            kskpi_pz = ks_pz + km_pz + pip_pz
            kskpi_e = ks_e + km_e + pip_e
            
            kskpi_m2 = kskpi_e**2 - kskpi_px**2 - kskpi_py**2 - kskpi_pz**2
            kskpi_m = np.sqrt(ak.to_numpy(kskpi_m2))
            
            # Create 2D histogram directly without using plot_cc_regions
            fig, ax = create_lhcb_figure(figsize=(12, 10))
            
            # Get data
            x_data = ak.to_numpy(combined_data[decay_mode]["B_MM"])
            y_data = kskpi_m  # Already a numpy array
            
            # Create 2D histogram
            h = ax.hist2d(x_data, y_data, bins=50, 
                         range=((5200, 5400), (2500, 6000)),
                         cmap='viridis', norm=mpl.colors.LogNorm())
            
            plt.colorbar(h[3], ax=ax, label="Events")
            
            # Add CC regions if provided
            for region in cc_regions:
                x_min, x_max, y_min, y_max, label = region
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   fill=False, edgecolor='red', linestyle='--', linewidth=2)
                ax.add_patch(rect)
                ax.text(x_min + (x_max - x_min) / 2, y_max + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                       label, ha='center', va='bottom', color='red', fontsize=12)
            
            # Finalize plot
            finalize_lhcb_figure(ax, 
                               xlabel="B Mass [MeV/c^{2}]", 
                               ylabel="K^{0}_{S}K^{-}#pi^{+} Mass [MeV/c^{2}]",
                               title=f"Correlated Channels - {DECAY_MODES[decay_mode]}",
                               simulation=False,
                               status="Preliminary")
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"cc_regions_{decay_mode}.pdf"))
            plt.close()
            
            print(f"  Saved CC regions plot for {decay_mode}")
            
            # Also create 1D histograms of K0s K- π+ mass
            fig, ax = create_lhcb_figure(figsize=(10, 8))
            
            plot_histogram(kskpi_m, bins=100, range=(2500, 6000),
                          label="K^{0}_{S}K^{-}#pi^{+} mass", ax=ax, histtype="step")
            
            # Add vertical lines for known resonances
            resonances = [
                (3097, "J/#psi"),
                (2984, "#eta_{c}"),
                (3511, "#chi_{c1}"),
                (3639, "#eta_{c}(2S)")
            ]
            
            for mass, name in resonances:
                ax.axvline(x=mass, color='red', linestyle='--', alpha=0.7)
                ax.text(mass, ax.get_ylim()[1]*0.95, name, ha='center', va='top', 
                       rotation=90, color='red')
            
            finalize_lhcb_figure(ax, 
                              title=f"K^{{0}}_{{S}}K^{{-}}#pi^{{+}} Invariant Mass - {DECAY_MODES[decay_mode]}",
                              xlabel="K^{0}_{S}K^{-}#pi^{+} Mass [MeV/c^{2}]",
                              ylabel="Events / 15 MeV/c^{2}",
                              simulation=False,
                              status="Preliminary")
            
            plt.savefig(os.path.join(output_dir, f"kskpi_mass_{decay_mode}.pdf"))
            plt.close()
            
            print(f"  Saved K^{{0}}_{{S}}K^{{-}}#pi^{{+}} mass plot for {decay_mode}")
            
            
def apply_selection_cuts(combined_data, output_dir="plots"):
    """Apply selection cuts and plot distributions after cuts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define selection cuts
    cuts = [
        "B_PT > 1000",                # B pT > 1 GeV/c
        "B_IPCHI2_OWNPV < 25",        # B IP chi2 < 25
        "B_FDCHI2_OWNPV > 100",       # B flight distance chi2 > 100
        "B_DIRA_OWNPV > 0.9999",      # B direction angle cosine > 0.9999
        "KS_FDCHI2_OWNPV > 25",       # K0S flight distance chi2 > 25
        "P0_ProbNNk > 0.2",                
        "P1_ProbNNk > 0.2",                
        "P2_ProbNNpi > 0.2"                 
    ]
    
    cut_labels = [
        "B p_{T} > 1 GeV/c",
        "B IP #chi^{2} < 25",
        "B FD #chi^{2} > 100",
        "B DIRA > 0.9999",
        "K^{0}_{S} FD #chi^{2} > 25",
        "K^{-} NNk > 0.2",
        "#pi^{+} NNk > 0.2",
        "K^{+} NN#pi > 0.2"
    ]
    
    for decay_mode in DECAY_MODES:
        if combined_data.get(decay_mode) is not None:
            print(f"\nApplying selection cuts for {decay_mode}...")
            
            # Apply cuts and plot B mass distribution
            fig, filtered_data = apply_cuts_and_plot(
                combined_data[decay_mode],
                cuts=cuts,
                variable="B_MM",
                bins=100,
                range=(5200, 5400),
                title=f"B Mass Distribution with Cuts - {DECAY_MODES[decay_mode]}",
                xlabel="B Mass [MeV/c^{2}]",
                ylabel="Events / 2 MeV/c^{2}",
                figsize=(10, 8),
                compare_before=True,
                cut_labels=cut_labels
            )
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"b_mass_cuts_{decay_mode}.pdf"))
            plt.close()
            
            print(f"  Saved B mass with cuts plot for {decay_mode}")
            
            # Fit B mass after cuts
            if len(filtered_data) > 0:
                hist, fit_func, canvas = fit_histogram_with_pyroot(
                    filtered_data["B_MM"],
                    bins=100,
                    range_min=5200,
                    range_max=5400,
                    title=f"B Mass Fit After Cuts - {DECAY_MODES[decay_mode]}",
                    fit_function="gaus(0) + pol1(3)",
                    save_path=os.path.join(output_dir, f"b_mass_fit_cuts_{decay_mode}.pdf")
                )
                
                print(f"  Saved B mass fit after cuts for {decay_mode}")
            else:
                print(f"  No events left after cuts for {decay_mode}")

def main():
    """Main analysis function."""
    # Create output directory
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MC data
    data = load_mc_data()
    
    # Process and combine data
    combined_data, dd_data, ll_data = process_and_combine_data(data)
    
    # Print statistics
    print_statistics(combined_data)
    
    # Plot B mass distributions
    plot_all_b_mass_distributions(combined_data, dd_data, ll_data, output_dir)
    
    # Plot PID variables
    plot_pid_variables(combined_data, output_dir)
    
    # Plot correlated channels (CC) regions
    plot_correlated_channels(combined_data, output_dir)
    
    # Apply selection cuts
    apply_selection_cuts(combined_data, output_dir)
    
    print("\nAnalysis completed. Results saved in the 'plots' directory.")

if __name__ == "__main__":
    main()