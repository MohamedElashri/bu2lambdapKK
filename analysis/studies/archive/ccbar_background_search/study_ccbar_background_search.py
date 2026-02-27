"""
Standalone study: cc̄ Background Search in M(Λ̄pK⁻) Spectrum.

Ported from: archive/analysis/studies/feedback_dec2024/study2_ccbar_background_search.py

What it does:
  Searches for charm-anticharm (cc̄) resonances in the M(Λ̄pK⁻) spectrum by
  separating events into two B⁺ mass categories:
    - B⁺ signal window [5255, 5305] MeV → cc̄ from B⁺ decay
    - B⁺ sidebands (M < 5200 or M > 5350) → cc̄ NOT from B⁺ decay
  Marks known cc̄ resonances with vertical reference lines.
  Produces full-spectrum and zoomed sideband-region plots.

Improvements over original:
  - matplotlib + mplhep (LHCb style) instead of ROOT TCanvas/TH1
  - Multi-panel layout: signal + sideband on one page, zoomed regions on second page
  - Proper legend for resonance reference lines
  - Summary CSV with event counts and bin-by-bin yields near resonances
  - Uses cached Step 2 data from the main pipeline (awkward arrays)

Snakemake injects:
  snakemake.params.config_dir
  snakemake.params.cache_dir
  snakemake.params.output_dir
  snakemake.output.signal_plot
  snakemake.output.sideband_plot
  snakemake.output.csv
"""

import sys
from pathlib import Path

# Ensure the project root (analysis/) is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
signal_plot_path = snakemake.output.signal_plot  # noqa: F821
sideband_plot_path = snakemake.output.sideband_plot  # noqa: F821
summary_csv_path = snakemake.output.csv  # noqa: F821

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

# Known cc̄ resonances: name → (mass [MeV], colour)
CC_RESONANCES = {
    r"$\eta_c(1S)$": (2983.9, "#ff7f0e"),
    r"$J/\psi$": (3096.9, "#d62728"),
    r"$\chi_{c0}$": (3414.7, "#1f77b4"),
    r"$\chi_{c1}$": (3510.7, "#2ca02c"),
    r"$\chi_{c2}$": (3556.2, "#9467bd"),
    r"$\psi(2S)$": (3686.1, "#8c564b"),
}

# M(Λ̄pK⁻) histogram range (full spectrum)
M_LPKM_MIN = 2800.0
M_LPKM_MAX = 4000.0
N_BINS = 120

# Zoomed sideband regions
LEFT_SIDEBAND_MAX = 3050.0
RIGHT_SIDEBAND_MIN = 3650.0

# B⁺ mass windows
bu_fixed = config.selection.get("bu_fixed_selection", {})
BU_SIGNAL_MIN = bu_fixed.get("mass_corrected_min", 5255.0)
BU_SIGNAL_MAX = bu_fixed.get("mass_corrected_max", 5305.0)
BU_SIDEBAND_LOW = 5200.0
BU_SIDEBAND_HIGH = 5350.0


# ---------------------------------------------------------------------------
# Helper: compute step dependencies
# ---------------------------------------------------------------------------
def compute_step_dependencies(step, extra_params=None):
    config_files = list(Path(config_dir).glob("*.toml"))
    code_files = []
    if step == "2":
        code_files = [
            project_root / "modules" / "data_handler.py",
            project_root / "modules" / "lambda_selector.py",
        ]
    return cache.compute_dependencies(
        config_files=config_files, code_files=code_files, extra_params=extra_params
    )


def _get_flat_branch(events, branch_name):
    """Get a branch as a flat (non-jagged) array, using ak.firsts if needed."""
    br = events[branch_name]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    return br


# ---------------------------------------------------------------------------
# Helper: draw resonance reference lines on an axis
# ---------------------------------------------------------------------------
def draw_resonance_lines(ax, x_min, x_max, y_max):
    """Draw vertical dashed lines for known cc̄ resonances within the axis range."""
    drawn = []
    for name, (mass, color) in CC_RESONANCES.items():
        if x_min < mass < x_max:
            ax.axvline(
                mass,
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"{name} ({mass:.0f})",
            )
            drawn.append(name)
    return drawn


# ---------------------------------------------------------------------------
# Helper: estimate local significance near a resonance
# ---------------------------------------------------------------------------
def estimate_peak_significance(mass_array, resonance_mass, window_half=15.0, sideband_half=50.0):
    """
    Rough significance estimate: (S) / sqrt(B) where
      S = events in [mass - window, mass + window] minus expected background
      B = background estimated from sidebands scaled to signal window width

    Returns (n_signal_window, n_background_est, significance)
    """
    in_window = np.sum(
        (mass_array > resonance_mass - window_half) & (mass_array < resonance_mass + window_half)
    )

    # Left sideband for background estimate
    left_sb = np.sum(
        (mass_array > resonance_mass - sideband_half - window_half)
        & (mass_array < resonance_mass - window_half)
    )
    # Right sideband for background estimate
    right_sb = np.sum(
        (mass_array > resonance_mass + window_half)
        & (mass_array < resonance_mass + sideband_half + window_half)
    )

    # Scale sideband to signal window width
    sb_width = 2 * sideband_half
    sig_width = 2 * window_half
    if sb_width > 0:
        bg_est = (left_sb + right_sb) * (sig_width / sb_width)
    else:
        bg_est = 0.0

    excess = in_window - bg_est
    significance = excess / np.sqrt(bg_est) if bg_est > 0 else 0.0

    return int(in_window), float(bg_est), float(significance)


# ---------------------------------------------------------------------------
# Load Step 2 cached data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("cc̄ BACKGROUND SEARCH STUDY")
print("=" * 80)

years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

step2_deps = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)

data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)
if data_dict is None:
    raise AnalysisError(
        "Step 2 cached data not found! Run Step 2 (load_data) first.\n"
        "  cd analysis/ && uv run snakemake load_data -j1"
    )

# Combine data across years and track types
print("\n[Loading real data]")
data_arrays = []
for year in data_dict:
    for track_type in data_dict[year]:
        arr = data_dict[year][track_type]
        if hasattr(arr, "layout"):
            data_arrays.append(arr)
all_data = ak.concatenate(data_arrays, axis=0)
print(f"  Total data (after Lambda cuts): {len(all_data):,}")

# Identify mass branches
mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"
print(f"  Mass branch: {mass_branch}")
print(f"  B+ mass branch: {bu_mass_branch}")

# Extract flat numpy arrays
mass_all = np.asarray(ak.to_numpy(_get_flat_branch(all_data, mass_branch)))
bu_mass_all = np.asarray(ak.to_numpy(_get_flat_branch(all_data, bu_mass_branch)))

# ---------------------------------------------------------------------------
# Separate by B⁺ mass region
# ---------------------------------------------------------------------------
print(f"\n  B⁺ signal window: [{BU_SIGNAL_MIN:.0f}, {BU_SIGNAL_MAX:.0f}] MeV")
print(f"  B⁺ sidebands: M(B⁺) < {BU_SIDEBAND_LOW:.0f} or M(B⁺) > {BU_SIDEBAND_HIGH:.0f} MeV")

signal_mask = (bu_mass_all > BU_SIGNAL_MIN) & (bu_mass_all < BU_SIGNAL_MAX)
sideband_mask = (bu_mass_all < BU_SIDEBAND_LOW) | (bu_mass_all > BU_SIDEBAND_HIGH)

mass_signal = mass_all[signal_mask]
mass_sideband = mass_all[sideband_mask]

print(f"\n  Events in B⁺ signal window: {len(mass_signal):,}")
print(f"  Events in B⁺ sidebands:     {len(mass_sideband):,}")

# ---------------------------------------------------------------------------
# Compute histograms
# ---------------------------------------------------------------------------
bin_edges_full = np.linspace(M_LPKM_MIN, M_LPKM_MAX, N_BINS + 1)
bin_centers_full = 0.5 * (bin_edges_full[:-1] + bin_edges_full[1:])
bin_width_full = bin_edges_full[1] - bin_edges_full[0]

h_signal, _ = np.histogram(mass_signal, bins=bin_edges_full)
h_sideband, _ = np.histogram(mass_sideband, bins=bin_edges_full)

# Zoomed regions
n_bins_left = 25
bin_edges_left = np.linspace(M_LPKM_MIN, LEFT_SIDEBAND_MAX, n_bins_left + 1)
bin_centers_left = 0.5 * (bin_edges_left[:-1] + bin_edges_left[1:])
bin_width_left = bin_edges_left[1] - bin_edges_left[0]

n_bins_right = 35
bin_edges_right = np.linspace(RIGHT_SIDEBAND_MIN, M_LPKM_MAX, n_bins_right + 1)
bin_centers_right = 0.5 * (bin_edges_right[:-1] + bin_edges_right[1:])
bin_width_right = bin_edges_right[1] - bin_edges_right[0]

# Zoomed histograms (signal window only — where we expect cc̄ from B⁺)
h_signal_left, _ = np.histogram(mass_signal, bins=bin_edges_left)
h_signal_right, _ = np.histogram(mass_signal, bins=bin_edges_right)

# Zoomed histograms (sidebands)
h_sideband_left, _ = np.histogram(mass_sideband, bins=bin_edges_left)
h_sideband_right, _ = np.histogram(mass_sideband, bins=bin_edges_right)

# ---------------------------------------------------------------------------
# Compute peak significances
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PEAK SIGNIFICANCE ESTIMATES (B⁺ signal window)")
print("=" * 80)

summary_rows = []

for name, (mass, _) in CC_RESONANCES.items():
    n_win, bg_est, sig = estimate_peak_significance(mass_signal, mass)
    # Strip LaTeX for CSV
    clean_name = name.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    print(
        f"  {clean_name:12s} ({mass:.0f} MeV): window={n_win:>6,}  bg_est={bg_est:>8.1f}  S/√B={sig:>6.1f}"
    )

    summary_rows.append(
        {
            "resonance": clean_name,
            "mass_mev": mass,
            "region": "signal_window",
            "n_in_window": n_win,
            "bg_estimate": round(bg_est, 1),
            "significance": round(sig, 1),
        }
    )

print("\n" + "=" * 80)
print("PEAK SIGNIFICANCE ESTIMATES (B⁺ sidebands)")
print("=" * 80)

for name, (mass, _) in CC_RESONANCES.items():
    n_win, bg_est, sig = estimate_peak_significance(mass_sideband, mass)
    clean_name = name.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    print(
        f"  {clean_name:12s} ({mass:.0f} MeV): window={n_win:>6,}  bg_est={bg_est:>8.1f}  S/√B={sig:>6.1f}"
    )

    summary_rows.append(
        {
            "resonance": clean_name,
            "mass_mev": mass,
            "region": "sidebands",
            "n_in_window": n_win,
            "bg_estimate": round(bg_est, 1),
            "significance": round(sig, 1),
        }
    )

# Add event count summary rows
summary_rows.append(
    {
        "resonance": "TOTAL",
        "mass_mev": 0,
        "region": "signal_window",
        "n_in_window": len(mass_signal),
        "bg_estimate": 0,
        "significance": 0,
    }
)
summary_rows.append(
    {
        "resonance": "TOTAL",
        "mass_mev": 0,
        "region": "sidebands",
        "n_in_window": len(mass_sideband),
        "bg_estimate": 0,
        "significance": 0,
    }
)

# Save summary CSV
output_path = Path(summary_csv_path)
output_path.parent.mkdir(exist_ok=True, parents=True)
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(output_path, index=False)
print(f"\n  Summary CSV saved to {summary_csv_path}")

# ---------------------------------------------------------------------------
# Plot 1: B⁺ signal window (full + zoomed)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING SIGNAL WINDOW PLOTS")
print("=" * 80)

signal_plot = Path(signal_plot_path)
signal_plot.parent.mkdir(exist_ok=True, parents=True)

with PdfPages(signal_plot) as pdf:
    # --- Page 1: Full spectrum in B⁺ signal window ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: full spectrum
    ax = axes[0]
    ax.fill_between(
        bin_centers_full,
        h_signal,
        step="mid",
        alpha=0.3,
        color="#1f77b4",
    )
    ax.step(
        bin_centers_full,
        h_signal,
        where="mid",
        color="#1f77b4",
        linewidth=2.0,
        label=f"Data ({len(mass_signal):,} events)",
    )
    draw_resonance_lines(ax, M_LPKM_MIN, M_LPKM_MAX, ax.get_ylim()[1])
    ax.set_xlim(M_LPKM_MIN, M_LPKM_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax.set_ylabel(f"Candidates / ({bin_width_full:.0f} MeV/$c^{{2}}$)", fontsize=14)
    ax.set_title(
        r"$B^{+}$ signal window: $c\bar{c}$ from $B^{+}$ decay",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2)

    # Right panel: zoomed around J/ψ peak
    ax2 = axes[1]
    jpsi_zoom_min, jpsi_zoom_max = 3020.0, 3180.0
    n_bins_zoom = 40
    be_zoom = np.linspace(jpsi_zoom_min, jpsi_zoom_max, n_bins_zoom + 1)
    bc_zoom = 0.5 * (be_zoom[:-1] + be_zoom[1:])
    bw_zoom = be_zoom[1] - be_zoom[0]
    h_zoom, _ = np.histogram(mass_signal, bins=be_zoom)

    ax2.fill_between(bc_zoom, h_zoom, step="mid", alpha=0.3, color="#1f77b4")
    ax2.step(bc_zoom, h_zoom, where="mid", color="#1f77b4", linewidth=2.0)
    draw_resonance_lines(ax2, jpsi_zoom_min, jpsi_zoom_max, ax2.get_ylim()[1])
    ax2.set_xlim(jpsi_zoom_min, jpsi_zoom_max)
    ax2.set_ylim(0, None)
    ax2.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax2.set_ylabel(f"Candidates / ({bw_zoom:.0f} MeV/$c^{{2}}$)", fontsize=14)
    ax2.set_title(r"Zoom: $J/\psi$ region", fontsize=14)
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        r"$c\bar{c}$ Background Search — $B^{+}$ Signal Window "
        rf"$[{BU_SIGNAL_MIN:.0f}, {BU_SIGNAL_MAX:.0f}]$ MeV/$c^{{2}}$",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 2: Zoomed left and right sideband regions (signal window data) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left sideband zoom (ηc region)
    ax = axes[0]
    ax.fill_between(
        bin_centers_left,
        h_signal_left,
        step="mid",
        alpha=0.3,
        color="#1f77b4",
    )
    ax.step(
        bin_centers_left,
        h_signal_left,
        where="mid",
        color="#1f77b4",
        linewidth=2.0,
    )
    draw_resonance_lines(ax, M_LPKM_MIN, LEFT_SIDEBAND_MAX, ax.get_ylim()[1])
    ax.set_xlim(M_LPKM_MIN, LEFT_SIDEBAND_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax.set_ylabel(f"Candidates / ({bin_width_left:.0f} MeV/$c^{{2}}$)", fontsize=14)
    ax.set_title(r"Left region ($\eta_c$ neighbourhood)", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")

    # Right sideband zoom (ψ(2S) region)
    ax = axes[1]
    ax.fill_between(
        bin_centers_right,
        h_signal_right,
        step="mid",
        alpha=0.3,
        color="#1f77b4",
    )
    ax.step(
        bin_centers_right,
        h_signal_right,
        where="mid",
        color="#1f77b4",
        linewidth=2.0,
    )
    draw_resonance_lines(ax, RIGHT_SIDEBAND_MIN, M_LPKM_MAX, ax.get_ylim()[1])
    ax.set_xlim(RIGHT_SIDEBAND_MIN, M_LPKM_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax.set_ylabel(f"Candidates / ({bin_width_right:.0f} MeV/$c^{{2}}$)", fontsize=14)
    ax.set_title(r"Right region ($\psi(2S)$ neighbourhood)", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        r"Zoomed Regions — $B^{+}$ Signal Window",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print(f"  Signal window plots saved to {signal_plot_path}")

# ---------------------------------------------------------------------------
# Plot 2: B⁺ sidebands (full + zoomed)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING SIDEBAND PLOTS")
print("=" * 80)

sideband_plot = Path(sideband_plot_path)
sideband_plot.parent.mkdir(exist_ok=True, parents=True)

with PdfPages(sideband_plot) as pdf:
    # --- Page 1: Full spectrum in B⁺ sidebands ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: full spectrum
    ax = axes[0]
    ax.fill_between(
        bin_centers_full,
        h_sideband,
        step="mid",
        alpha=0.3,
        color="#d62728",
    )
    ax.step(
        bin_centers_full,
        h_sideband,
        where="mid",
        color="#d62728",
        linewidth=2.0,
        label=f"Data ({len(mass_sideband):,} events)",
    )
    draw_resonance_lines(ax, M_LPKM_MIN, M_LPKM_MAX, ax.get_ylim()[1])
    ax.set_xlim(M_LPKM_MIN, M_LPKM_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax.set_ylabel(f"Candidates / ({bin_width_full:.0f} MeV/$c^{{2}}$)", fontsize=14)
    ax.set_title(
        r"$B^{+}$ sidebands: $c\bar{c}$ NOT from $B^{+}$ decay",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2)

    # Right panel: zoomed around J/ψ peak
    ax2 = axes[1]
    h_sb_zoom, _ = np.histogram(mass_sideband, bins=be_zoom)

    ax2.fill_between(bc_zoom, h_sb_zoom, step="mid", alpha=0.3, color="#d62728")
    ax2.step(bc_zoom, h_sb_zoom, where="mid", color="#d62728", linewidth=2.0)
    draw_resonance_lines(ax2, jpsi_zoom_min, jpsi_zoom_max, ax2.get_ylim()[1])
    ax2.set_xlim(jpsi_zoom_min, jpsi_zoom_max)
    ax2.set_ylim(0, None)
    ax2.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax2.set_ylabel(f"Candidates / ({bw_zoom:.0f} MeV/$c^{{2}}$)", fontsize=14)
    ax2.set_title(r"Zoom: $J/\psi$ region", fontsize=14)
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        r"$c\bar{c}$ Background Search — $B^{+}$ Sidebands "
        rf"($M(B^{{+}}) < {BU_SIDEBAND_LOW:.0f}$ or $> {BU_SIDEBAND_HIGH:.0f}$ MeV/$c^{{2}}$)",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 2: Zoomed left and right sideband regions (B⁺ sideband data) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left sideband zoom
    ax = axes[0]
    ax.fill_between(
        bin_centers_left,
        h_sideband_left,
        step="mid",
        alpha=0.3,
        color="#d62728",
    )
    ax.step(
        bin_centers_left,
        h_sideband_left,
        where="mid",
        color="#d62728",
        linewidth=2.0,
    )
    draw_resonance_lines(ax, M_LPKM_MIN, LEFT_SIDEBAND_MAX, ax.get_ylim()[1])
    ax.set_xlim(M_LPKM_MIN, LEFT_SIDEBAND_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax.set_ylabel(f"Candidates / ({bin_width_left:.0f} MeV/$c^{{2}}$)", fontsize=14)
    ax.set_title(r"Left region ($\eta_c$ neighbourhood)", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")

    # Right sideband zoom
    ax = axes[1]
    ax.fill_between(
        bin_centers_right,
        h_sideband_right,
        step="mid",
        alpha=0.3,
        color="#d62728",
    )
    ax.step(
        bin_centers_right,
        h_sideband_right,
        where="mid",
        color="#d62728",
        linewidth=2.0,
    )
    draw_resonance_lines(ax, RIGHT_SIDEBAND_MIN, M_LPKM_MAX, ax.get_ylim()[1])
    ax.set_xlim(RIGHT_SIDEBAND_MIN, M_LPKM_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax.set_ylabel(f"Candidates / ({bin_width_right:.0f} MeV/$c^{{2}}$)", fontsize=14)
    ax.set_title(r"Right region ($\psi(2S)$ neighbourhood)", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        r"Zoomed Regions — $B^{+}$ Sidebands",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # --- Page 3: Overlay signal vs sideband (normalised) ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Normalise to unit area for shape comparison
    h_sig_norm = h_signal / (h_signal.sum() * bin_width_full) if h_signal.sum() > 0 else h_signal
    h_sb_norm = (
        h_sideband / (h_sideband.sum() * bin_width_full) if h_sideband.sum() > 0 else h_sideband
    )

    ax.step(
        bin_centers_full,
        h_sig_norm,
        where="mid",
        color="#1f77b4",
        linewidth=2.0,
        label=r"$B^{+}$ signal window",
    )
    ax.step(
        bin_centers_full,
        h_sb_norm,
        where="mid",
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label=r"$B^{+}$ sidebands",
    )
    draw_resonance_lines(ax, M_LPKM_MIN, M_LPKM_MAX, ax.get_ylim()[1])
    ax.set_xlim(M_LPKM_MIN, M_LPKM_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=14)
    ax.set_ylabel(r"Normalised density [MeV$^{-1}$]", fontsize=14)
    ax.set_title(
        r"Shape comparison: $B^{+}$ signal vs sidebands (normalised)",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="upper right", ncol=2)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print(f"  Sideband plots saved to {sideband_plot_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("cc̄ BACKGROUND SEARCH STUDY COMPLETE")
print("=" * 80)
print("\n  Outputs:")
print(f"    {signal_plot_path}")
print(f"    {sideband_plot_path}")
print(f"    {summary_csv_path}")
print(f"\n  B⁺ signal window: {len(mass_signal):,} events")
print(f"  B⁺ sidebands:     {len(mass_sideband):,} events")
print("=" * 80)
