"""
Sideband Background Study — Phase B: Template Extraction.

Ported from: archive/analysis/studies/sideband_background/extract_template.py

Extracts M(Λ̄pK⁻) background template from B⁺ mass sideband regions.
The template can be used as a data-driven background model in mass fitting.

Method:
  1. Select events in near-signal M(B⁺) sidebands (near-left + right)
  2. Create M(Λ̄pK⁻) histogram
  3. Optionally smooth with KDE
  4. Save template as ROOT histogram and RooFit workspace
  5. Visualise raw vs smoothed template

Snakemake injects:
  snakemake.params.config_dir / cache_dir / output_dir
  snakemake.output.plot / root
"""

import sys
from pathlib import Path

# Ensure the project root (analysis/) is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
study_dir = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(study_dir) not in sys.path:
    sys.path.insert(0, str(study_dir))

import awkward as ak  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import ROOT  # noqa: E402
from config import CHARMONIUM_LINES, MASS_CONFIG  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from modules.cache_manager import CacheManager  # noqa: E402
from modules.data_handler import TOMLConfig  # noqa: E402
from modules.exceptions import AnalysisError  # noqa: E402

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
plot_path = snakemake.output.plot  # noqa: F821
root_path = snakemake.output.root  # noqa: F821

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

# Template sideband regions
TEMPLATE_REGIONS = [
    MASS_CONFIG.TEMPLATE_SIDEBAND_LEFT,
    MASS_CONFIG.TEMPLATE_SIDEBAND_RIGHT,
]
SIDEBAND_LABEL = "Near sidebands [4500-5150] + [5330-5500]"

# KDE smoothing bandwidth (in bins)
KDE_SIGMA = 3.0


# ---------------------------------------------------------------------------
# Helpers
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
    br = events[branch_name]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    return br


# ---------------------------------------------------------------------------
# Load Step 2 cached data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SIDEBAND BACKGROUND: TEMPLATE EXTRACTION")
print("=" * 80)
print(f"  Sideband regions: {SIDEBAND_LABEL}")

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

# Combine data
print("\n[Loading real data]")
data_arrays = []
for year in data_dict:
    for track_type in data_dict[year]:
        arr = data_dict[year][track_type]
        if hasattr(arr, "layout"):
            data_arrays.append(arr)
all_data = ak.concatenate(data_arrays, axis=0)
print(f"  Total data (after Lambda cuts): {len(all_data):,}")

# Identify branches
mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_MM"

mlpk_all = np.asarray(ak.to_numpy(_get_flat_branch(all_data, mass_branch)))
bu_mass_all = np.asarray(ak.to_numpy(_get_flat_branch(all_data, bu_mass_branch)))

# ---------------------------------------------------------------------------
# Select sideband events
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SELECTING SIDEBAND EVENTS")
print("=" * 80)

sideband_mask = np.zeros(len(bu_mass_all), dtype=bool)
for mbu_min, mbu_max in TEMPLATE_REGIONS:
    region_mask = (bu_mass_all > mbu_min) & (bu_mass_all < mbu_max)
    n_region = np.sum(region_mask)
    print(f"  M(B⁺) in [{mbu_min:.0f}, {mbu_max:.0f}]: {n_region:,} events")
    sideband_mask |= region_mask

mlpk_sideband = mlpk_all[sideband_mask]
# Restrict to M(Λ̄pK⁻) range
in_range = (mlpk_sideband >= MASS_CONFIG.MLPK_MIN) & (mlpk_sideband <= MASS_CONFIG.MLPK_MAX)
mlpk_sideband = mlpk_sideband[in_range]
print(f"\n  Total sideband events in M(Λ̄pK⁻) range: {len(mlpk_sideband):,}")

if len(mlpk_sideband) == 0:
    raise AnalysisError("No events found in sideband regions!")

# ---------------------------------------------------------------------------
# Create histograms
# ---------------------------------------------------------------------------
bin_edges = np.linspace(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX, MASS_CONFIG.N_BINS_MLPK + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width = bin_edges[1] - bin_edges[0]

h_raw, _ = np.histogram(mlpk_sideband, bins=bin_edges)

# Smooth with Gaussian filter
h_smooth = gaussian_filter1d(h_raw.astype(float), sigma=KDE_SIGMA)
# Preserve integral
if np.sum(h_smooth) > 0:
    h_smooth *= np.sum(h_raw) / np.sum(h_smooth)

print(f"  Raw histogram: {MASS_CONFIG.N_BINS_MLPK} bins, integral = {np.sum(h_raw)}")
print(f"  Smoothed histogram: sigma = {KDE_SIGMA} bins, integral = {np.sum(h_smooth):.0f}")

# ---------------------------------------------------------------------------
# Save to ROOT file
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING ROOT FILE")
print("=" * 80)

root_out = Path(root_path)
root_out.parent.mkdir(exist_ok=True, parents=True)

tfile = ROOT.TFile.Open(str(root_out), "RECREATE")

# Raw histogram
h_root_raw = ROOT.TH1D(
    "background_template_raw",
    "Background template (raw)",
    MASS_CONFIG.N_BINS_MLPK,
    MASS_CONFIG.MLPK_MIN,
    MASS_CONFIG.MLPK_MAX,
)
h_root_raw.Sumw2()
for i in range(MASS_CONFIG.N_BINS_MLPK):
    h_root_raw.SetBinContent(i + 1, h_raw[i])
    h_root_raw.SetBinError(i + 1, np.sqrt(max(h_raw[i], 0)))
h_root_raw.Write()

# Smoothed histogram
h_root_smooth = ROOT.TH1D(
    "background_template_smooth",
    "Background template (smoothed)",
    MASS_CONFIG.N_BINS_MLPK,
    MASS_CONFIG.MLPK_MIN,
    MASS_CONFIG.MLPK_MAX,
)
h_root_smooth.Sumw2()
for i in range(MASS_CONFIG.N_BINS_MLPK):
    h_root_smooth.SetBinContent(i + 1, h_smooth[i])
    h_root_smooth.SetBinError(i + 1, np.sqrt(max(h_smooth[i], 0)))
h_root_smooth.Write()

# Normalised versions
h_root_raw_norm = h_root_raw.Clone("background_template_raw_normalized")
if h_root_raw_norm.Integral() > 0:
    h_root_raw_norm.Scale(1.0 / h_root_raw_norm.Integral())
h_root_raw_norm.Write()

h_root_smooth_norm = h_root_smooth.Clone("background_template_smooth_normalized")
if h_root_smooth_norm.Integral() > 0:
    h_root_smooth_norm.Scale(1.0 / h_root_smooth_norm.Integral())
h_root_smooth_norm.Write()

# RooFit workspace
mass_var = ROOT.RooRealVar(
    "mass",
    "M(LpKm)",
    MASS_CONFIG.MLPK_MIN,
    MASS_CONFIG.MLPK_MAX,
    "MeV/c^{2}",
)
data_hist = ROOT.RooDataHist(
    "bkg_template_datahist",
    "Background template",
    ROOT.RooArgList(mass_var),
    h_root_smooth,
)
hist_pdf = ROOT.RooHistPdf(
    "bkg_template_pdf",
    "Background template PDF",
    ROOT.RooArgSet(mass_var),
    data_hist,
    2,
)
workspace = ROOT.RooWorkspace("w_bkg_template", "Background Template Workspace")
getattr(workspace, "import")(hist_pdf)
workspace.Write()

tfile.Close()
print(f"  Saved: {root_path}")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

plot_out = Path(plot_path)
plot_out.parent.mkdir(exist_ok=True, parents=True)

with PdfPages(plot_out) as pdf:
    # --- Page 1: Raw + smoothed template ---
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.step(
        bin_centers, h_raw, where="mid", color="#1f77b4", linewidth=1.5, label="Raw sideband data"
    )
    ax.step(
        bin_centers, h_smooth, where="mid", color="#d62728", linewidth=2, label="Smoothed template"
    )

    # Charmonium reference lines
    y_max = max(np.max(h_raw), np.max(h_smooth)) * 1.3
    for mass, clabel, color in CHARMONIUM_LINES:
        if MASS_CONFIG.MLPK_MIN < mass < MASS_CONFIG.MLPK_MAX:
            ax.axvline(mass, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.text(mass + 5, y_max * 0.85, clabel, fontsize=7, rotation=90, va="top", color="gray")

    ax.set_xlim(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^2$]", fontsize=14)
    ax.set_ylabel(f"Candidates / {bin_width:.0f} MeV", fontsize=14)
    ax.set_title("Background Template from M(B⁺) Sidebands", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")

    ax.text(
        0.02,
        0.95,
        f"Sideband: {SIDEBAND_LABEL}\nEntries: {len(mlpk_sideband):,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Normalised template shape ---
    fig, ax = plt.subplots(figsize=(12, 8))

    h_raw_norm = h_raw / (np.sum(h_raw) * bin_width) if np.sum(h_raw) > 0 else h_raw
    h_smooth_norm = h_smooth / (np.sum(h_smooth) * bin_width) if np.sum(h_smooth) > 0 else h_smooth

    ax.fill_between(bin_centers, h_smooth_norm, alpha=0.3, color="#d62728", step="mid")
    ax.step(
        bin_centers,
        h_smooth_norm,
        where="mid",
        color="#d62728",
        linewidth=2,
        label="Smoothed (normalised)",
    )
    ax.step(
        bin_centers,
        h_raw_norm,
        where="mid",
        color="#1f77b4",
        linewidth=1,
        alpha=0.7,
        label="Raw (normalised)",
    )

    for mass, clabel, color in CHARMONIUM_LINES:
        if MASS_CONFIG.MLPK_MIN < mass < MASS_CONFIG.MLPK_MAX:
            ax.axvline(mass, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlim(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^2$]", fontsize=14)
    ax.set_ylabel("Probability density [1/MeV]", fontsize=14)
    ax.set_title("Normalised Background Template", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Plots saved to {plot_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEMPLATE EXTRACTION COMPLETE")
print("=" * 80)
print("\n  Outputs:")
print(f"    {plot_path}")
print(f"    {root_path}")
print(f"\n  Sideband events: {len(mlpk_sideband):,}")
print(f"  Smoothing: Gaussian sigma = {KDE_SIGMA} bins")
print("=" * 80)
