"""
Standalone study: Signal Efficiency from M(B⁺) Fits.

Ported from: archive/analysis/studies/feedback_dec2024/study3_signal_efficiency_fits.py

What it does:
  Fits M(B⁺) mass distribution with Crystal Ball (signal) + ARGUS (background)
  for two categories:
    1. All events (before optimisable selection cuts)
    2. Events passing all selection cuts
  Signal efficiency = (signal yield after cuts) / (signal yield before cuts)
  This is more robust than simple counting because it subtracts background via the fit.

  Performed separately for each MC signal state and for Data.

Improvements over original:
  - RooFit kept for fitting (Crystal Ball + ARGUS)
  - Fit result visualisation ported to matplotlib + mplhep (LHCb style)
  - Pull distributions below each fit plot
  - Cleaner efficiency summary table as CSV
  - Proper error propagation on efficiency
  - Uses cached Step 2 data from the main pipeline (awkward arrays)

Snakemake injects:
  snakemake.params.config_dir
  snakemake.params.cache_dir
  snakemake.params.output_dir
  snakemake.output.plot
  snakemake.output.csv
"""

import math
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
import ROOT
from matplotlib.backends.backend_pdf import PdfPages

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError

# Suppress ROOT GUI and RooFit verbosity
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
plot_path = snakemake.output.plot  # noqa: F821
csv_path = snakemake.output.csv  # noqa: F821

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

# MC states
MC_STATES = ["jpsi", "etac", "chic0", "chic1"]

STATE_LABELS = {
    "jpsi": r"$J/\psi$",
    "etac": r"$\eta_c(1S)$",
    "chic0": r"$\chi_{c0}$",
    "chic1": r"$\chi_{c1}$",
    "data": "Data",
}

STATE_COLORS = {
    "jpsi": "#1f77b4",
    "etac": "#d62728",
    "chic0": "#2ca02c",
    "chic1": "#ff7f0e",
    "data": "#7f7f7f",
}

# Fit range for M(B⁺)
FIT_RANGE = (5150.0, 5450.0)
N_FIT_BINS = 60

# Selection cuts (same as study 1 / manual_cuts in selection.toml)
CUTS_ORDER = [
    ("Bu_DTF_chi2", "less", 30.0),
    ("Bu_FDCHI2_OWNPV", "greater", 100.0),
    ("Bu_IPCHI2_OWNPV", "less", 10.0),
    ("Bu_PT", "greater", 3000.0),
    ("h1_ProbNNk", "greater", 0.1),
    ("h2_ProbNNk", "greater", 0.1),
    ("p_ProbNNp", "greater", 0.1),
]


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
    """Get a branch as a flat (non-jagged) array, using ak.firsts if needed."""
    br = events[branch_name]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    return br


def build_all_cuts_mask(events):
    """Build boolean mask for events passing ALL selection cuts."""
    ref = _get_flat_branch(events, "Bu_MM_corrected")
    mask = ak.ones_like(ref, dtype=bool)
    for branch, cut_type, value in CUTS_ORDER:
        if branch not in events.fields:
            print(f"  WARNING: branch '{branch}' not found, skipping")
            continue
        br = _get_flat_branch(events, branch)
        if cut_type == "greater":
            mask = mask & (br > value)
        else:
            mask = mask & (br < value)
    return mask


def calculate_efficiency(n_total, n_total_err, n_pass, n_pass_err):
    """
    Calculate efficiency with error propagation.

    ε = N_pass / N_total
    σ_ε = sqrt((σ_pass/N_total)² + (N_pass·σ_total/N_total²)²)
    """
    if n_total <= 0:
        return 0.0, 0.0
    eff = n_pass / n_total
    term1 = (n_pass_err / n_total) ** 2
    term2 = ((n_pass * n_total_err) / (n_total**2)) ** 2
    eff_err = math.sqrt(term1 + term2)
    return eff, eff_err


# ---------------------------------------------------------------------------
# RooFit: fit M(B⁺) with Crystal Ball + ARGUS
# ---------------------------------------------------------------------------
def fit_bu_mass(mass_np, label=""):
    """
    Fit M(B⁺) distribution with Crystal Ball (signal) + ARGUS (background).

    Args:
        mass_np: numpy array of M(B⁺) values in the fit range
        label: descriptive label for printout

    Returns:
        dict with keys: n_signal, n_signal_err, n_background, n_background_err,
                        mean, sigma, chi2_ndf, status,
                        bin_centers, h_data, h_total, h_signal, h_background, h_pull
    """
    n_events = len(mass_np)
    print(f"  [{label}] Events in fit range: {n_events:,}")

    if n_events < 100:
        print(f"  WARNING: Too few events ({n_events}), skipping fit")
        return None

    # Create mass observable
    mass = ROOT.RooRealVar("Bu_mass", "M(B^{+}) [MeV/c^{2}]", FIT_RANGE[0], FIT_RANGE[1])
    mass.setBins(N_FIT_BINS)

    # Fill RooDataSet
    dataset = ROOT.RooDataSet("data", "Data", ROOT.RooArgSet(mass))
    for m in mass_np:
        mass.setVal(float(m))
        dataset.add(ROOT.RooArgSet(mass))

    # Convert to binned (faster)
    datahist = ROOT.RooDataHist("datahist", "Binned data", ROOT.RooArgSet(mass), dataset)

    # Signal: Crystal Ball
    mean_var = ROOT.RooRealVar("mean", "mean", 5279.0, 5270.0, 5290.0)
    sigma_var = ROOT.RooRealVar("sigma", "sigma", 10.0, 3.0, 25.0)
    alpha_var = ROOT.RooRealVar("alpha", "alpha", 1.5, 0.5, 5.0)
    n_var = ROOT.RooRealVar("n_cb", "n", 2.0, 0.5, 10.0)
    signal_pdf = ROOT.RooCBShape(
        "signal", "Crystal Ball", mass, mean_var, sigma_var, alpha_var, n_var
    )

    # Background: ARGUS
    m0_var = ROOT.RooRealVar("m0", "ARGUS endpoint", 5290.0)
    m0_var.setConstant(True)
    c_var = ROOT.RooRealVar("c_argus", "ARGUS shape", -20.0, -100.0, -0.1)
    bkg_pdf = ROOT.RooArgusBG("background", "ARGUS background", mass, m0_var, c_var)

    # Yields
    n_sig_init = max(n_events * 0.1, 10)
    n_bkg_init = max(n_events * 0.9, 10)
    n_signal = ROOT.RooRealVar("n_signal", "Signal yield", n_sig_init, 0, n_events * 2)
    n_background = ROOT.RooRealVar("n_background", "Background yield", n_bkg_init, 0, n_events * 2)

    # Total model
    model = ROOT.RooAddPdf(
        "model",
        "Signal + Background",
        ROOT.RooArgList(signal_pdf, bkg_pdf),
        ROOT.RooArgList(n_signal, n_background),
    )

    # Fit
    fit_result = model.fitTo(
        datahist,
        ROOT.RooFit.Save(),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Strategy(2),
        ROOT.RooFit.Extended(True),
    )

    status = fit_result.status()
    if status != 0:
        print(f"  WARNING: Fit did not converge (status={status})")

    n_sig_val = n_signal.getVal()
    n_sig_err = n_signal.getError()
    n_bkg_val = n_background.getVal()
    n_bkg_err = n_background.getError()
    mean_val = mean_var.getVal()
    sigma_val = sigma_var.getVal()

    print(f"  Signal: {n_sig_val:.0f} ± {n_sig_err:.0f}")
    print(f"  Background: {n_bkg_val:.0f} ± {n_bkg_err:.0f}")
    print(f"  Mean: {mean_val:.2f} MeV, Sigma: {sigma_val:.2f} MeV")

    # Extract curves for matplotlib plotting
    frame = mass.frame(ROOT.RooFit.Bins(N_FIT_BINS))
    datahist.plotOn(frame, ROOT.RooFit.Name("data"))
    model.plotOn(frame, ROOT.RooFit.Name("total"))
    model.plotOn(frame, ROOT.RooFit.Components("signal"), ROOT.RooFit.Name("sig"))
    model.plotOn(frame, ROOT.RooFit.Components("background"), ROOT.RooFit.Name("bkg"))

    # Extract histogram data for matplotlib
    bin_width = (FIT_RANGE[1] - FIT_RANGE[0]) / N_FIT_BINS
    bin_edges = np.linspace(FIT_RANGE[0], FIT_RANGE[1], N_FIT_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Data histogram
    h_data = np.array([datahist.weight(i) for i in range(N_FIT_BINS)])
    h_data_err = np.sqrt(h_data)

    # Evaluate PDFs at bin centers
    total_norm = n_sig_val + n_bkg_val
    h_total = np.zeros(N_FIT_BINS)
    h_signal_curve = np.zeros(N_FIT_BINS)
    h_bkg_curve = np.zeros(N_FIT_BINS)

    for i, bc in enumerate(bin_centers):
        mass.setVal(bc)
        h_total[i] = model.getVal(ROOT.RooArgSet(mass)) * total_norm * bin_width
        h_signal_curve[i] = signal_pdf.getVal(ROOT.RooArgSet(mass)) * n_sig_val * bin_width
        h_bkg_curve[i] = bkg_pdf.getVal(ROOT.RooArgSet(mass)) * n_bkg_val * bin_width

    # Pull distribution
    h_pull = np.zeros(N_FIT_BINS)
    for i in range(N_FIT_BINS):
        if h_data_err[i] > 0:
            h_pull[i] = (h_data[i] - h_total[i]) / h_data_err[i]

    # Chi2/ndf
    chi2_frame = mass.frame()
    datahist.plotOn(chi2_frame)
    model.plotOn(chi2_frame)
    chi2_ndf = chi2_frame.chiSquare()

    return {
        "n_signal": n_sig_val,
        "n_signal_err": n_sig_err,
        "n_background": n_bkg_val,
        "n_background_err": n_bkg_err,
        "mean": mean_val,
        "sigma": sigma_val,
        "chi2_ndf": chi2_ndf,
        "status": status,
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "h_data": h_data,
        "h_data_err": h_data_err,
        "h_total": h_total,
        "h_signal": h_signal_curve,
        "h_background": h_bkg_curve,
        "h_pull": h_pull,
    }


# ---------------------------------------------------------------------------
# Matplotlib: plot fit result with pull distribution
# ---------------------------------------------------------------------------
def plot_fit_result(ax_fit, ax_pull, fit_res, title, color="#1f77b4"):
    """
    Plot fit result on a (fit, pull) axis pair.

    Args:
        ax_fit: Axes for the fit plot (upper panel)
        ax_pull: Axes for the pull distribution (lower panel)
        fit_res: Dictionary returned by fit_bu_mass()
        title: Plot title
        color: Colour for signal component
    """
    bc = fit_res["bin_centers"]
    bw = fit_res["bin_width"]

    # Data points with error bars
    ax_fit.errorbar(
        bc,
        fit_res["h_data"],
        yerr=fit_res["h_data_err"],
        fmt="ko",
        markersize=3,
        linewidth=1,
        label="Data",
        zorder=5,
    )

    # Total fit
    ax_fit.plot(bc, fit_res["h_total"], "-", color="blue", linewidth=2, label="Total fit")

    # Signal component
    ax_fit.plot(
        bc,
        fit_res["h_signal"],
        "--",
        color=color,
        linewidth=1.5,
        label=f"Signal ({fit_res['n_signal']:.0f}±{fit_res['n_signal_err']:.0f})",
    )

    # Background component
    ax_fit.plot(
        bc,
        fit_res["h_background"],
        "--",
        color="gray",
        linewidth=1.5,
        label=f"Background ({fit_res['n_background']:.0f})",
    )

    ax_fit.set_xlim(FIT_RANGE)
    ax_fit.set_ylim(0, None)
    ax_fit.set_ylabel(f"Candidates / ({bw:.1f} MeV/$c^{{2}}$)", fontsize=11)
    ax_fit.set_title(title, fontsize=12)
    ax_fit.legend(fontsize=8, loc="upper right")
    ax_fit.tick_params(labelbottom=False)

    # Pull distribution
    ax_pull.bar(bc, fit_res["h_pull"], width=bw * 0.9, color="gray", alpha=0.6)
    ax_pull.axhline(0, color="black", linewidth=0.8)
    ax_pull.axhline(3, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_pull.axhline(-3, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_pull.set_xlim(FIT_RANGE)
    ax_pull.set_ylim(-4.5, 4.5)
    ax_pull.set_xlabel(r"$M(B^{+})$ [MeV/$c^{2}$]", fontsize=11)
    ax_pull.set_ylabel("Pull", fontsize=11)


# ---------------------------------------------------------------------------
# Load Step 2 cached data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SIGNAL EFFICIENCY FITS STUDY")
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

mc_dict = cache.load("step2_mc_after_lambda", dependencies=step2_deps)
if mc_dict is None:
    raise AnalysisError(
        "Step 2 cached MC not found! Run Step 2 (load_data) first.\n"
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
print(f"  Total data: {len(all_data):,}")

# Combine MC per state
print("\n[Loading signal MC]")
mc_combined = {}
for state in MC_STATES:
    if state not in mc_dict:
        print(f"  WARNING: No MC for {state}, skipping")
        continue
    arrays = []
    for year in mc_dict[state]:
        for track_type in mc_dict[state][year]:
            arr = mc_dict[state][year][track_type]
            if hasattr(arr, "layout"):
                arrays.append(arr)
    if arrays:
        mc_combined[state] = ak.concatenate(arrays, axis=0)
        print(f"  MC/{state}: {len(mc_combined[state]):,} events")

# Identify B⁺ mass branch
bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"
print(f"\n  B+ mass branch: {bu_mass_branch}")

# ---------------------------------------------------------------------------
# Perform fits: MC (per state) and Data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PERFORMING FITS")
print("=" * 80)

fit_results = {}  # {sample: {"all": fit_res, "pass": fit_res}}
summary_rows = []

# --- MC fits (per state) ---
for state in mc_combined:
    print(f"\n{'='*60}")
    print(f"MC: {state}")
    print(f"{'='*60}")

    events = mc_combined[state]
    bu_mass = np.asarray(ak.to_numpy(_get_flat_branch(events, bu_mass_branch)))

    # Filter to fit range
    in_range = (bu_mass >= FIT_RANGE[0]) & (bu_mass <= FIT_RANGE[1])

    # All events
    mass_all = bu_mass[in_range]
    res_all = fit_bu_mass(mass_all, label=f"MC_{state}_all")

    # Events passing cuts
    pass_mask = np.asarray(ak.to_numpy(build_all_cuts_mask(events)))
    mass_pass = bu_mass[pass_mask & in_range]
    res_pass = fit_bu_mass(mass_pass, label=f"MC_{state}_pass")

    fit_results[f"MC_{state}"] = {"all": res_all, "pass": res_pass}

    # Efficiency
    if res_all is not None and res_pass is not None:
        eff, eff_err = calculate_efficiency(
            res_all["n_signal"],
            res_all["n_signal_err"],
            res_pass["n_signal"],
            res_pass["n_signal_err"],
        )
        print(f"  → Efficiency: {eff:.4f} ± {eff_err:.4f} ({eff * 100:.2f}%)")

        summary_rows.append(
            {
                "sample": f"MC_{state}",
                "n_signal_all": round(res_all["n_signal"], 1),
                "n_signal_all_err": round(res_all["n_signal_err"], 1),
                "n_signal_pass": round(res_pass["n_signal"], 1),
                "n_signal_pass_err": round(res_pass["n_signal_err"], 1),
                "efficiency": round(eff, 6),
                "efficiency_err": round(eff_err, 6),
                "efficiency_pct": round(eff * 100, 2),
                "fit_status_all": res_all["status"],
                "fit_status_pass": res_pass["status"],
                "chi2_ndf_all": round(res_all["chi2_ndf"], 2),
                "chi2_ndf_pass": round(res_pass["chi2_ndf"], 2),
            }
        )

    # Counting method for comparison
    n_count_all = len(mass_all)
    n_count_pass = len(mass_pass)
    count_eff = n_count_pass / n_count_all if n_count_all > 0 else 0.0
    print(f"  → Counting efficiency: {count_eff:.4f} ({count_eff * 100:.2f}%)")

# --- Data fits ---
print(f"\n{'='*60}")
print("DATA")
print(f"{'='*60}")

bu_mass_data = np.asarray(ak.to_numpy(_get_flat_branch(all_data, bu_mass_branch)))
in_range_data = (bu_mass_data >= FIT_RANGE[0]) & (bu_mass_data <= FIT_RANGE[1])

mass_all_data = bu_mass_data[in_range_data]
res_all_data = fit_bu_mass(mass_all_data, label="Data_all")

pass_mask_data = np.asarray(ak.to_numpy(build_all_cuts_mask(all_data)))
mass_pass_data = bu_mass_data[pass_mask_data & in_range_data]
res_pass_data = fit_bu_mass(mass_pass_data, label="Data_pass")

fit_results["Data"] = {"all": res_all_data, "pass": res_pass_data}

if res_all_data is not None and res_pass_data is not None:
    eff, eff_err = calculate_efficiency(
        res_all_data["n_signal"],
        res_all_data["n_signal_err"],
        res_pass_data["n_signal"],
        res_pass_data["n_signal_err"],
    )
    print(f"  → Efficiency: {eff:.4f} ± {eff_err:.4f} ({eff * 100:.2f}%)")

    summary_rows.append(
        {
            "sample": "Data",
            "n_signal_all": round(res_all_data["n_signal"], 1),
            "n_signal_all_err": round(res_all_data["n_signal_err"], 1),
            "n_signal_pass": round(res_pass_data["n_signal"], 1),
            "n_signal_pass_err": round(res_pass_data["n_signal_err"], 1),
            "efficiency": round(eff, 6),
            "efficiency_err": round(eff_err, 6),
            "efficiency_pct": round(eff * 100, 2),
            "fit_status_all": res_all_data["status"],
            "fit_status_pass": res_pass_data["status"],
            "chi2_ndf_all": round(res_all_data["chi2_ndf"], 2),
            "chi2_ndf_pass": round(res_pass_data["chi2_ndf"], 2),
        }
    )

# ---------------------------------------------------------------------------
# Save summary CSV
# ---------------------------------------------------------------------------
csv_out = Path(csv_path)
csv_out.parent.mkdir(exist_ok=True, parents=True)
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(csv_out, index=False)
print(f"\n  Summary CSV saved to {csv_path}")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

plot_out = Path(plot_path)
plot_out.parent.mkdir(exist_ok=True, parents=True)

with PdfPages(plot_out) as pdf:
    # One page per sample: 2 columns (all, pass) × 2 rows (fit, pull)
    for sample_key, sample_fits in fit_results.items():
        res_all_s = sample_fits["all"]
        res_pass_s = sample_fits["pass"]

        if res_all_s is None or res_pass_s is None:
            print(f"  Skipping {sample_key} (fit failed)")
            continue

        # Determine colour
        if sample_key.startswith("MC_"):
            state = sample_key.replace("MC_", "")
            color = STATE_COLORS.get(state, "#1f77b4")
            label = STATE_LABELS.get(state, state)
        else:
            color = STATE_COLORS["data"]
            label = "Data"

        # Compute efficiency for title
        eff, eff_err = calculate_efficiency(
            res_all_s["n_signal"],
            res_all_s["n_signal_err"],
            res_pass_s["n_signal"],
            res_pass_s["n_signal_err"],
        )

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[3, 1],
            hspace=0.05,
            wspace=0.3,
        )

        ax_fit_all = fig.add_subplot(gs[0, 0])
        ax_pull_all = fig.add_subplot(gs[1, 0], sharex=ax_fit_all)
        ax_fit_pass = fig.add_subplot(gs[0, 1])
        ax_pull_pass = fig.add_subplot(gs[1, 1], sharex=ax_fit_pass)

        plot_fit_result(
            ax_fit_all,
            ax_pull_all,
            res_all_s,
            f"{label} — All events",
            color=color,
        )
        plot_fit_result(
            ax_fit_pass,
            ax_pull_pass,
            res_pass_s,
            f"{label} — Passing cuts",
            color=color,
        )

        fig.suptitle(
            rf"{label}: $\varepsilon = {eff:.4f} \pm {eff_err:.4f}$ ({eff*100:.2f}%)",
            fontsize=16,
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

    # --- Summary page: efficiency comparison ---
    fig, ax = plt.subplots(figsize=(12, 7))

    samples = []
    effs = []
    eff_errs = []
    colors = []

    for row in summary_rows:
        samples.append(row["sample"])
        effs.append(row["efficiency_pct"])
        eff_errs.append(row["efficiency_err"] * 100)
        s = row["sample"].replace("MC_", "")
        colors.append(STATE_COLORS.get(s, STATE_COLORS.get("data", "#7f7f7f")))

    x_pos = np.arange(len(samples))
    ax.bar(x_pos, effs, yerr=eff_errs, color=colors, alpha=0.8, capsize=5, edgecolor="black")

    # Labels
    display_labels = []
    for s in samples:
        s_clean = s.replace("MC_", "")
        display_labels.append(STATE_LABELS.get(s_clean, s))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.set_ylabel("Signal Efficiency [%]", fontsize=14)
    ax.set_title(
        r"Signal Efficiency from $M(B^{+})$ Fits (Crystal Ball + ARGUS)",
        fontsize=14,
    )
    ax.set_ylim(0, max(effs) * 1.3 if effs else 100)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (e, ee) in enumerate(zip(effs, eff_errs)):
        ax.text(i, e + ee + 1, f"{e:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Summary table page ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_title("Signal Efficiency Summary Table", fontsize=14, pad=20)

    col_labels = [
        "Sample",
        "N_sig (all)",
        "N_sig (pass)",
        "Efficiency",
        "χ²/ndf (all)",
        "χ²/ndf (pass)",
        "Status",
    ]
    table_data = []
    for row in summary_rows:
        status_str = "OK" if row["fit_status_all"] == 0 and row["fit_status_pass"] == 0 else "WARN"
        table_data.append(
            [
                row["sample"],
                f"{row['n_signal_all']:.0f} ± {row['n_signal_all_err']:.0f}",
                f"{row['n_signal_pass']:.0f} ± {row['n_signal_pass_err']:.0f}",
                f"{row['efficiency_pct']:.2f} ± {row['efficiency_err']*100:.2f}%",
                f"{row['chi2_ndf_all']:.2f}",
                f"{row['chi2_ndf_pass']:.2f}",
                status_str,
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Plots saved to {plot_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SIGNAL EFFICIENCY FITS STUDY COMPLETE")
print("=" * 80)
print("\n  Outputs:")
print(f"    {plot_path}")
print(f"    {csv_path}")
print(f"\n  {len(mc_combined)} MC states + Data fitted")
print("=" * 80)
