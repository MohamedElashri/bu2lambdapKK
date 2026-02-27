"""
Sideband Background Study — Phase C: Model Comparison.

Ported from: archive/analysis/studies/sideband_background/compare_background_models.py
             archive/analysis/studies/sideband_background/template_fitter.py

Compares ARGUS (parametric) vs sideband template (data-driven) background
models for M(Λ̄pK⁻) fitting. Both use the same signal model (Voigtian for
each charmonium state), differing only in background treatment.

The yield difference provides a systematic uncertainty estimate from
background modeling.

Method:
  1. Load B⁺ signal-region data from cache
  2. Load background template from extract_template output
  3. Build signal (Voigtian × 5 states) + ARGUS background model
  4. Build signal (Voigtian × 5 states) + template background model
  5. Fit both, compare yields → systematic uncertainty

Snakemake injects:
  snakemake.params.config_dir / cache_dir / output_dir
  snakemake.input.template
  snakemake.output.plot / csv
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
import pandas as pd  # noqa: E402
import ROOT  # noqa: E402
from config import CHARMONIUM_LINES, MASS_CONFIG  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

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
template_file = snakemake.input.template  # noqa: F821
plot_path = snakemake.output.plot  # noqa: F821
csv_path = snakemake.output.csv  # noqa: F821

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

# Charmonium states for signal model
SIGNAL_STATES = [
    ("etac", "etac_1s", r"$\eta_c(1S)$"),
    ("jpsi", "jpsi", r"$J/\psi$"),
    ("chic0", "chic0", r"$\chi_{c0}$"),
    ("chic1", "chic1", r"$\chi_{c1}$"),
    ("etac_2s", "etac_2s", r"$\eta_c(2S)$"),
]

STATE_COLORS = {
    "etac": "#d62728",
    "jpsi": "#1f77b4",
    "chic0": "#2ca02c",
    "chic1": "#ff7f0e",
    "etac_2s": "#9467bd",
}

# Keep-alive list for ROOT objects
_keep_alive = []


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


def create_signal_pdf(mass_var, state_name, pdg_mass, pdg_width, resolution):
    """Create Voigtian signal PDF for a charmonium state."""
    mean = ROOT.RooRealVar(f"mean_{state_name}", f"Mean {state_name}", pdg_mass)
    mean.setConstant(True)
    ROOT.SetOwnership(mean, False)

    width = ROOT.RooRealVar(f"width_{state_name}", f"Width {state_name}", pdg_width)
    width.setConstant(True)
    ROOT.SetOwnership(width, False)

    pdf = ROOT.RooVoigtian(
        f"sig_{state_name}",
        f"Signal {state_name}",
        mass_var,
        mean,
        width,
        resolution,
    )
    ROOT.SetOwnership(pdf, False)
    _keep_alive.extend([mean, width, pdf])
    return pdf


def create_argus_background(mass_var):
    """Create ARGUS background PDF."""
    m0 = ROOT.RooRealVar("m0_argus", "ARGUS endpoint", MASS_CONFIG.MLPK_MAX + 200.0)
    m0.setConstant(True)
    ROOT.SetOwnership(m0, False)

    c = ROOT.RooRealVar("c_argus", "ARGUS slope", -20.0, -100.0, -0.1)
    ROOT.SetOwnership(c, False)

    p = ROOT.RooRealVar("p_argus", "ARGUS power", 0.5)
    p.setConstant(True)
    ROOT.SetOwnership(p, False)

    argus = ROOT.RooArgusBG("bkg_argus", "ARGUS background", mass_var, m0, c, p)
    ROOT.SetOwnership(argus, False)
    _keep_alive.extend([m0, c, p, argus])
    return argus


def load_template_background(mass_var, template_path):
    """Load sideband template background PDF from ROOT file."""
    tfile = ROOT.TFile.Open(str(template_path), "READ")
    if not tfile or tfile.IsZombie():
        raise AnalysisError(f"Cannot open template file: {template_path}")

    hist = tfile.Get("background_template_smooth")
    if not hist:
        hist = tfile.Get("background_template_raw")
    if not hist:
        tfile.Close()
        raise AnalysisError(f"Template histogram not found in {template_path}")

    hist_clone = hist.Clone("bkg_template_hist")
    hist_clone.SetDirectory(0)
    ROOT.SetOwnership(hist_clone, False)
    tfile.Close()

    # Normalise
    if hist_clone.Integral() > 0:
        hist_clone.Scale(1.0 / hist_clone.Integral())

    data_hist = ROOT.RooDataHist(
        "bkg_template_datahist",
        "Background template",
        ROOT.RooArgList(mass_var),
        hist_clone,
    )
    ROOT.SetOwnership(data_hist, False)

    pdf = ROOT.RooHistPdf(
        "bkg_template",
        "Template background",
        ROOT.RooArgSet(mass_var),
        data_hist,
        2,
    )
    ROOT.SetOwnership(pdf, False)
    _keep_alive.extend([hist_clone, data_hist, pdf])
    return pdf


def build_model(mass_var, bkg_pdf, model_name, resolution):
    """Build signal + background model."""
    pdf_list = ROOT.RooArgList()
    coef_list = ROOT.RooArgList()
    yields = {}

    for state_name, config_key, _ in SIGNAL_STATES:
        pdg_mass = config.particles["pdg_masses"].get(config_key, 3000.0)
        pdg_width = config.particles["pdg_widths"].get(config_key, 10.0)

        sig_pdf = create_signal_pdf(
            mass_var, f"{state_name}_{model_name}", pdg_mass, pdg_width, resolution
        )

        yield_var = ROOT.RooRealVar(
            f"N_{state_name}_{model_name}",
            f"Yield {state_name}",
            1000,
            0,
            1e6,
        )
        ROOT.SetOwnership(yield_var, False)
        yields[state_name] = yield_var
        pdf_list.add(sig_pdf)
        coef_list.add(yield_var)
        _keep_alive.append(yield_var)

    # Background
    bkg_yield = ROOT.RooRealVar(
        f"N_bkg_{model_name}",
        "Background yield",
        10000,
        0,
        1e7,
    )
    ROOT.SetOwnership(bkg_yield, False)
    yields["background"] = bkg_yield
    pdf_list.add(bkg_pdf)
    coef_list.add(bkg_yield)
    _keep_alive.append(bkg_yield)

    total_pdf = ROOT.RooAddPdf(
        f"model_{model_name}",
        f"Total PDF ({model_name})",
        pdf_list,
        coef_list,
    )
    ROOT.SetOwnership(total_pdf, False)
    _keep_alive.append(total_pdf)
    return total_pdf, yields


def perform_fit(model, data):
    """Perform extended maximum likelihood fit."""
    result = model.fitTo(
        data,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Strategy(2),
        ROOT.RooFit.Extended(True),
    )
    ROOT.SetOwnership(result, False)
    _keep_alive.append(result)
    return result


def extract_fit_curve(mass_var, data_hist, model, n_bins):
    """Extract fit curve and components as numpy arrays for matplotlib plotting."""
    frame = mass_var.frame(ROOT.RooFit.Bins(n_bins))
    data_hist.plotOn(frame)
    model.plotOn(frame)

    # Total curve
    curve_total = frame.getCurve("model_*")
    if curve_total is None:
        # Try getting the last curve
        curve_total = frame.getObject(frame.numItems() - 1)

    x_vals = []
    y_vals = []
    if curve_total:
        for i in range(curve_total.GetN()):
            x_vals.append(curve_total.GetPointX(i))
            y_vals.append(curve_total.GetPointY(i))

    return np.array(x_vals), np.array(y_vals)


# ---------------------------------------------------------------------------
# Load Step 2 cached data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SIDEBAND BACKGROUND: MODEL COMPARISON")
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

# Select B⁺ signal region
bu_signal_min = MASS_CONFIG.SIGNAL_MIN
bu_signal_max = MASS_CONFIG.SIGNAL_MAX
signal_mask = (bu_mass_all > bu_signal_min) & (bu_mass_all < bu_signal_max)
mlpk_signal = mlpk_all[signal_mask]

# Restrict to M(Λ̄pK⁻) range
in_range = (mlpk_signal >= MASS_CONFIG.MLPK_MIN) & (mlpk_signal <= MASS_CONFIG.MLPK_MAX)
mlpk_signal = mlpk_signal[in_range]
print(
    f"  Events in B⁺ signal region [{bu_signal_min:.0f}-{bu_signal_max:.0f}]: {len(mlpk_signal):,}"
)

if len(mlpk_signal) == 0:
    raise AnalysisError("No events in B⁺ signal region!")

# ---------------------------------------------------------------------------
# Create RooFit dataset
# ---------------------------------------------------------------------------
print("\n[Creating RooFit dataset]")

mass_var = ROOT.RooRealVar(
    "M_LpKm",
    "M(LpKm)",
    MASS_CONFIG.MLPK_MIN,
    MASS_CONFIG.MLPK_MAX,
    "MeV/c^{2}",
)
mass_var.setBins(MASS_CONFIG.N_BINS_MLPK)
ROOT.SetOwnership(mass_var, False)

# Fill RooDataSet
data_set = ROOT.RooDataSet("data", "Data", ROOT.RooArgSet(mass_var))
ROOT.SetOwnership(data_set, False)
for m in mlpk_signal:
    mass_var.setVal(float(m))
    data_set.add(ROOT.RooArgSet(mass_var))
print(f"  RooDataSet entries: {data_set.numEntries():,}")

# Binned data for fitting
roo_data_hist = ROOT.RooDataHist(
    "data_hist",
    "Binned data",
    ROOT.RooArgSet(mass_var),
    data_set,
)
ROOT.SetOwnership(roo_data_hist, False)

# ---------------------------------------------------------------------------
# Build and fit ARGUS model
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FIT 1: ARGUS BACKGROUND")
print("=" * 80)

resolution_argus = ROOT.RooRealVar("sigma_res_argus", "Resolution", 8.0, 1.0, 30.0)
ROOT.SetOwnership(resolution_argus, False)
_keep_alive.append(resolution_argus)

bkg_argus = create_argus_background(mass_var)
model_argus, yields_argus = build_model(mass_var, bkg_argus, "argus", resolution_argus)
result_argus = perform_fit(model_argus, roo_data_hist)
print(f"  Fit status: {result_argus.status()}, EDM: {result_argus.edm():.2e}")

for state_name, _, label in SIGNAL_STATES:
    y = yields_argus[state_name]
    print(f"    N({label}) = {y.getVal():.0f} +/- {y.getError():.0f}")
y_bkg = yields_argus["background"]
print(f"    N(bkg) = {y_bkg.getVal():.0f} +/- {y_bkg.getError():.0f}")

# ---------------------------------------------------------------------------
# Build and fit Template model
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FIT 2: TEMPLATE BACKGROUND")
print("=" * 80)

resolution_template = ROOT.RooRealVar("sigma_res_template", "Resolution", 8.0, 1.0, 30.0)
ROOT.SetOwnership(resolution_template, False)
_keep_alive.append(resolution_template)

bkg_template = load_template_background(mass_var, template_file)
model_template, yields_template = build_model(
    mass_var, bkg_template, "template", resolution_template
)
result_template = perform_fit(model_template, roo_data_hist)
print(f"  Fit status: {result_template.status()}, EDM: {result_template.edm():.2e}")

for state_name, _, label in SIGNAL_STATES:
    y = yields_template[state_name]
    print(f"    N({label}) = {y.getVal():.0f} +/- {y.getError():.0f}")
y_bkg = yields_template["background"]
print(f"    N(bkg) = {y_bkg.getVal():.0f} +/- {y_bkg.getError():.0f}")

# ---------------------------------------------------------------------------
# Comparison results
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("YIELD COMPARISON")
print("=" * 80)

comparison_rows = []
for state_name, _, label in SIGNAL_STATES:
    n_argus = yields_argus[state_name].getVal()
    e_argus = yields_argus[state_name].getError()
    n_template = yields_template[state_name].getVal()
    e_template = yields_template[state_name].getError()
    diff = n_template - n_argus
    rel_diff = diff / n_argus * 100 if n_argus != 0 else 0.0

    print(
        f"  {label:15s}  ARGUS: {n_argus:8.0f} +/- {e_argus:6.0f}  "
        f"Template: {n_template:8.0f} +/- {e_template:6.0f}  "
        f"Δ = {diff:+8.0f} ({rel_diff:+.1f}%)"
    )

    comparison_rows.append(
        {
            "state": state_name,
            "label": label.replace("$", "").replace("\\", ""),
            "yield_argus": round(n_argus, 1),
            "error_argus": round(e_argus, 1),
            "yield_template": round(n_template, 1),
            "error_template": round(e_template, 1),
            "difference": round(diff, 1),
            "relative_diff_pct": round(rel_diff, 2),
            "systematic_pct": round(abs(rel_diff), 2),
            "fit_status_argus": result_argus.status(),
            "fit_status_template": result_template.status(),
        }
    )

# Background row
n_bkg_a = yields_argus["background"].getVal()
e_bkg_a = yields_argus["background"].getError()
n_bkg_t = yields_template["background"].getVal()
e_bkg_t = yields_template["background"].getError()
diff_bkg = n_bkg_t - n_bkg_a
rel_diff_bkg = diff_bkg / n_bkg_a * 100 if n_bkg_a != 0 else 0.0

comparison_rows.append(
    {
        "state": "background",
        "label": "Background",
        "yield_argus": round(n_bkg_a, 1),
        "error_argus": round(e_bkg_a, 1),
        "yield_template": round(n_bkg_t, 1),
        "error_template": round(e_bkg_t, 1),
        "difference": round(diff_bkg, 1),
        "relative_diff_pct": round(rel_diff_bkg, 2),
        "systematic_pct": round(abs(rel_diff_bkg), 2),
        "fit_status_argus": result_argus.status(),
        "fit_status_template": result_template.status(),
    }
)

# Save CSV
csv_out = Path(csv_path)
csv_out.parent.mkdir(exist_ok=True, parents=True)
df = pd.DataFrame(comparison_rows)
df.to_csv(csv_out, index=False)
print(f"\n  Comparison CSV saved to {csv_path}")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

plot_out = Path(plot_path)
plot_out.parent.mkdir(exist_ok=True, parents=True)

bin_edges = np.linspace(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX, MASS_CONFIG.N_BINS_MLPK + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width = bin_edges[1] - bin_edges[0]

h_data, _ = np.histogram(mlpk_signal, bins=bin_edges)
h_data_err = np.sqrt(h_data)


# Evaluate model PDF on a grid of points for matplotlib plotting
def eval_pdf_on_grid(pdf, mass_var, n_points=500):
    """Evaluate a RooAbsPdf on a grid and return (x, y) arrays scaled to data."""
    x_min = mass_var.getMin()
    x_max = mass_var.getMax()
    x_arr = np.linspace(x_min, x_max, n_points)
    y_arr = np.zeros(n_points)
    norm_set = ROOT.RooArgSet(mass_var)
    for i, x_val in enumerate(x_arr):
        mass_var.setVal(x_val)
        y_arr[i] = pdf.getVal(norm_set)
    return x_arr, y_arr


with PdfPages(plot_out) as pdf:
    # --- Page 1: Side-by-side fit plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for ax, model, yields, result, title_suffix in [
        (ax1, model_argus, yields_argus, result_argus, "ARGUS Background"),
        (ax2, model_template, yields_template, result_template, "Template Background"),
    ]:
        # Data points
        ax.errorbar(
            bin_centers,
            h_data,
            yerr=h_data_err,
            fmt="ko",
            markersize=3,
            linewidth=1,
            label="Data",
            zorder=10,
        )

        # Evaluate total model PDF on grid, scale to data
        x_grid, y_pdf = eval_pdf_on_grid(model, mass_var)
        # Scale: integral of PDF * N_events * bin_width
        n_total_yield = sum(yields[s].getVal() for s in yields)
        y_scaled = y_pdf * n_total_yield * (x_grid[1] - x_grid[0])
        ax.plot(x_grid, y_scaled, "b-", linewidth=2, label="Total fit")

        # Background component: scale by background yield fraction
        bkg_name = "bkg_argus" if "ARGUS" in title_suffix else "bkg_template"
        bkg_pdf_obj = model.pdfList().find(bkg_name)
        if bkg_pdf_obj:
            x_bkg, y_bkg_pdf = eval_pdf_on_grid(bkg_pdf_obj, mass_var)
            n_bkg_val = yields["background"].getVal()
            y_bkg_scaled = y_bkg_pdf * n_bkg_val * (x_bkg[1] - x_bkg[0])
            ax.plot(x_bkg, y_bkg_scaled, "r--", linewidth=1.5, label="Background")

        # Charmonium reference lines
        for mass, clabel, color in CHARMONIUM_LINES:
            if MASS_CONFIG.MLPK_MIN < mass < MASS_CONFIG.MLPK_MAX:
                ax.axvline(mass, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_xlim(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX)
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^2$]", fontsize=12)
        ax.set_ylabel(f"Candidates / {bin_width:.0f} MeV", fontsize=12)
        ax.set_title(title_suffix, fontsize=13)
        ax.legend(fontsize=9, loc="upper right")

        # Yield text box
        text_lines = [f"Status: {result.status()}"]
        for state_name, _, label in SIGNAL_STATES:
            n = yields[state_name].getVal()
            e = yields[state_name].getError()
            text_lines.append(f"N({label})={n:.0f}±{e:.0f}")
        n_bkg = yields["background"].getVal()
        text_lines.append(f"N(bkg)={n_bkg:.0f}")
        ax.text(
            0.02,
            0.95,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            family="monospace",
        )

    fig.suptitle(
        r"$M(\bar{\Lambda}pK^{-})$ Fit: ARGUS vs Template Background",
        fontsize=15,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Yield comparison bar chart ---
    fig, ax = plt.subplots(figsize=(12, 7))

    signal_rows = [r for r in comparison_rows if r["state"] != "background"]
    states = [r["label"] for r in signal_rows]
    x_pos = np.arange(len(states))
    bar_width = 0.35

    y_argus = [r["yield_argus"] for r in signal_rows]
    e_argus = [r["error_argus"] for r in signal_rows]
    y_template = [r["yield_template"] for r in signal_rows]
    e_template = [r["error_template"] for r in signal_rows]

    ax.bar(
        x_pos - bar_width / 2,
        y_argus,
        bar_width,
        yerr=e_argus,
        label="ARGUS bkg",
        color="#4472C4",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )
    ax.bar(
        x_pos + bar_width / 2,
        y_template,
        bar_width,
        yerr=e_template,
        label="Template bkg",
        color="#ED7D31",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )

    # Relative difference labels
    for i, r in enumerate(signal_rows):
        y_max_bar = max(r["yield_argus"], r["yield_template"])
        e_max = max(r["error_argus"], r["error_template"])
        ax.text(
            i,
            y_max_bar + e_max + 50,
            f"{r['relative_diff_pct']:+.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(states, fontsize=10)
    ax.set_ylabel("Signal Yield", fontsize=12)
    ax.set_title("Signal Yield Comparison: ARGUS vs Template Background", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: Summary table ---
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    ax.set_title("Background Model Comparison: Yield Summary", fontsize=14, pad=20)

    col_labels = [
        "State",
        "ARGUS yield",
        "ARGUS err",
        "Template yield",
        "Template err",
        "Difference",
        "Rel. diff [%]",
        "Syst. [%]",
    ]
    table_data = []
    for r in comparison_rows:
        table_data.append(
            [
                r["label"],
                f"{r['yield_argus']:.0f}",
                f"{r['error_argus']:.0f}",
                f"{r['yield_template']:.0f}",
                f"{r['error_template']:.0f}",
                f"{r['difference']:+.0f}",
                f"{r['relative_diff_pct']:+.1f}",
                f"{r['systematic_pct']:.1f}",
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
    table.scale(1.0, 2.2)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

    # Highlight background row
    last_row = len(table_data)
    for j in range(len(col_labels)):
        table[last_row, j].set_facecolor("#D9E2F3")

    # Summary text
    avg_syst = np.mean([r["systematic_pct"] for r in signal_rows])
    ax.text(
        0.5,
        0.05,
        f"Average signal systematic from background modeling: {avg_syst:.1f}%\n"
        f"ARGUS fit status: {result_argus.status()}  |  Template fit status: {result_template.status()}",
        transform=ax.transAxes,
        fontsize=11,
        ha="center",
        fontweight="bold",
    )

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Plots saved to {plot_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETE")
print("=" * 80)
print("\n  Outputs:")
print(f"    {plot_path}")
print(f"    {csv_path}")
print(f"\n  Signal region events: {len(mlpk_signal):,}")
print(f"  ARGUS fit status: {result_argus.status()}")
print(f"  Template fit status: {result_template.status()}")
print(f"  Average signal systematic: {avg_syst:.1f}%")
print("=" * 80)
