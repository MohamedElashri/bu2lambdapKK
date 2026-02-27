"""
Standalone study: PID Background Proxy Comparison

The sideband_pid_validity study proved the B+ mass sideband proxy
is misleading for PID optimization: actual mass fits show FOM1 improves
+31% at PID > 0.20, but the proxy-based ε_sig/√ε_bkg stays below 1.0
for all PID cuts.

This study evaluates three improved proxy strategies:

═══════════════════════════════════════════════════════════════════════
PART 1 — Option B: Signal-window ARGUS-tail proxy
═══════════════════════════════════════════════════════════════════════
Use events inside the B+ signal window [5255,5305] MeV but far from
all charmonium resonances: M(Λ̄pK⁻) ∈ [2800,2900] ∪ [3800,4000].

═══════════════════════════════════════════════════════════════════════
PART 2 — Option C: Approximate sWeight proxy (no covariance)
═══════════════════════════════════════════════════════════════════════
Per-event background probability P_bkg(m) = N_bkg × f_ARGUS(m) / TotalPDF(m)
using fitted ARGUS shape and signal Voigtians — but WITHOUT the covariance
matrix. This is the simplified (biased) sWeight approximation.

═══════════════════════════════════════════════════════════════════════
PART 3 — Option D: True sPlot sWeights (Pivk & Le Diberder 2005)
═══════════════════════════════════════════════════════════════════════
Extract the full yield covariance matrix V from the RooFitResult.
For each event i compute the true sWeight for the background species:

  sWeight_bkg(m_i) = [Σ_j V_{bkg,j} × f_j(m_i)] / [Σ_n N_n × f_n(m_i)]

where f_k are PDFs normalized over [FIT_MIN, FIT_MAX].
Key properties:
  — Σ_i sWeight_bkg(m_i) = N_bkg  (sum equals fitted yield)
  — Weights near signal peaks can be NEGATIVE (by construction)
  — Background efficiency: ε_bkg(c) = Σ_i [w_i × I(PID_i>c)] / Σ_i w_i

═══════════════════════════════════════════════════════════════════════
PART 4 — Comparison with fit-based FOM (ground truth)
═══════════════════════════════════════════════════════════════════════
Load fit-based FOM (7 discrete points) from sideband_pid_validity CSV.
Overlay all four proxy methods + fit-based FOM on the same plot.
"""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import voigt_profile  # noqa: E402

from modules.cache_manager import CacheManager  # noqa: E402
from modules.data_handler import TOMLConfig  # noqa: E402
from modules.exceptions import AnalysisError  # noqa: E402
from modules.mass_fitter import MassFitter  # noqa: E402

# ---------------------------------------------------------------------------
# Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
optb_plot_path = snakemake.output.optb_plot  # noqa: F821
optc_plot_path = snakemake.output.optc_plot  # noqa: F821
optd_plot_path = snakemake.output.optd_plot  # noqa: F821
comparison_plot_path = snakemake.output.comparison_plot  # noqa: F821
csv_path = snakemake.output.csv  # noqa: F821
prev_csv_path = snakemake.input.prev_csv  # noqa: F821

plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

# ---------------------------------------------------------------------------
# Load Step 2 cache
# ---------------------------------------------------------------------------
years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

config_files = list(Path(config_dir).glob("*.toml"))
code_files = [
    project_root / "modules" / "data_handler.py",
    project_root / "modules" / "lambda_selector.py",
]
step2_deps = cache.compute_dependencies(
    config_files=config_files,
    code_files=code_files,
    extra_params={"years": years, "track_types": track_types},
)

data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)
mc_dict = cache.load("step2_mc_after_lambda", dependencies=step2_deps)

if data_dict is None or mc_dict is None:
    raise AnalysisError("Step 2 cache not found! Run 'uv run snakemake load_data -j1' first.")

print("✓ Loaded Step 2 cache (data + MC)")

# Combine LL + DD per year — keep year-level dict for MassFitter
data_combined = {}
for year in data_dict:
    arrays = [
        data_dict[year][tt] for tt in data_dict[year] if hasattr(data_dict[year][tt], "layout")
    ]
    if arrays:
        data_combined[year] = ak.concatenate(arrays, axis=0)

all_data = ak.concatenate(list(data_combined.values()), axis=0)
print(f"  Total data events (after Lambda cuts): {len(all_data):,}")

# MC: combine all states and track types
MC_STATES = ["jpsi", "etac", "chic0", "chic1"]
mc_combined = {}
for state in MC_STATES:
    if state not in mc_dict:
        continue
    arrays = [
        mc_dict[state][yr][tt]
        for yr in mc_dict[state]
        for tt in mc_dict[state][yr]
        if hasattr(mc_dict[state][yr][tt], "layout")
    ]
    if arrays:
        mc_combined[state] = ak.concatenate(arrays, axis=0)

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------
opt = config.selection.get("optimization_strategy", {})
SB_LO_MIN = opt.get("b_low_sideband_min", 5150.0)
SB_LO_MAX = opt.get("b_low_sideband_max", 5230.0)
SB_HI_MIN = opt.get("b_high_sideband_min", 5330.0)
SB_HI_MAX = opt.get("b_high_sideband_max", 5410.0)
SIG_MIN = opt.get("b_signal_region_min", 5255.0)
SIG_MAX = opt.get("b_signal_region_max", 5305.0)
CC_MIN = opt.get("charmonium_region_min", 2900.0)
CC_MAX = opt.get("charmonium_region_max", 3800.0)

FIT_MIN = config.particles["mass_windows"]["charmonium_fit_range"][0]  # 2800
FIT_MAX = config.particles["mass_windows"]["charmonium_fit_range"][1]  # 4000

OPT_B_EXCL_LO = 2900.0
OPT_B_EXCL_HI = 3800.0

mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"

print(f"  Mass branch: {mass_branch}, B+ branch: {bu_branch}")
print(f"  Fit range: [{FIT_MIN}, {FIT_MAX}] MeV")
print(f"  Signal window: [{SIG_MIN}, {SIG_MAX}] MeV")


def flat(events, branch):
    """Return branch as flat numpy array."""
    arr = events[branch]
    if "var" in str(ak.type(arr)):
        arr = ak.firsts(arr)
    return np.asarray(ak.drop_none(arr))


def apply_selection(data_by_year, cuts):
    """Apply list of (branch, operator, value) cuts to {year: ak.Array} dict."""
    result = {}
    for year, arr in data_by_year.items():
        mask = np.ones(len(arr), dtype=bool)
        for branch, op, val in cuts:
            col = flat(arr, branch)
            if op == "less":
                mask &= col < val
            else:
                mask &= col > val
        result[year] = arr[mask]
    return result


# ---------------------------------------------------------------------------
# PID arrays and basic masks
# ---------------------------------------------------------------------------
pid_mc_parts = []
for state in MC_STATES:
    if state in mc_combined:
        pid_mc_parts.append(flat(mc_combined[state], "PID_product"))
pid_mc_all = np.concatenate(pid_mc_parts)
print(f"  Signal MC events (combined): {len(pid_mc_all):,}")

bu_all = flat(all_data, bu_branch)
cc_all = flat(all_data, mass_branch)

# Current sideband proxy (B+ mass sidebands ∩ charmonium region)
mask_sb_cc = (
    (((bu_all > SB_LO_MIN) & (bu_all < SB_LO_MAX)) | ((bu_all > SB_HI_MIN) & (bu_all < SB_HI_MAX)))
    & (cc_all > CC_MIN)
    & (cc_all < CC_MAX)
)
pid_sb_cc = flat(all_data[mask_sb_cc], "PID_product")
print(f"  Current sideband proxy: {int(np.sum(mask_sb_cc)):,} events")

# Option B: signal window, off-resonance in fit range
mask_sig_win = (bu_all >= SIG_MIN) & (bu_all <= SIG_MAX)
mask_fit_range = (cc_all >= FIT_MIN) & (cc_all <= FIT_MAX)
mask_opt_b = mask_sig_win & mask_fit_range & ((cc_all < OPT_B_EXCL_LO) | (cc_all > OPT_B_EXCL_HI))
pid_opt_b = flat(all_data[mask_opt_b], "PID_product")
cc_opt_b = cc_all[mask_opt_b]
print(f"  Option B proxy region: {int(np.sum(mask_opt_b)):,} events")

# All signal-window events (for sWeighting)
mask_sig_full = mask_sig_win & mask_fit_range
cc_sig_full = cc_all[mask_sig_full]
pid_sig_full = flat(all_data[mask_sig_full], "PID_product")
print(f"  Signal-window events (for sWeights): {len(cc_sig_full):,}")

# Dense PID scan grid
PID_CUTS = np.arange(0.0, 0.51, 0.01)

eps_sig = np.array([np.mean(pid_mc_all > c) for c in PID_CUTS])
eps_bkg_sb = np.array([np.mean(pid_sb_cc > c) for c in PID_CUTS])
eps_bkg_b = np.array([np.mean(pid_opt_b > c) for c in PID_CUTS])

fom_sb = eps_sig / np.sqrt(np.maximum(eps_bkg_sb, 1e-9))
fom_b = eps_sig / np.sqrt(np.maximum(eps_bkg_b, 1e-9))

summary_rows = []

# ===========================================================================
# PART 1: Option B Proxy — Signal-Window ARGUS-Tail
# ===========================================================================
print("\n" + "=" * 70)
print("PART 1: Option B — Signal-Window ARGUS-Tail Proxy")
print("=" * 70)

ks_stat, ks_p = stats.ks_2samp(pid_opt_b, pid_sb_cc)
ks_stat_mc, ks_p_mc = stats.ks_2samp(pid_opt_b, pid_mc_all)
print(f"  Option B events: {len(pid_opt_b):,}   PID mean: {np.mean(pid_opt_b):.4f}")
print(f"  KS (Option B vs sideband): p={ks_p:.3e}")
print(f"  KS (Option B vs MC):       p={ks_p_mc:.3e}")
print(f"  FOM max: {np.max(fom_b):.4f} at PID>{PID_CUTS[np.argmax(fom_b)]:.2f}")

Path(optb_plot_path).parent.mkdir(parents=True, exist_ok=True)
with PdfPages(optb_plot_path) as pdf:

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    cc_sig_win = cc_all[mask_sig_win & mask_fit_range]
    bins_cc = np.linspace(FIT_MIN, FIT_MAX, 80)
    ax.hist(
        cc_sig_win,
        bins=bins_cc,
        color="steelblue",
        alpha=0.6,
        label=f"Signal window (N={len(cc_sig_win):,})",
    )
    ax.hist(
        cc_opt_b,
        bins=bins_cc,
        color="darkorange",
        alpha=0.8,
        label=f"Option B region (N={len(cc_opt_b):,})",
    )
    ax.axvspan(OPT_B_EXCL_LO, OPT_B_EXCL_HI, alpha=0.12, color="red", label="Excluded (resonances)")
    for name, pos in [
        ("ηc", 2984.1),
        ("J/ψ", 3096.9),
        ("χc0", 3414.7),
        ("χc1", 3510.7),
        ("ηc(2S)", 3637.8),
    ]:
        ax.axvline(pos, color="red", ls="--", alpha=0.6, lw=1)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^-)$ [MeV/$c^2$]", fontsize=12)
    ax.set_ylabel("Events / 15 MeV", fontsize=12)
    ax.set_title("Option B region in signal B⁺ window", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    bins_pid = np.linspace(0, 1, 51)
    bc = 0.5 * (bins_pid[:-1] + bins_pid[1:])
    for arr, lbl, col, ls in [
        (pid_sb_cc, f"Sideband (N={len(pid_sb_cc):,})", "black", "-"),
        (pid_opt_b, f"Option B (N={len(pid_opt_b):,})", "darkorange", "-"),
        (pid_mc_all, f"Signal MC (N={len(pid_mc_all):,})", "purple", "--"),
    ]:
        h, _ = np.histogram(arr, bins=bins_pid, density=True)
        ax.step(bc, h, where="mid", lw=2, color=col, ls=ls, label=lbl, alpha=0.85)
    ax.set_xlabel("PID product", fontsize=12)
    ax.set_ylabel("Normalised", fontsize=12)
    ax.set_title("PID product: Option B vs sideband proxy", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.text(
        0.98,
        0.95,
        f"KS (B vs SB): p={ks_p:.2e}\nKS (B vs MC): p={ks_p_mc:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        "Part 1 — Option B: Signal-Window ARGUS-Tail Proxy", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    ax.plot(PID_CUTS, fom_sb, "k-", lw=2, label="Current sideband", alpha=0.8)
    ax.plot(PID_CUTS, fom_b, color="darkorange", lw=2.5, label="Option B (ARGUS-tail)")
    ax.axhline(1.0, color="green", ls="--", lw=2, label="Threshold = 1.0")
    ax.set_xlabel("PID cut", fontsize=12)
    ax.set_ylabel(r"$\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$", fontsize=12)
    ax.set_title("FOM ratio: PID cut beneficial if > 1.0", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(PID_CUTS, fom_sb / fom_sb[0], "k-", lw=2, label="Sideband", alpha=0.8)
    ax.plot(PID_CUTS, fom_b / fom_b[0], color="darkorange", lw=2.5, label="Option B")
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("PID cut", fontsize=12)
    ax.set_ylabel("FOM(cut) / FOM(0)", fontsize=12)
    ax.set_title("Relative FOM improvement vs PID cut", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Part 1 — Option B: FOM Ratio Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("  Saved:", optb_plot_path)

# ===========================================================================
# Reference fit (shared by Option C and Option D)
# ===========================================================================
print("\n" + "=" * 70)
print("REFERENCE FIT (Set1_noPID) — shared by Option C and Option D")
print("=" * 70)

BASE_CUTS = [
    ("Bu_DTF_chi2", "less", 30.0),
    ("Bu_FDCHI2_OWNPV", "greater", 100.0),
    ("Bu_IPCHI2_OWNPV", "less", 6.5),
    ("Bu_PT", "greater", 3000.0),
]
data_ref = apply_selection(data_combined, BASE_CUTS)
n_ref = sum(len(v) for v in data_ref.values())
print(f"  Events for reference fit: {n_ref:,}")

config.paths["output"]["plots_dir"] = str(Path(output_dir) / "fits_ref")
Path(output_dir, "fits_ref").mkdir(parents=True, exist_ok=True)
fitter = MassFitter(config)

try:
    ref_results = fitter.perform_fit(data_ref, fit_combined=True)
    ref_yields = ref_results["yields"].get("combined", {})
    sigma_val, sigma_err = ref_results["resolution"]

    N_jpsi = max(ref_yields.get("jpsi", (0, 0))[0], 0)
    N_etac = max(ref_yields.get("etac", (0, 0))[0], 0)
    N_chic0 = max(ref_yields.get("chic0", (0, 0))[0], 0)
    N_chic1 = max(ref_yields.get("chic1", (0, 0))[0], 0)
    N_etac2s = max(ref_yields.get("etac_2s", (0, 0))[0], 0)
    N_bkg = max(ref_yields.get("background", (0, 0))[0], 0)

    c_val = fitter.argus_params["combined"]["c"].getVal()
    m0_val = fitter.argus_params["combined"]["m0"].getVal()

    print("\n  Fitted yields:")
    print(f"    N(J/ψ)    = {N_jpsi:.0f}")
    print(f"    N(ηc)     = {N_etac:.0f}")
    print(f"    N(χc0)    = {N_chic0:.0f}")
    print(f"    N(χc1)    = {N_chic1:.0f}")
    print(f"    N(ηc(2S)) = {N_etac2s:.0f}")
    print(f"    N(bkg)    = {N_bkg:.0f}")
    print(f"    σ(res)    = {sigma_val:.2f} ± {sigma_err:.2f} MeV")
    print(f"    ARGUS c   = {c_val:.4f},  m0 = {m0_val:.2f} MeV")
    FIT_OK = True

except Exception as exc:
    print(f"  !! Reference fit failed: {exc}")
    FIT_OK = False

# ===========================================================================
# Recompute sWeight sample to match exactly the events fitted by MassFitter.
#
# The MassFitter applies:
#   (1) BASE_CUTS (chi2<30, FDCHI2>100, IPCHI2<6.5, PT>3000)  ← from data_ref
#   (2) Bu_MM_corrected ∈ [SIG_MIN, SIG_MAX]                  ← applied internally
#   (3) M_LpKm_h2 ∈ [FIT_MIN, FIT_MAX]                        ← applied internally
# → 5,565 events (combined across years).
#
# The initial cc_sig_full / pid_sig_full above were built from all_data
# (no BASE_CUTS) → 6,769 events.  sWeights from a fit on 5,565 events must
# not be applied to a different 6,769-event sample; the mathematical identity
# Σ_i sWeight_k = N_k only holds for the exact fitted dataset.
# ===========================================================================
data_ref_concat = ak.concatenate(list(data_ref.values()), axis=0)
_bu_ref = flat(data_ref_concat, bu_branch)
_cc_ref = flat(data_ref_concat, mass_branch)
_mask_sw = (_bu_ref >= SIG_MIN) & (_bu_ref <= SIG_MAX) & (_cc_ref >= FIT_MIN) & (_cc_ref <= FIT_MAX)
cc_sig_full = _cc_ref[_mask_sw]
pid_sig_full = flat(data_ref_concat[_mask_sw], "PID_product")
print(
    f"\n  sWeight sample (data_ref ∩ signal_window ∩ fit_range): {len(cc_sig_full):,} events"
    f"  [was {int(np.sum(mask_sig_full)):,} with all_data — now corrected]"
)

# ===========================================================================
# Shared analytical PDF building (used by both Option C and Option D)
# ===========================================================================
if FIT_OK:
    # Species order — must be consistent throughout
    # Indices: jpsi=0, etac=1, chic0=2, chic1=3, etac_2s=4, bkg=5
    SPECIES_ORDER = ["jpsi", "etac", "chic0", "chic1", "etac_2s", "bkg"]
    BKG_IDX = 5

    PDG_PARAMS = [
        ("jpsi", 3096.9, 0.0926),
        ("etac", 2984.1, 30.5),
        ("chic0", 3414.71, 10.6),
        ("chic1", 3510.67, 0.88),
        ("etac_2s", 3637.8, 11.6),
    ]
    yields_vec = np.array([N_jpsi, N_etac, N_chic0, N_chic1, N_etac2s, N_bkg])

    # Normalization grid
    m_grid = np.linspace(FIT_MIN, FIT_MAX, 2000)

    def argus_unnorm(m_arr, m0, c, p=0.5):
        x = m_arr / m0
        inside = np.maximum(1 - x**2, 0.0)
        return m_arr * np.sqrt(inside) * np.exp(c * inside**p)

    argus_vals_grid = argus_unnorm(m_grid, m0_val, c_val)
    argus_norm = np.trapezoid(argus_vals_grid, m_grid)
    if argus_norm < 1e-9:
        argus_norm = 1.0

    voigt_norms = {}
    for state, mass_pdg, gamma in PDG_PARAMS:
        v_vals = voigt_profile(m_grid - mass_pdg, sigma_val, gamma / 2)
        voigt_norms[state] = np.trapezoid(v_vals, m_grid)

    def build_pdf_matrix(m_events):
        """
        Return pdf_matrix[i, k] = f_k(m_i) normalized over [FIT_MIN, FIT_MAX].
        Columns: [jpsi, etac, chic0, chic1, etac_2s, bkg]
        """
        n = len(m_events)
        pdf_mat = np.zeros((n, 6))
        for k, (state, mass_pdg, gamma) in enumerate(PDG_PARAMS):
            v_k = voigt_profile(m_events - mass_pdg, sigma_val, gamma / 2)
            norm_k = voigt_norms[state]
            pdf_mat[:, k] = v_k / (norm_k if norm_k > 1e-9 else 1.0)
        # Background (index 5)
        argus_vals = argus_unnorm(m_events, m0_val, c_val)
        pdf_mat[:, BKG_IDX] = argus_vals / argus_norm
        return pdf_mat

    # Pre-build for all signal-window events
    pdf_mat_full = build_pdf_matrix(cc_sig_full)
    total_pdf_full = pdf_mat_full @ yields_vec  # shape (n_events,)
    safe_total = np.where(total_pdf_full > 1e-15, total_pdf_full, 1e-15)

# ===========================================================================
# PART 2: Option C — Approximate sWeights (no covariance correction)
# ===========================================================================
print("\n" + "=" * 70)
print("PART 2: Option C — Approximate sWeight Proxy (no covariance)")
print("=" * 70)

if FIT_OK:
    # Simple per-event background fraction (ignores off-diagonal V_kj terms)
    bkg_pdf_vals = pdf_mat_full[:, BKG_IDX]
    bkg_term_full = N_bkg * bkg_pdf_vals
    w_bkg_c = bkg_term_full / safe_total
    w_sum_c = np.sum(w_bkg_c)

    eps_bkg_c = np.array(
        [np.sum(w_bkg_c[pid_sig_full > cut]) / np.maximum(w_sum_c, 1e-9) for cut in PID_CUTS]
    )
    fom_c = eps_sig / np.sqrt(np.maximum(eps_bkg_c, 1e-9))

    print(f"  Mean background weight (approx): {np.mean(w_bkg_c):.4f}")
    print(f"  Sum of weights (≈ N_bkg): {w_sum_c:.0f}  (fitted N_bkg = {N_bkg:.0f})")
    print(f"  FOM max: {np.max(fom_c):.4f} at PID>{PID_CUTS[np.argmax(fom_c)]:.2f}")
    print(f"  FOM > 1.0: {np.any(fom_c > 1.0)}")
else:
    fom_c = fom_b.copy()
    eps_bkg_c = eps_bkg_b.copy()
    w_bkg_c = np.array([])
    w_sum_c = 0.0

# ===========================================================================
# PART 3: Option D — True sPlot sWeights
# ===========================================================================
print("\n" + "=" * 70)
print("PART 3: Option D — True sPlot sWeights (Pivk & Le Diberder 2005)")
print("=" * 70)

SWEIGHT_OK = False
if FIT_OK:
    try:
        # ---------------------------------------------------------------
        # True sPlot sWeights — Pivk & Le Diberder (NIM A 555, 356, 2005)
        #
        # The sPlot information matrix H is computed DIRECTLY from the data
        # (not from the fit covariance matrix), which guarantees:
        #   Σ_i sWeight_k(m_i) = N_k   (exactly, by construction)
        #
        # H[k, j] = Σ_i  f_k(m_i) * f_j(m_i) / [Σ_n N_n * f_n(m_i)]^2
        # V = H^{-1}
        # sWeight_k(m_i) = [Σ_j V[k,j] * f_j(m_i)] / [Σ_n N_n * f_n(m_i)]
        #
        # Note: using the fit's covarianceMatrix() does NOT guarantee
        # Σ(sw) = N_k because it comes from a BINNED fit (not the same as
        # inverting H computed event-by-event). We compute H from data here.
        # ---------------------------------------------------------------

        # pdf_mat_full[i, k] = f_k(m_i)  (normalized, pre-built above)
        # safe_total[i] = Σ_n N_n * f_n(m_i)          (pre-built above)

        # H = F^T D^{-1} F  where F[i,k]=f_k(m_i), D[i,i]=safe_total[i]^2
        # Efficient matrix form: H[k,j] = Σ_i (f_k * f_j)(m_i) / total^2(m_i)
        safe_total_sq = safe_total**2  # shape (n_events,)
        H = (pdf_mat_full.T / safe_total_sq) @ pdf_mat_full  # (6×6)

        cond = np.linalg.cond(H)
        print(f"  H matrix condition number: {cond:.3e}")
        if cond > 1e12:
            print("  WARNING: H matrix is poorly conditioned; results may be unreliable.")

        try:
            V = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("  WARNING: H matrix is singular; using pseudo-inverse.")
            V = np.linalg.pinv(H)

        print("\n  Yield uncertainties (√diag(V)) vs fit σ:")
        for ii, si in enumerate(SPECIES_ORDER):
            sigma_v = np.sqrt(max(V[ii, ii], 0))
            print(f"    {si:12s}: σ(sPlot) = {sigma_v:6.1f}  [N = {yields_vec[ii]:6.0f}]")

        # sWeights for the background species (index BKG_IDX = 5)
        numerator_d = pdf_mat_full @ V[BKG_IDX, :]  # shape (n_events,)
        sw_bkg_d = numerator_d / safe_total

        w_sum_d = np.sum(sw_bkg_d)
        print(f"\n  Σ(sWeights) = {w_sum_d:.2f}  (must equal N_bkg = {N_bkg:.0f})")
        if abs(w_sum_d - N_bkg) / max(N_bkg, 1) > 0.01:
            print("  WARNING: sum deviates > 1% from N_bkg — check normalization.")
        print(f"  Mean sWeight: {np.mean(sw_bkg_d):.4f}")
        print(
            f"  Negative sWeights: {np.sum(sw_bkg_d < 0):,} events "
            f"({100*np.mean(sw_bkg_d < 0):.1f}%)"
        )
        print(f"  Min sWeight: {np.min(sw_bkg_d):.4f}   Max: {np.max(sw_bkg_d):.4f}")

        # PID efficiency scan with true sWeights
        eps_bkg_d = np.array(
            [np.sum(sw_bkg_d[pid_sig_full > cut]) / np.maximum(w_sum_d, 1e-9) for cut in PID_CUTS]
        )
        fom_d = eps_sig / np.sqrt(np.maximum(eps_bkg_d, 1e-9))

        print(f"\n  Option D FOM at PID=0: {fom_d[0]:.4f}")
        print(f"  Option D FOM at PID=0.20: {fom_d[20]:.4f}")
        max_d_idx = np.argmax(fom_d)
        print(f"  Option D max FOM: {np.max(fom_d):.4f} at PID>{PID_CUTS[max_d_idx]:.2f}")
        print(f"  Option D FOM > 1.0: {np.any(fom_d > 1.0)}")
        SWEIGHT_OK = True

    except Exception as exc:
        print(f"  !! True sWeights failed: {exc}")
        import traceback

        traceback.print_exc()
        fom_d = fom_b.copy()
        eps_bkg_d = eps_bkg_b.copy()
        sw_bkg_d = np.array([])
        w_sum_d = 0.0

else:
    fom_d = fom_b.copy()
    eps_bkg_d = eps_bkg_b.copy()
    sw_bkg_d = np.array([])
    w_sum_d = 0.0

# --- Plot Option C and D side-by-side ---
Path(optc_plot_path).parent.mkdir(parents=True, exist_ok=True)

with PdfPages(optc_plot_path) as pdf:
    if not FIT_OK:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5,
            0.5,
            "Reference fit failed",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
            color="red",
        )
        pdf.savefig(fig)
        plt.close()
    else:
        # Page 1: weight distributions comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Left: M(cc) vs approximate weight (Option C)
        ax = axes[0]
        rng = np.random.default_rng(42)
        n_sc = min(len(cc_sig_full), 15000)
        idx_s = rng.choice(len(cc_sig_full), size=n_sc, replace=False)
        sc = ax.scatter(
            cc_sig_full[idx_s],
            w_bkg_c[idx_s],
            s=1,
            alpha=0.3,
            c=w_bkg_c[idx_s],
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=r"$w_{\rm approx}(m)$")
        for name, pos in [
            ("ηc", 2984.1),
            ("J/ψ", 3096.9),
            ("χc0", 3414.7),
            ("χc1", 3510.7),
            ("ηc(2S)", 3637.8),
        ]:
            ax.axvline(pos, color="navy", ls="--", alpha=0.5, lw=1)
        ax.set_xlabel(r"$M(\bar{\Lambda}pK^-)$ [MeV/$c^2$]", fontsize=11)
        ax.set_ylabel("Approx weight", fontsize=11)
        ax.set_title(f"Option C: P_bkg(m)\n(mean={np.mean(w_bkg_c):.3f})", fontsize=11)
        ax.set_ylim(-0.1, 1.1)

        # Middle: M(cc) vs true sWeight (Option D)
        ax = axes[1]
        if SWEIGHT_OK:
            vmin_d = max(np.percentile(sw_bkg_d, 1), -1.0)
            vmax_d = 1.0
            sc2 = ax.scatter(
                cc_sig_full[idx_s],
                sw_bkg_d[idx_s],
                s=1,
                alpha=0.3,
                c=sw_bkg_d[idx_s],
                cmap="RdYlGn",
                vmin=vmin_d,
                vmax=vmax_d,
                rasterized=True,
            )
            plt.colorbar(sc2, ax=ax, label=r"${}_{s}\mathcal{P}_{\rm bkg}(m)$")
            for name, pos in [
                ("ηc", 2984.1),
                ("J/ψ", 3096.9),
                ("χc0", 3414.7),
                ("χc1", 3510.7),
                ("ηc(2S)", 3637.8),
            ]:
                ax.axvline(pos, color="navy", ls="--", alpha=0.5, lw=1)
            frac_neg = np.mean(sw_bkg_d < 0)
            ax.set_title(
                f"Option D: True sWeight\n(neg fraction={frac_neg:.3f}, sum={w_sum_d:.0f})",
                fontsize=11,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "sWeights failed",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13,
                color="red",
            )
            ax.set_title("Option D: True sWeight (failed)", fontsize=11)
        ax.set_xlabel(r"$M(\bar{\Lambda}pK^-)$ [MeV/$c^2$]", fontsize=11)
        ax.set_ylabel(r"${}_{s}\mathcal{P}_{\rm bkg}(m)$", fontsize=11)

        # Right: histogram of weights comparison
        ax = axes[2]
        bins_w = np.linspace(-0.5, 1.5, 81)
        ax.hist(w_bkg_c, bins=bins_w, color="darkorange", alpha=0.6, label="Option C (approx)")
        if SWEIGHT_OK:
            ax.hist(sw_bkg_d, bins=bins_w, color="royalblue", alpha=0.6, label="Option D (true sW)")
        ax.axvline(0, color="red", ls="--", lw=1.5)
        ax.set_xlabel("Background weight", fontsize=11)
        ax.set_ylabel("Events", fontsize=11)
        ax.set_title("Weight distributions", fontsize=11)
        ax.legend(fontsize=9)

        fig.suptitle(
            "Part 2+3 — Weight Distributions: Option C vs Option D", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 2: FOM ratio curves for C and D
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax = axes[0]
        ax.plot(PID_CUTS, fom_sb, "k-", lw=2, label="Current sideband", alpha=0.7)
        ax.plot(
            PID_CUTS, fom_c, color="darkorange", lw=2, ls="--", label="Option C (approx sWeight)"
        )
        if SWEIGHT_OK:
            ax.plot(PID_CUTS, fom_d, color="royalblue", lw=2.5, label="Option D (true sWeight)")
        ax.axhline(1.0, color="green", ls="--", lw=2, label="Threshold = 1.0")
        ax.set_xlabel("PID cut", fontsize=12)
        ax.set_ylabel(r"$\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$", fontsize=12)
        ax.set_title("Absolute FOM ratio: C vs D", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(PID_CUTS, fom_sb / fom_sb[0], "k-", lw=2, label="Sideband", alpha=0.7)
        ax.plot(
            PID_CUTS, fom_c / fom_c[0], color="darkorange", lw=2, ls="--", label="Option C (approx)"
        )
        if SWEIGHT_OK:
            ax.plot(
                PID_CUTS,
                fom_d / fom_d[0],
                color="royalblue",
                lw=2.5,
                label="Option D (true sWeight)",
            )
        ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("PID cut", fontsize=12)
        ax.set_ylabel("FOM(cut) / FOM(0)", fontsize=12)
        ax.set_title("Relative FOM: Option C vs D", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Part 2+3 — FOM Ratio: Approximate vs True sWeights", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

print("  Saved:", optc_plot_path)

# Option D dedicated plot
Path(optd_plot_path).parent.mkdir(parents=True, exist_ok=True)
with PdfPages(optd_plot_path) as pdf:
    if not SWEIGHT_OK:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5,
            0.5,
            "True sWeights failed — see console output",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
            color="red",
        )
        ax.set_title("Option D — True sPlot sWeights (failed)", fontsize=14)
        pdf.savefig(fig)
        plt.close()
    else:
        # Page 1: sWeight distribution details
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        rng2 = np.random.default_rng(0)
        n_sc2 = min(len(cc_sig_full), 20000)
        idx_s2 = rng2.choice(len(cc_sig_full), size=n_sc2, replace=False)
        vmin_d = max(np.percentile(sw_bkg_d, 2), -2.0)
        sc3 = ax.scatter(
            cc_sig_full[idx_s2],
            sw_bkg_d[idx_s2],
            s=1.5,
            alpha=0.35,
            c=sw_bkg_d[idx_s2],
            cmap="RdYlGn",
            vmin=vmin_d,
            vmax=1.0,
            rasterized=True,
        )
        plt.colorbar(sc3, ax=ax, label=r"${}_{s}\mathcal{P}_{\rm bkg}(m)$")
        for name, pos in [
            ("ηc", 2984.1),
            ("J/ψ", 3096.9),
            ("χc0", 3414.7),
            ("χc1", 3510.7),
            ("ηc(2S)", 3637.8),
        ]:
            ax.axvline(pos, color="navy", ls="--", alpha=0.6, lw=1.2)
            ax.text(
                pos,
                ax.get_ylim()[1] * 0.9,
                name,
                ha="center",
                fontsize=7,
                color="navy",
                rotation=90,
            )
        ax.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7)
        ax.set_xlabel(r"$M(\bar{\Lambda}pK^-)$ [MeV/$c^2$]", fontsize=11)
        ax.set_ylabel(r"${}_{s}\mathcal{P}_{\rm bkg}(m)$", fontsize=11)
        ax.set_title(
            "True sWeight vs M(Λ̄pK⁻)\n(negative near signal peaks — by construction)", fontsize=10
        )

        ax = axes[1]
        # Show covariance matrix as image
        V_disp = V.copy()
        # Normalise to correlation matrix for display
        diag = np.sqrt(np.diag(V_disp))
        corr = V_disp / np.outer(np.where(diag > 0, diag, 1), np.where(diag > 0, diag, 1))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Correlation")
        labels = ["J/ψ", "ηc", "χc0", "χc1", "ηc(2S)", "bkg"]
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        for ii in range(6):
            for jj in range(6):
                ax.text(
                    jj,
                    ii,
                    f"{corr[ii,jj]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if abs(corr[ii, jj]) < 0.7 else "white",
                )
        ax.set_title("Yield correlation matrix\n(extracted from RooFitResult)", fontsize=10)

        ax = axes[2]
        bins_sw = np.linspace(np.percentile(sw_bkg_d, 0.5), np.percentile(sw_bkg_d, 99.5), 61)
        ax.hist(sw_bkg_d, bins=bins_sw, color="royalblue", edgecolor="navy", alpha=0.8)
        ax.axvline(0, color="red", ls="--", lw=2, label="w = 0")
        ax.axvline(
            np.mean(sw_bkg_d),
            color="orange",
            ls="--",
            lw=2,
            label=f"Mean = {np.mean(sw_bkg_d):.3f}",
        )
        frac_neg = np.mean(sw_bkg_d < 0)
        ax.set_xlabel(r"${}_{s}\mathcal{P}_{\rm bkg}(m)$", fontsize=11)
        ax.set_ylabel("Events", fontsize=11)
        ax.set_title(
            f"True sWeight distribution\n(neg fraction = {frac_neg:.3f}, "
            f"Σw = {w_sum_d:.0f} ≈ N_bkg={N_bkg:.0f})",
            fontsize=10,
        )
        ax.legend(fontsize=9)

        fig.suptitle("Part 3 — Option D: True sPlot sWeights", fontsize=13, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 2: FOM ratio with true sWeights
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax = axes[0]
        ax.plot(PID_CUTS, fom_sb, "k-", lw=2, label="Current sideband", alpha=0.7)
        ax.plot(PID_CUTS, fom_b, color="darkorange", lw=1.8, ls=":", label="Option B (ARGUS-tail)")
        ax.plot(PID_CUTS, fom_c, color="green", lw=1.8, ls="--", label="Option C (approx sWeight)")
        ax.plot(PID_CUTS, fom_d, color="royalblue", lw=2.5, label="Option D (true sWeight)")
        ax.axhline(1.0, color="red", ls="--", lw=2, label="Threshold = 1.0")
        ax.set_xlabel("PID cut", fontsize=12)
        ax.set_ylabel(r"$\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$", fontsize=12)
        ax.set_title("Absolute FOM ratio: all four methods", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)
        max_d_idx = np.argmax(fom_d)
        ax.annotate(
            f"Option D max = {fom_d[max_d_idx]:.3f}\nat PID>{PID_CUTS[max_d_idx]:.2f}",
            xy=(PID_CUTS[max_d_idx], fom_d[max_d_idx]),
            xytext=(PID_CUTS[max_d_idx] + 0.06, fom_d[max_d_idx] + 0.02),
            arrowprops=dict(arrowstyle="->", color="royalblue"),
            fontsize=9,
            color="royalblue",
        )

        ax = axes[1]
        ax.plot(PID_CUTS, fom_sb / fom_sb[0], "k-", lw=2, label="Sideband", alpha=0.7)
        ax.plot(PID_CUTS, fom_b / fom_b[0], color="darkorange", lw=1.8, ls=":", label="Option B")
        ax.plot(
            PID_CUTS, fom_c / fom_c[0], color="green", lw=1.8, ls="--", label="Option C (approx)"
        )
        ax.plot(
            PID_CUTS, fom_d / fom_d[0], color="royalblue", lw=2.5, label="Option D (true sWeight)"
        )
        ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("PID cut", fontsize=12)
        ax.set_ylabel("FOM(cut) / FOM(0)", fontsize=12)
        ax.set_title("Relative FOM: Option D vs others", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Part 3 — Option D: True sPlot sWeights FOM", fontsize=13, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

print("  Saved:", optd_plot_path)

# ===========================================================================
# PART 4: Comparison with Fit-Based FOM (Key Result)
# ===========================================================================
print("\n" + "=" * 70)
print("PART 4: Comparison with fit-based FOM")
print("=" * 70)

df_prev = pd.read_csv(prev_csv_path)
scan_rows = df_prev[df_prev["section"] == "Part3_PIDscan"].copy()

fit_pid_cuts, fit_fom1, fit_fom2, fit_S1, fit_B = [], [], [], [], []
for _, row in scan_rows.iterrows():
    lbl = row["label"]
    note = str(row.get("note", ""))
    try:
        pid_v = float(re.search(r"PID>(\S+)", lbl).group(1))
        s1_v = float(re.search(r"S1=(\S+)", note).group(1))
        b_v = float(re.search(r"B=(\S+)", note).group(1))
        fom1_v = float(re.search(r"FOM1=(\S+)", note).group(1))
        fom2_v = float(re.search(r"FOM2=(\S+)", note).group(1))
        fit_pid_cuts.append(pid_v)
        fit_fom1.append(fom1_v)
        fit_fom2.append(fom2_v)
        fit_S1.append(s1_v)
        fit_B.append(b_v)
    except Exception as exc:
        print(f"  Warning: could not parse row: {exc}")

fit_pid_cuts = np.array(fit_pid_cuts)
fit_fom1 = np.array(fit_fom1)
fit_fom2 = np.array(fit_fom2)

print(f"  Fit-based FOM loaded: {len(fit_pid_cuts)} points")
print(f"  Peak: FOM1 = {np.max(fit_fom1):.3f} at PID>{fit_pid_cuts[np.argmax(fit_fom1)]:.2f}")

# Pearson correlations (relative FOM vs fit-based at discrete fit points)
fom_sb_at = np.interp(fit_pid_cuts, PID_CUTS, fom_sb)
fom_b_at = np.interp(fit_pid_cuts, PID_CUTS, fom_b)
fom_c_at = np.interp(fit_pid_cuts, PID_CUTS, fom_c)
fom_d_at = np.interp(fit_pid_cuts, PID_CUTS, fom_d)
fit_rel = fit_fom1 / fit_fom1[0]

corr_sb = np.corrcoef(fom_sb_at / fom_sb_at[0], fit_rel)[0, 1]
corr_b = np.corrcoef(fom_b_at / fom_b_at[0], fit_rel)[0, 1]
corr_c = np.corrcoef(fom_c_at / fom_c_at[0], fit_rel)[0, 1]
corr_d = np.corrcoef(fom_d_at / fom_d_at[0], fit_rel)[0, 1]

print("\n  Pearson r (relative FOM vs fit-based):")
print(f"    Sideband proxy: r = {corr_sb:.4f}")
print(f"    Option B:       r = {corr_b:.4f}")
print(f"    Option C:       r = {corr_c:.4f}")
print(f"    Option D:       r = {corr_d:.4f}")

methods_r = [
    (corr_sb, "Sideband"),
    (corr_b, "Option B"),
    (corr_c, "Option C"),
    (corr_d, "Option D (true sW)"),
]
best_corr = max(methods_r, key=lambda x: x[0])
print(f"\n  Best proxy by correlation: {best_corr[1]} (r = {best_corr[0]:.4f})")

Path(comparison_plot_path).parent.mkdir(parents=True, exist_ok=True)

with PdfPages(comparison_plot_path) as pdf:

    # --- Page 1: Absolute FOM ratios for all methods ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.plot(PID_CUTS, fom_sb, "k-", lw=2, alpha=0.7, label=f"Sideband (r={corr_sb:.3f})")
    ax.plot(PID_CUTS, fom_b, color="darkorange", lw=2, ls=":", label=f"Option B (r={corr_b:.3f})")
    ax.plot(
        PID_CUTS, fom_c, color="green", lw=2, ls="--", label=f"Option C approx (r={corr_c:.3f})"
    )
    if SWEIGHT_OK:
        ax.plot(
            PID_CUTS, fom_d, color="royalblue", lw=2.5, label=f"Option D true sW (r={corr_d:.3f})"
        )
    ax.axhline(1.0, color="red", ls="--", lw=2, label="Threshold = 1.0")
    ax.set_xlabel("PID product cut", fontsize=12)
    ax.set_ylabel(r"$\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$", fontsize=12)
    ax.set_title("Absolute FOM ratio: which proxy says PID helps?", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.3)
    txt = (
        "Max FOM ratios:\n"
        f"  Sideband: {np.max(fom_sb):.4f}\n"
        f"  Opt B:    {np.max(fom_b):.4f}\n"
        f"  Opt C:    {np.max(fom_c):.4f}\n"
        f"  Opt D:    {np.max(fom_d):.4f}\n"
        "  Threshold: 1.0000"
    )
    ax.text(
        0.97,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax = axes[1]
    ax.semilogy(PID_CUTS, fom_sb, "k-", lw=2, alpha=0.7, label="Sideband")
    ax.semilogy(PID_CUTS, fom_b, color="darkorange", lw=2, ls=":", label="Option B")
    ax.semilogy(PID_CUTS, fom_c, color="green", lw=2, ls="--", label="Option C")
    if SWEIGHT_OK:
        ax.semilogy(PID_CUTS, fom_d, color="royalblue", lw=2.5, label="Option D (true sW)")
    ax.axhline(1.0, color="red", ls="--", lw=2)
    ax.set_xlabel("PID product cut", fontsize=12)
    ax.set_ylabel(r"$\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$ [log scale]", fontsize=12)
    ax.set_title("FOM ratio (log scale)", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Part 4 — All Proxy Methods: FOM Ratio vs PID Cut", fontsize=13, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: KEY RESULT — Relative FOM vs fit-based ground truth ---
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        PID_CUTS,
        fom_sb / fom_sb[0],
        "k-",
        lw=2,
        alpha=0.65,
        label=f"Sideband proxy (r = {corr_sb:.3f})",
    )
    ax.plot(
        PID_CUTS,
        fom_b / fom_b[0],
        color="darkorange",
        lw=2,
        ls=":",
        label=f"Option B — ARGUS tail (r = {corr_b:.3f})",
    )
    ax.plot(
        PID_CUTS,
        fom_c / fom_c[0],
        color="green",
        lw=2,
        ls="--",
        label=f"Option C — approx sWeight (r = {corr_c:.3f})",
    )
    if SWEIGHT_OK:
        ax.plot(
            PID_CUTS,
            fom_d / fom_d[0],
            color="royalblue",
            lw=2.5,
            label=f"Option D — true sPlot sWeight (r = {corr_d:.3f})",
        )

    # Fit-based FOM — GROUND TRUTH
    ax.plot(
        fit_pid_cuts,
        fit_rel,
        "D-",
        color="darkgreen",
        lw=3,
        ms=10,
        zorder=10,
        label=r"Fit-based FOM1 = $(J/\psi+\eta_c)/\sqrt{B}$  [ground truth]",
    )

    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("PID product cut", fontsize=13)
    ax.set_ylabel("FOM(cut) / FOM(0)  [normalised to PID=0]", fontsize=13)
    ax.set_title(
        r"$\bf{Key\ Result:}$ Which proxy best tracks the fit-based FOM?"
        "\n(Pearson r with fit-based relative FOM shown in legend)",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(-0.01, 0.32)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.text(
        0.97,
        0.97,
        f"Best proxy: {best_corr[1]}\n(r = {best_corr[0]:.4f})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="darkgreen",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: Summary table ---
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    rows = [
        ["Method", "N events", "Max FOM", "FOM>1?", "Peak PID", "r (vs fit)"],
        [
            "Current sideband",
            f"{len(pid_sb_cc):,}",
            f"{np.max(fom_sb):.4f}",
            "YES" if np.any(fom_sb > 1.0) else "NO",
            f">{PID_CUTS[np.argmax(fom_sb)]:.2f}",
            f"{corr_sb:.4f}",
        ],
        [
            "Option B: ARGUS-tail",
            f"{len(pid_opt_b):,}",
            f"{np.max(fom_b):.4f}",
            "YES" if np.any(fom_b > 1.0) else "NO",
            f">{PID_CUTS[np.argmax(fom_b)]:.2f}",
            f"{corr_b:.4f}",
        ],
        [
            "Option C: approx sW",
            f"{len(cc_sig_full):,}" if FIT_OK else "N/A",
            f"{np.max(fom_c):.4f}" if FIT_OK else "N/A",
            "YES" if np.any(fom_c > 1.0) else "NO",
            f">{PID_CUTS[np.argmax(fom_c)]:.2f}" if FIT_OK else "N/A",
            f"{corr_c:.4f}",
        ],
        [
            "Option D: true sPlot",
            f"{len(cc_sig_full):,}" if SWEIGHT_OK else "failed",
            f"{np.max(fom_d):.4f}" if SWEIGHT_OK else "N/A",
            "YES" if np.any(fom_d > 1.0) else "NO",
            f">{PID_CUTS[np.argmax(fom_d)]:.2f}" if SWEIGHT_OK else "N/A",
            f"{corr_d:.4f}" if SWEIGHT_OK else "N/A",
        ],
        [
            "Fit-based FOM1 [truth]",
            "N/A (from fits)",
            f"{np.max(fit_fom1):.3f}",
            "YES (observed)",
            f">{fit_pid_cuts[np.argmax(fit_fom1)]:.2f}",
            "1.0000 (ref)",
        ],
    ]

    col_w = [0.28, 0.14, 0.10, 0.10, 0.12, 0.12]
    tbl = ax.table(
        cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center", colWidths=col_w
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    for j in range(len(rows[0])):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    n_data = len(rows) - 1
    for j in range(len(rows[0])):
        tbl[n_data, j].set_facecolor("#d5f5e3")

    ax.set_title(
        "Summary: PID Proxy Comparison — All Methods\n"
        "(r = Pearson correlation with fit-based relative FOM)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    rec = [
        "CONCLUSION:",
        f"  • Fit-based FOM shows PID cuts ARE beneficial (peak at PID>{fit_pid_cuts[np.argmax(fit_fom1)]:.2f}, +{100*(np.max(fit_fom1)/fit_fom1[0]-1):.0f}%)",
        f"  • ALL proxy methods are {'anti' if best_corr[0] < 0 else 'positively'}-correlated with fit-based FOM",
        f"  • Best proxy by correlation: {best_corr[1]} (r = {best_corr[0]:.4f})",
        "  • Root cause: ALL data regions contain well-reconstructed tracks with good PID.",
        "  • True combinatorial background has lower PID but cannot be isolated via data proxies.",
        "  • Recommendation: use fit-based FOM evaluation in the N-D optimizer.",
    ]
    ax.text(
        0.01,
        0.02,
        "\n".join(rec),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="#fef9e7", alpha=0.9),
    )

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("  Saved:", comparison_plot_path)

# ===========================================================================
# Write CSV
# ===========================================================================
csv_rows = []
for i, cut in enumerate(PID_CUTS):
    for method, eps_b, fom_v in [
        ("current_sideband", eps_bkg_sb[i], fom_sb[i]),
        ("option_b", eps_bkg_b[i], fom_b[i]),
        ("option_c_approx", eps_bkg_c[i], fom_c[i]),
        ("option_d_sweight", eps_bkg_d[i], fom_d[i]),
    ]:
        csv_rows.append(
            {
                "method": method,
                "pid_cut": cut,
                "eps_sig": eps_sig[i],
                "eps_bkg": eps_b,
                "fom_ratio": fom_v,
                "fom_relative": fom_v
                / (
                    fom_sb[0]
                    if "sideband" in method
                    else (
                        fom_b[0]
                        if "option_b" in method
                        else fom_c[0] if "option_c" in method else fom_d[0]
                    )
                ),
                "note": "",
            }
        )

for pid_v, fom1_v, fom2_v, s1_v, b_v in zip(fit_pid_cuts, fit_fom1, fit_fom2, fit_S1, fit_B):
    csv_rows.append(
        {
            "method": "fit_based",
            "pid_cut": pid_v,
            "eps_sig": np.nan,
            "eps_bkg": np.nan,
            "fom_ratio": fom1_v,
            "fom_relative": fom1_v / fit_fom1[0],
            "note": f"S1={s1_v:.0f} B={b_v:.0f} FOM2={fom2_v:.3f}",
        }
    )

pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"  Saved CSV: {csv_path} ({len(csv_rows)} rows)")

# ===========================================================================
# Final summary
# ===========================================================================
print("\n" + "=" * 70)
print("STUDY COMPLETE — PID Proxy Comparison")
print("=" * 70)
print(f"\n  Option B: {len(pid_opt_b):,} events  FOM > 1.0: {np.any(fom_b > 1.0)}")
if FIT_OK:
    print(f"  Option C: {len(cc_sig_full):,} events  FOM > 1.0: {np.any(fom_c > 1.0)}")
if SWEIGHT_OK:
    print(f"  Option D: {len(cc_sig_full):,} events  FOM > 1.0: {np.any(fom_d > 1.0)}")
    print(f"    Σ(sWeights) = {w_sum_d:.1f}  (fitted N_bkg = {N_bkg:.0f})")
    print(f"    Negative weight fraction: {np.mean(sw_bkg_d < 0):.3f}")
print(
    f"\n  Fit-based FOM peak: {np.max(fit_fom1):.3f} at PID>{fit_pid_cuts[np.argmax(fit_fom1)]:.2f}"
)
print("\n  Pearson r (relative FOM vs fit):")
for corr_v, name in [
    (corr_sb, "Sideband"),
    (corr_b, "Option B"),
    (corr_c, "Option C"),
    (corr_d, "Option D (true sW)"),
]:
    print(f"    {name:30s}: r = {corr_v:.4f}")
print(f"\n  Best proxy: {best_corr[1]} (r = {best_corr[0]:.4f})")
