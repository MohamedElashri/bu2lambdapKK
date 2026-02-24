"""
Standalone study: Sideband Validity & Fit-Based Cut Comparison

═══════════════════════════════════════════════════════════════════════════
PART 1 — Is the B+ sideband a valid PID proxy?
═══════════════════════════════════════════════════════════════════════════
The N-D optimizer uses B+ mass sidebands (with charmonium M(Λ̄pK⁻) filter)
as background proxy.  Sideband events are real, well-identified particles;
their PID may be systematically HIGHER than true combinatorial background in
the signal region.  We compare PID distributions across five data regions:

  1. Current proxy: sidebands ∩ charmonium region      [5150,5230]∪[5330,5410] MeV, M(cc) ∈ [2900,3800]
  2. Sideband (no cc filter): sidebands, all M(cc)
  3. Upper sideband only: [5330,5410] MeV, charmonium filter ON
  4. Lower sideband only: [5150,5230] MeV, charmonium filter ON
  5. No-charmonium region: signal window [5255,5305], M(cc) > 4000 MeV

═══════════════════════════════════════════════════════════════════════════
PART 2 — FOM ratio with alternative background proxies
═══════════════════════════════════════════════════════════════════════════
Re-compute ε_sig/√ε_bkg for each proxy over a dense PID scan [0, 0.5].
If ALL proxies keep the ratio below 1.0 → PID is genuinely non-discriminating.
If any proxy shows ratio > 1.0 → the current sideband proxy is misleading.

═══════════════════════════════════════════════════════════════════════════
PART 3 — Fit-based FOM: 1D PID scan with actual mass fits (7 points)
═══════════════════════════════════════════════════════════════════════════
Fix base cuts (Set 1: chi2<30, FDCHI2>100, IPCHI2<6.5, PT>3000) and scan
PID_product ∈ [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30].
For each point: apply cuts → M(Λ̄pK⁻) fit → extract fitted yields → compute
  FOM1(fit) = (N_jpsi + N_etac) / √N_bkg
  FOM2(fit) = (N_chic0 + N_chic1) / (√(N_chic0+N_chic1) + √N_bkg)
Compare fitted FOM vs proxy-based FOM to test whether the optimizer is correct.

═══════════════════════════════════════════════════════════════════════════
PART 4 — Cut set comparison via actual fits (4 scenarios)
═══════════════════════════════════════════════════════════════════════════
Fit four scenarios and compare fitted yields, S/B, and significance:
  Set1_noPID : chi2<30, FDCHI2>100, IPCHI2<6.5, PT>3000, PID>0.0
  Set2_noPID : chi2<20, FDCHI2>100, IPCHI2<5.5, PT>3000, PID>0.0
  Set1_PID10 : chi2<30, FDCHI2>100, IPCHI2<6.5, PT>3000, PID>0.10
  Set2_PID10 : chi2<20, FDCHI2>100, IPCHI2<5.5, PT>3000, PID>0.10
"""

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
validity_plot_path = snakemake.output.validity_plot  # noqa: F821
pid_scan_plot_path = snakemake.output.pid_scan_plot  # noqa: F821
cutset_plot_path = snakemake.output.cutset_plot  # noqa: F821
csv_path = snakemake.output.csv  # noqa: F821

plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

# ---------------------------------------------------------------------------
# Load Step 2 cache
# ---------------------------------------------------------------------------
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

# Combine LL + DD per year
data_combined = {}
for year in data_dict:
    arrays = [
        data_dict[year][tt] for tt in data_dict[year] if hasattr(data_dict[year][tt], "layout")
    ]
    if arrays:
        data_combined[year] = ak.concatenate(arrays, axis=0)

all_data = ak.concatenate(list(data_combined.values()), axis=0)
print(f"  Total data events (after Lambda cuts): {len(all_data):,}")

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

STATE_LABELS = {
    "jpsi": r"$J/\psi$",
    "etac": r"$\eta_c$",
    "chic0": r"$\chi_{c0}$",
    "chic1": r"$\chi_{c1}$",
}
STATE_COLORS = {"jpsi": "#1f77b4", "etac": "#d62728", "chic0": "#2ca02c", "chic1": "#ff7f0e"}

# Combined MC for PID analysis
pid_mc_all = np.concatenate(
    [
        np.asarray(
            ak.drop_none(
                ak.firsts(mc_combined[s]["PID_product"])
                if "var" in str(ak.type(mc_combined[s]["PID_product"]))
                else mc_combined[s]["PID_product"]
            )
        )
        for s in mc_combined
    ]
)

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
NO_CC_MIN = opt.get("no_charmonium_mass_min", 4000.0)

mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"


def flat(events, branch):
    """Return branch as flat numpy array."""
    arr = events[branch]
    if "var" in str(ak.type(arr)):
        arr = ak.firsts(arr)
    return np.asarray(ak.drop_none(arr))


# ---------------------------------------------------------------------------
# Helper: apply a dict of cuts to data_combined → returns {year: ak.Array}
# ---------------------------------------------------------------------------
def apply_selection(data_by_year, cuts):
    """
    cuts: list of (branch_name, operator, value)
          operator ∈ {"less", "greater"}
    Returns {year: filtered_array}
    """
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
# Define data regions for Part 1
# ---------------------------------------------------------------------------
bu_all = flat(all_data, bu_branch)
cc_all = flat(all_data, mass_branch)

# 1. Current proxy: sideband ∩ charmonium filter
mask_sb_cc = (
    (((bu_all > SB_LO_MIN) & (bu_all < SB_LO_MAX)) | ((bu_all > SB_HI_MIN) & (bu_all < SB_HI_MAX)))
    & (cc_all > CC_MIN)
    & (cc_all < CC_MAX)
)

# 2. Sideband, no charmonium filter
mask_sb_nocc = ((bu_all > SB_LO_MIN) & (bu_all < SB_LO_MAX)) | (
    (bu_all > SB_HI_MIN) & (bu_all < SB_HI_MAX)
)

# 3. Upper sideband only, charmonium filter
mask_upper_sb = (
    ((bu_all > SB_HI_MIN) & (bu_all < SB_HI_MAX)) & (cc_all > CC_MIN) & (cc_all < CC_MAX)
)

# 4. Lower sideband only, charmonium filter
mask_lower_sb = (
    ((bu_all > SB_LO_MIN) & (bu_all < SB_LO_MAX)) & (cc_all > CC_MIN) & (cc_all < CC_MAX)
)

# 5. No-charmonium region (signal B+ window, M(cc) > 4 GeV)
mask_nocc_sig = (bu_all > SIG_MIN) & (bu_all < SIG_MAX) & (cc_all > NO_CC_MIN)

REGION_DEFS = [
    ("Current proxy\n(SB ∩ cc filter)", mask_sb_cc, "black"),
    ("SB, no cc filter", mask_sb_nocc, "royalblue"),
    (f"Upper SB [{SB_HI_MIN:.0f}–{SB_HI_MAX:.0f}]", mask_upper_sb, "darkorange"),
    (f"Lower SB [{SB_LO_MIN:.0f}–{SB_LO_MAX:.0f}]", mask_lower_sb, "green"),
    (r"No-cc ($M(cc)>4$ GeV)", mask_nocc_sig, "crimson"),
]

for lbl, m, _ in REGION_DEFS:
    print(f"  {lbl.replace(chr(10), ' ')}: {int(np.sum(m)):,} events")

summary_rows = []

# ===========================================================================
# PART 1: PID distributions across data regions
# ===========================================================================
print("\n" + "=" * 70)
print("PART 1: PID distributions across data regions")
print("=" * 70)

PID_VARS = [
    ("PID_product", r"$p \times K^+ \times K^-$ PID product"),
    ("p_ProbNNp", r"$p_{\rm ProbNNp}$ (bachelor proton)"),
    ("h1_ProbNNk", r"$K^+$ ProbNNk"),
    ("h2_ProbNNk", r"$K^-$ ProbNNk"),
]

Path(validity_plot_path).parent.mkdir(parents=True, exist_ok=True)

with PdfPages(validity_plot_path) as pdf:

    # --- Page 1: PID product across regions ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    bins = np.linspace(0, 1, 51)
    bc = 0.5 * (bins[:-1] + bins[1:])

    ax = axes[0]
    for lbl, mask, color in REGION_DEFS:
        if np.sum(mask) < 5:
            continue
        arr = flat(all_data[mask], "PID_product")
        h, _ = np.histogram(arr, bins=bins, density=True)
        ax.step(
            bc,
            h,
            where="mid",
            lw=2,
            color=color,
            label=f"{lbl.replace(chr(10), ' ')} (N={int(np.sum(mask)):,})",
            alpha=0.85,
        )

    h_mc, _ = np.histogram(pid_mc_all, bins=bins, density=True)
    ax.step(
        bc,
        h_mc,
        where="mid",
        lw=2.5,
        color="purple",
        ls="--",
        label=f"Signal MC (N={len(pid_mc_all):,})",
    )
    ax.set_xlabel("PID product", fontsize=12)
    ax.set_ylabel("Normalised", fontsize=12)
    ax.set_title("PID product: all data regions vs signal MC", fontsize=12)
    ax.legend(fontsize=8, loc="upper center")
    ax.set_xlim(0, 1)

    ax = axes[1]
    for lbl, mask, color in REGION_DEFS:
        if np.sum(mask) < 5:
            continue
        arr = flat(all_data[mask], "PID_product")
        h, _ = np.histogram(arr, bins=bins, density=True)
        ax.step(bc, h, where="mid", lw=2, color=color, alpha=0.85, label=lbl.replace(chr(10), " "))
    ax.set_xlabel("PID product", fontsize=12)
    ax.set_ylabel("Normalised (log scale)", fontsize=12)
    ax.set_title("PID product — log scale", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="lower center")
    ax.set_xlim(0, 1)

    fig.suptitle(
        "Part 1: Is the sideband PID distribution a valid background proxy?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Individual PID variables (2×2 grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for idx, (branch, label) in enumerate(PID_VARS):
        ax = axes_flat[idx]
        bins_v = np.linspace(0, 1, 51)
        bc_v = 0.5 * (bins_v[:-1] + bins_v[1:])

        for lbl, mask, color in REGION_DEFS:
            if np.sum(mask) < 5:
                continue
            arr = flat(all_data[mask], branch)
            h, _ = np.histogram(arr, bins=bins_v, density=True)
            ax.step(
                bc_v,
                h,
                where="mid",
                lw=1.8,
                color=color,
                alpha=0.85,
                label=lbl.replace(chr(10), " "),
            )

        if branch in mc_combined.get("jpsi", ak.Array([])).fields if mc_combined else []:
            pass  # skip MC overlay on individual vars for clarity
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Normalised", fontsize=11)
        ax.set_title(label, fontsize=11)
        if idx == 0:
            ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle(
        "Individual PID variables: comparison across data regions", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: KS test matrix (PID product) ---
    region_names = [lbl.replace("\n", " ") for lbl, _, _ in REGION_DEFS]
    pid_arrs = [flat(all_data[mask], "PID_product") for _, mask, _ in REGION_DEFS]

    n = len(REGION_DEFS)
    ks_pval = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if len(pid_arrs[i]) > 0 and len(pid_arrs[j]) > 0:
                _, p = stats.ks_2samp(pid_arrs[i], pid_arrs[j])
                ks_pval[i, j] = ks_pval[j, i] = p

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log10(ks_pval + 1e-300), vmin=-10, vmax=0, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}$(KS p-value)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(region_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(region_names, fontsize=9)
    for i in range(n):
        for j in range(n):
            pv = ks_pval[i, j]
            txt = f"{pv:.2e}" if pv < 0.01 else f"{pv:.3f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=8,
                color="black" if pv > 0.001 else "white",
            )
    ax.set_title("KS test p-values: PID product (green=similar, red=different)", fontsize=12)
    fig.suptitle(
        "Low p-value → distributions differ → sideband is biased proxy", fontsize=11, style="italic"
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    for i, (lbl, _, _) in enumerate(REGION_DEFS):
        arr = pid_arrs[i]
        summary_rows.append(
            {
                "section": "Part1_RegionStats",
                "label": lbl.replace("\n", " "),
                "n_events": len(arr),
                "pid_mean": float(np.mean(arr)) if len(arr) else np.nan,
                "pid_median": float(np.median(arr)) if len(arr) else np.nan,
                "pid_std": float(np.std(arr)) if len(arr) else np.nan,
                "note": "",
            }
        )

print("  Part 1 plots saved.")

# ===========================================================================
# PART 2: FOM ratio with alternative background proxies
# ===========================================================================
print("\n" + "=" * 70)
print("PART 2: FOM ratio with alternative background proxies")
print("=" * 70)

cut_dense = np.linspace(0.001, 0.5, 300)

proxy_defs = [
    ("Current (SB ∩ cc)", mask_sb_cc, "black"),
    ("SB, no cc filter", mask_sb_nocc, "royalblue"),
    ("Upper SB only (cc)", mask_upper_sb, "darkorange"),
    ("Lower SB only (cc)", mask_lower_sb, "green"),
    (r"No-cc ($M>4$ GeV)", mask_nocc_sig, "crimson"),
]

with PdfPages(validity_plot_path) as _:
    pass  # already opened above; appending continues via re-open below

# Re-open pdf in append mode isn't straightforward with PdfPages.
# Instead, we'll collect all pages in the same open block.
# We need to redo this - collect Part2 and Part3 pages in the SAME PdfPages context.
# The Part 1 pdf block is already closed. We need to reopen it.
# Solution: collect all matplotlib figures, then write them all at once.

# We'll collect figs to write in a combined pass at the end.
# Let me restructure to write all pages for validity_plot_path together.

# Re-collect: generate Part 2 figures here, write them after the re-open
part2_figs = []

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
for lbl, mask, color in proxy_defs:
    if np.sum(mask) < 5:
        continue
    pid_bkg = flat(all_data[mask], "PID_product")
    surv_sig = np.array([np.mean(pid_mc_all > c) for c in cut_dense])
    surv_bkg = np.array([np.mean(pid_bkg > c) for c in cut_dense])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(surv_bkg > 0, surv_sig / np.sqrt(surv_bkg), np.nan)
    ax.plot(cut_dense, ratio, lw=2, color=color, label=lbl, alpha=0.85)

ax.axhline(1.0, color="green", lw=2.5, ls="--", label="Baseline = 1 (no cut)")
ax.set_xlabel("PID product cut threshold", fontsize=12)
ax.set_ylabel(r"$\varepsilon_{\rm sig}\ /\ \sqrt{\varepsilon_{\rm bkg}}$", fontsize=12)
ax.set_title(r"FOM ratio vs PID cut: all background proxies", fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 0.5)
ax.set_ylim(0.5, 1.3)
ax.grid(True, alpha=0.3)
ax.axhspan(0.5, 1.0, alpha=0.05, color="red")
ax.text(0.55, 0.15, "FOM loss\nregion", transform=ax.transAxes, fontsize=9, color="red", alpha=0.7)

# Right: survival curves overlaid
ax = axes[1]
for lbl, mask, color in proxy_defs:
    if np.sum(mask) < 5:
        continue
    pid_bkg = flat(all_data[mask], "PID_product")
    surv_bkg = np.array([np.mean(pid_bkg > c) for c in cut_dense])
    ax.plot(cut_dense, surv_bkg, lw=2, color=color, label=lbl, alpha=0.85)

surv_sig = np.array([np.mean(pid_mc_all > c) for c in cut_dense])
ax.plot(cut_dense, surv_sig, lw=2.5, color="purple", ls="--", label="Signal MC")
ax.set_xlabel("PID product cut threshold", fontsize=12)
ax.set_ylabel("Fraction passing cut", fontsize=12)
ax.set_title("Survival fractions: signal MC vs background proxies", fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

fig.suptitle(
    r"Part 2: FOM ratio $\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$ across proxy choices"
    "\n(if all proxies stay below 1.0 → PID genuinely non-discriminating)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
part2_figs.append(fig)

# Print summary table
print(
    f"\n  {'Proxy':<35}  {'ratio@0.05':>10}  {'ratio@0.10':>10}  {'max_ratio':>10}  {'all<1?':>6}"
)
for lbl, mask, _ in proxy_defs:
    if np.sum(mask) < 5:
        continue
    pid_bkg = flat(all_data[mask], "PID_product")
    ratios = np.array(
        [(np.mean(pid_mc_all > c) / np.sqrt(max(np.mean(pid_bkg > c), 1e-9))) for c in cut_dense]
    )
    r05 = float(np.mean(pid_mc_all > 0.05) / np.sqrt(max(np.mean(pid_bkg > 0.05), 1e-9)))
    r10 = float(np.mean(pid_mc_all > 0.10) / np.sqrt(max(np.mean(pid_bkg > 0.10), 1e-9)))
    rmax = float(np.nanmax(ratios))
    all_lt1 = bool(np.all(ratios[np.isfinite(ratios)] <= 1.0))
    print(
        f"  {lbl.replace(chr(10),' '):<35}  {r05:>10.4f}  {r10:>10.4f}  {rmax:>10.4f}  {'YES' if all_lt1 else 'NO!':>6}"
    )
    summary_rows.append(
        {
            "section": "Part2_FOMratio",
            "label": lbl.replace("\n", " "),
            "n_events": int(np.sum(mask)),
            "pid_mean": np.nan,
            "pid_median": np.nan,
            "pid_std": np.nan,
            "note": f"ratio@0.05={r05:.4f} ratio@0.10={r10:.4f} max={rmax:.4f} all<1={'YES' if all_lt1 else 'NO'}",
        }
    )

plt.close("all")

# Append Part 2 figures to the PDF
with PdfPages(validity_plot_path) as pdf:
    # Re-generate Part 1 plots so we can write all in one block
    # Part 1: PID product comparison
    bins = np.linspace(0, 1, 51)
    bc = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    for lbl, mask, color in REGION_DEFS:
        if np.sum(mask) < 5:
            continue
        arr = flat(all_data[mask], "PID_product")
        h, _ = np.histogram(arr, bins=bins, density=True)
        ax.step(
            bc,
            h,
            where="mid",
            lw=2,
            color=color,
            label=f"{lbl.replace(chr(10),' ')} (N={int(np.sum(mask)):,})",
            alpha=0.85,
        )
    h_mc, _ = np.histogram(pid_mc_all, bins=bins, density=True)
    ax.step(
        bc,
        h_mc,
        where="mid",
        lw=2.5,
        color="purple",
        ls="--",
        label=f"Signal MC (N={len(pid_mc_all):,})",
    )
    ax.set_xlabel("PID product", fontsize=12)
    ax.set_ylabel("Normalised", fontsize=12)
    ax.set_title("PID product: all data regions vs signal MC", fontsize=12)
    ax.legend(fontsize=8, loc="upper center")
    ax.set_xlim(0, 1)
    ax2 = axes[1]
    for lbl, mask, color in REGION_DEFS:
        if np.sum(mask) < 5:
            continue
        arr = flat(all_data[mask], "PID_product")
        h, _ = np.histogram(arr, bins=bins, density=True)
        ax2.step(bc, h, where="mid", lw=2, color=color, alpha=0.85, label=lbl.replace(chr(10), " "))
    ax2.set_xlabel("PID product", fontsize=12)
    ax2.set_ylabel("Normalised (log scale)", fontsize=12)
    ax2.set_title("PID product — log scale", fontsize=12)
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1)
    fig.suptitle(
        "Part 1: Is the sideband PID a valid background proxy?", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # Part 1: Individual PID variables
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    for idx, (branch, label) in enumerate(PID_VARS):
        ax = axes_flat[idx]
        bins_v = np.linspace(0, 1, 51)
        bc_v = 0.5 * (bins_v[:-1] + bins_v[1:])
        for lbl, mask, color in REGION_DEFS:
            if np.sum(mask) < 5:
                continue
            arr = flat(all_data[mask], branch)
            h, _ = np.histogram(arr, bins=bins_v, density=True)
            ax.step(
                bc_v,
                h,
                where="mid",
                lw=1.8,
                color=color,
                alpha=0.85,
                label=lbl.replace(chr(10), " "),
            )
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Normalised", fontsize=11)
        ax.set_title(label, fontsize=11)
        if idx == 0:
            ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    fig.suptitle(
        "Part 1: Individual PID variables across data regions", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # Part 1: KS test matrix
    pid_arrs = [flat(all_data[mask], "PID_product") for _, mask, _ in REGION_DEFS]
    region_names = [lbl.replace("\n", " ") for lbl, _, _ in REGION_DEFS]
    n = len(REGION_DEFS)
    ks_pval = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if len(pid_arrs[i]) > 0 and len(pid_arrs[j]) > 0:
                _, p = stats.ks_2samp(pid_arrs[i], pid_arrs[j])
                ks_pval[i, j] = ks_pval[j, i] = p
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log10(ks_pval + 1e-300), vmin=-10, vmax=0, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}$(KS p-value)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(region_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(region_names, fontsize=9)
    for i in range(n):
        for j in range(n):
            pv = ks_pval[i, j]
            txt = f"{pv:.2e}" if pv < 0.01 else f"{pv:.3f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=8,
                color="black" if pv > 0.001 else "white",
            )
    ax.set_title(
        "KS p-values: PID product across regions\n(green=similar, red=significantly different)",
        fontsize=11,
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # Part 2: FOM ratio figure
    for f2 in part2_figs:
        pdf.savefig(f2)
        plt.close(f2)

print("  Part 1 & 2 plots saved to:", validity_plot_path)

# ===========================================================================
# PART 3: Fit-based FOM — 1D PID scan (7 fits)
# ===========================================================================
print("\n" + "=" * 70)
print("PART 3: Fit-based FOM: 1D PID scan")
print("=" * 70)

# Base cuts (Set 1 from split_charmonium_opt)
BASE_CUTS = [
    ("Bu_DTF_chi2", "less", 30.0),
    ("Bu_FDCHI2_OWNPV", "greater", 100.0),
    ("Bu_IPCHI2_OWNPV", "less", 6.5),
    ("Bu_PT", "greater", 3000.0),
]

PID_SCAN_CUTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Also prepare proxy-based FOM for comparison
pid_sb_arr = flat(all_data[mask_sb_cc], "PID_product")
pid_mc_all_arr = pid_mc_all

# Set plots dir for fitter
config.paths["output"]["plots_dir"] = str(Path(output_dir) / "fits")
Path(output_dir, "fits").mkdir(parents=True, exist_ok=True)
fitter = MassFitter(config)

pid_scan_results = []

for pid_cut in PID_SCAN_CUTS:
    cuts = BASE_CUTS + [("PID_product", "greater", pid_cut)]
    data_cut = apply_selection(data_combined, cuts)
    n_total = sum(len(v) for v in data_cut.values())
    print(f"\n  PID > {pid_cut:.2f}: {n_total:,} events before B+ mass cut")

    try:
        res = fitter.perform_fit(data_cut, fit_combined=True)
        yld = res["yields"].get("combined", {})

        n_jpsi = yld.get("jpsi", (0, 0))
        n_etac = yld.get("etac", (0, 0))
        n_chic0 = yld.get("chic0", (0, 0))
        n_chic1 = yld.get("chic1", (0, 0))
        n_bkg = yld.get("background", (0, 0))

        S1 = max(n_jpsi[0] + n_etac[0], 0)
        S2 = max(n_chic0[0] + n_chic1[0], 0)
        B = max(n_bkg[0], 1e-3)
        eS1 = np.sqrt(n_jpsi[1] ** 2 + n_etac[1] ** 2)
        eS2 = np.sqrt(n_chic0[1] ** 2 + n_chic1[1] ** 2)
        eB = n_bkg[1]

        fom1 = S1 / np.sqrt(B)
        fom2 = S2 / (np.sqrt(max(S2, 0)) + np.sqrt(B))
        # Errors via propagation (rough)
        efom1 = fom1 * np.sqrt((eS1 / max(S1, 1)) ** 2 + (0.5 * eB / B) ** 2)
        efom2 = 0.0  # skip error on fom2 for simplicity

        # Proxy-based FOM for comparison
        e_sig_pid = float(np.mean(pid_mc_all_arr > pid_cut))
        e_bkg_pid = float(np.mean(pid_sb_arr > pid_cut))
        proxy_ratio = e_sig_pid / np.sqrt(max(e_bkg_pid, 1e-9))

        print(f"    S1(J/ψ+ηc) = {S1:.0f} ± {eS1:.0f}")
        print(f"    S2(χc0+χc1) = {S2:.0f} ± {eS2:.0f}")
        print(f"    B = {B:.0f} ± {eB:.0f}")
        print(f"    FOM1(fit) = {fom1:.3f}   FOM2(fit) = {fom2:.3f}")
        print(f"    Proxy ratio = {proxy_ratio:.4f}")

        pid_scan_results.append(
            {
                "pid_cut": pid_cut,
                "n_events": n_total,
                "S1": S1,
                "eS1": eS1,
                "S2": S2,
                "eS2": eS2,
                "B": B,
                "eB": eB,
                "fom1_fit": fom1,
                "efom1": efom1,
                "fom2_fit": fom2,
                "proxy_ratio": proxy_ratio,
                "n_jpsi": n_jpsi[0],
                "n_etac": n_etac[0],
                "n_chic0": n_chic0[0],
                "n_chic1": n_chic1[0],
            }
        )

        summary_rows.append(
            {
                "section": "Part3_PIDscan",
                "label": f"PID>{pid_cut:.2f}",
                "n_events": n_total,
                "pid_mean": np.nan,
                "pid_median": np.nan,
                "pid_std": np.nan,
                "note": f"S1={S1:.0f} S2={S2:.0f} B={B:.0f} FOM1={fom1:.3f} FOM2={fom2:.3f}",
            }
        )

    except Exception as exc:
        print(f"    !! Fit failed: {exc}")
        pid_scan_results.append(
            {
                "pid_cut": pid_cut,
                "n_events": n_total,
                "S1": np.nan,
                "eS1": np.nan,
                "S2": np.nan,
                "eS2": np.nan,
                "B": np.nan,
                "eB": np.nan,
                "fom1_fit": np.nan,
                "efom1": np.nan,
                "fom2_fit": np.nan,
                "proxy_ratio": np.nan,
                "n_jpsi": np.nan,
                "n_etac": np.nan,
                "n_chic0": np.nan,
                "n_chic1": np.nan,
            }
        )

df_scan = pd.DataFrame(pid_scan_results)

Path(pid_scan_plot_path).parent.mkdir(parents=True, exist_ok=True)

with PdfPages(pid_scan_plot_path) as pdf:

    # --- Page 1: FOM1 and FOM2 from fit vs PID cut ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    valid = df_scan["fom1_fit"].notna()
    cuts_v = df_scan.loc[valid, "pid_cut"].values
    fom1_v = df_scan.loc[valid, "fom1_fit"].values
    efom1_v = df_scan.loc[valid, "efom1"].values
    fom2_v = df_scan.loc[valid, "fom2_fit"].values
    proxy_v = df_scan.loc[valid, "proxy_ratio"].values

    ax = axes[0]
    ax.errorbar(
        cuts_v,
        fom1_v / fom1_v[0],
        yerr=efom1_v / fom1_v[0],
        fmt="o-",
        color="navy",
        lw=2.5,
        ms=8,
        capsize=4,
        label=r"FOM1 (fit) = $(J/\psi + \eta_c) / \sqrt{B}$",
    )
    ax.plot(
        cuts_v,
        fom2_v / fom2_v[0],
        "s-",
        color="darkred",
        lw=2,
        ms=7,
        label=r"FOM2 (fit) = $(\chi_{c0}+\chi_{c1}) / (\sqrt{S}+\sqrt{B})$",
    )
    ax.plot(
        cuts_v,
        proxy_v / proxy_v[0],
        "^--",
        color="gray",
        lw=1.5,
        ms=6,
        label=r"Proxy: $\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$ (current)",
    )
    ax.axhline(1.0, color="green", lw=2, ls="--", alpha=0.8, label="Baseline (no PID cut)")
    ax.set_xlabel("PID product cut threshold", fontsize=13)
    ax.set_ylabel("FOM / FOM(no cut)", fontsize=13)
    ax.set_title("Fit-based FOM vs PID cut\n(normalised to no-cut baseline)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 0.35)
    ax.grid(True, alpha=0.3)
    ax.fill_between(
        cuts_v,
        (fom1_v / fom1_v[0]) - (efom1_v / fom1_v[0]),
        (fom1_v / fom1_v[0]) + (efom1_v / fom1_v[0]),
        alpha=0.15,
        color="navy",
    )

    # Right: absolute fitted yields vs PID cut
    ax = axes[1]
    states_to_plot = [
        ("n_jpsi", STATE_LABELS["jpsi"], STATE_COLORS["jpsi"]),
        ("n_etac", STATE_LABELS["etac"], STATE_COLORS["etac"]),
        ("n_chic0", STATE_LABELS["chic0"], STATE_COLORS["chic0"]),
        ("n_chic1", STATE_LABELS["chic1"], STATE_COLORS["chic1"]),
    ]
    for col, lbl, color in states_to_plot:
        vals = df_scan.loc[valid, col].values
        ax.plot(cuts_v, vals, "o-", lw=2, color=color, label=lbl, ms=6)
    ax.set_xlabel("PID product cut threshold", fontsize=13)
    ax.set_ylabel("Fitted yield (combined)", fontsize=13)
    ax.set_title("Per-state fitted yields vs PID cut", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 0.35)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Part 3: Fit-based FOM — does PID_product actually improve the fit?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: FOM comparison vs proxy method ---
    fig, ax = plt.subplots(figsize=(12, 7))
    if valid.sum() > 0:
        ax.plot(
            cuts_v,
            fom1_v / fom1_v[0],
            "o-",
            color="navy",
            lw=2.5,
            ms=8,
            label=r"FOM1 from FIT: $(J/\psi+\eta_c)/\sqrt{B}$",
        )
        ax.plot(
            cuts_v,
            fom2_v / fom2_v[0],
            "s-",
            color="darkred",
            lw=2,
            ms=7,
            label=r"FOM2 from FIT: $(\chi_{c0}+\chi_{c1})/(\sqrt{S}+\sqrt{B})$",
        )
        ax.plot(
            cuts_v,
            proxy_v / proxy_v[0],
            "^--",
            color="gray",
            lw=2,
            ms=7,
            label=r"PROXY method: $\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$",
        )
    ax.axhline(1.0, color="green", lw=2, ls="--", label="No cut = 1.0")
    ax.set_xlabel("PID product cut", fontsize=13)
    ax.set_ylabel("Normalised FOM", fontsize=13)
    ax.set_title(
        "Fit-based FOM vs proxy-based FOM\n"
        "(same shape → proxy method is reliable; different → proxy is misleading)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.35)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("  Part 3 plots saved to:", pid_scan_plot_path)

# ===========================================================================
# PART 4: Cut set comparison via fits (4 scenarios)
# ===========================================================================
print("\n" + "=" * 70)
print("PART 4: Cut set comparison via actual fits")
print("=" * 70)

CUT_SETS = {
    "Set1_noPID": [
        ("Bu_DTF_chi2", "less", 30.0),
        ("Bu_FDCHI2_OWNPV", "greater", 100.0),
        ("Bu_IPCHI2_OWNPV", "less", 6.5),
        ("Bu_PT", "greater", 3000.0),
        ("PID_product", "greater", 0.0),
    ],
    "Set2_noPID": [
        ("Bu_DTF_chi2", "less", 20.0),
        ("Bu_FDCHI2_OWNPV", "greater", 100.0),
        ("Bu_IPCHI2_OWNPV", "less", 5.5),
        ("Bu_PT", "greater", 3000.0),
        ("PID_product", "greater", 0.0),
    ],
    "Set1_PID10": [
        ("Bu_DTF_chi2", "less", 30.0),
        ("Bu_FDCHI2_OWNPV", "greater", 100.0),
        ("Bu_IPCHI2_OWNPV", "less", 6.5),
        ("Bu_PT", "greater", 3000.0),
        ("PID_product", "greater", 0.10),
    ],
    "Set2_PID10": [
        ("Bu_DTF_chi2", "less", 20.0),
        ("Bu_FDCHI2_OWNPV", "greater", 100.0),
        ("Bu_IPCHI2_OWNPV", "less", 5.5),
        ("Bu_PT", "greater", 3000.0),
        ("PID_product", "greater", 0.10),
    ],
}

CUT_COLORS = {
    "Set1_noPID": "#1f77b4",
    "Set2_noPID": "#ff7f0e",
    "Set1_PID10": "#2ca02c",
    "Set2_PID10": "#d62728",
}

cs_results = {}
for set_name, cuts in CUT_SETS.items():
    data_cut = apply_selection(data_combined, cuts)
    n_total = sum(len(v) for v in data_cut.values())
    print(f"\n  {set_name}: {n_total:,} events before B+ mass cut")

    try:
        res = fitter.perform_fit(data_cut, fit_combined=True)
        yld = res["yields"].get("combined", {})

        row = {"n_total": n_total}
        for state in ["jpsi", "etac", "chic0", "chic1", "background"]:
            v, e = yld.get(state, (0, 0))
            row[f"N_{state}"] = max(v, 0)
            row[f"e_{state}"] = e

        B = max(row["N_background"], 1e-3)
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            S = row[f"N_{state}"]
            row[f"SoverB_{state}"] = S / B if B > 0 else np.nan
            row[f"sig_{state}"] = S / np.sqrt(S + B) if (S + B) > 0 else 0.0

        cs_results[set_name] = row
        print(
            f"    N_jpsi={row['N_jpsi']:.0f} N_etac={row['N_etac']:.0f} "
            f"N_chic0={row['N_chic0']:.0f} N_chic1={row['N_chic1']:.0f} "
            f"N_bkg={row['N_background']:.0f}"
        )

        summary_rows.append(
            {
                "section": "Part4_CutSets",
                "label": set_name,
                "n_events": n_total,
                "pid_mean": np.nan,
                "pid_median": np.nan,
                "pid_std": np.nan,
                "note": (
                    f"jpsi={row['N_jpsi']:.0f} etac={row['N_etac']:.0f} "
                    f"chic0={row['N_chic0']:.0f} chic1={row['N_chic1']:.0f} "
                    f"bkg={row['N_background']:.0f}"
                ),
            }
        )

    except Exception as exc:
        print(f"    !! Fit failed: {exc}")
        cs_results[set_name] = None

Path(cutset_plot_path).parent.mkdir(parents=True, exist_ok=True)

with PdfPages(cutset_plot_path) as pdf:

    set_names = list(cs_results.keys())
    valid_sets = [s for s in set_names if cs_results[s] is not None]
    xs = np.arange(len(valid_sets))
    bar_w = 0.18

    # --- Page 1: Fitted yields per state per cut set ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for i, state in enumerate(["jpsi", "etac", "chic0", "chic1"]):
        yields = [cs_results[s][f"N_{state}"] for s in valid_sets]
        errors = [cs_results[s][f"e_{state}"] for s in valid_sets]
        ax.bar(
            xs + i * bar_w - 1.5 * bar_w,
            yields,
            bar_w,
            yerr=errors,
            capsize=3,
            label=STATE_LABELS[state],
            color=STATE_COLORS[state],
            alpha=0.85,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(valid_sets, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Fitted signal yield", fontsize=12)
    ax.set_title("Fitted signal yields per state", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    bkg_yields = [cs_results[s]["N_background"] for s in valid_sets]
    bkg_errs = [cs_results[s]["e_background"] for s in valid_sets]
    ax.bar(
        xs,
        bkg_yields,
        0.5,
        yerr=bkg_errs,
        capsize=4,
        color=[CUT_COLORS[s] for s in valid_sets],
        alpha=0.85,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(valid_sets, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Fitted background yield", fontsize=12)
    ax.set_title("Background yield comparison", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Part 4: Fitted yields — 4 cut set scenarios", fontsize=13, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: S/B and significance ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for i, state in enumerate(["jpsi", "etac", "chic0", "chic1"]):
        sb = [cs_results[s][f"SoverB_{state}"] for s in valid_sets]
        ax.bar(
            xs + i * bar_w - 1.5 * bar_w,
            sb,
            bar_w,
            label=STATE_LABELS[state],
            color=STATE_COLORS[state],
            alpha=0.85,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(valid_sets, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("S/B ratio", fontsize=12)
    ax.set_title("Signal-to-background ratio per state", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    for i, state in enumerate(["jpsi", "etac", "chic0", "chic1"]):
        sig = [cs_results[s][f"sig_{state}"] for s in valid_sets]
        ax.bar(
            xs + i * bar_w - 1.5 * bar_w,
            sig,
            bar_w,
            label=STATE_LABELS[state],
            color=STATE_COLORS[state],
            alpha=0.85,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(valid_sets, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(r"Significance $S/\sqrt{S+B}$", fontsize=12)
    ax.set_title("Statistical significance per state", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Part 4: S/B and significance — does PID >0.10 improve fit quality?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: Summary table as matplotlib table ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    states_to_show = ["jpsi", "etac", "chic0", "chic1"]
    col_headers = (
        ["Cut set", "N_bkg"]
        + [f"N_{s}" for s in states_to_show]
        + [f"S/B_{s}" for s in states_to_show]
    )
    table_data = []
    for s in valid_sets:
        r = cs_results[s]
        row_data = [s, f"{r['N_background']:.0f}"]
        for st in states_to_show:
            row_data.append(f"{r[f'N_{st}']:.0f}±{r[f'e_{st}']:.0f}")
        for st in states_to_show:
            row_data.append(f"{r[f'SoverB_{st}']:.3f}")
        table_data.append(row_data)
    tbl = ax.table(cellText=table_data, colLabels=col_headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    ax.set_title("Summary: fitted yields and S/B for all 4 cut sets", fontsize=12, pad=20)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("  Part 4 plots saved to:", cutset_plot_path)

# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------
Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(csv_path, index=False)
print(f"\n  Summary CSV saved to: {csv_path}")

# ---------------------------------------------------------------------------
# Final print
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

if pid_scan_results and not np.isnan(pid_scan_results[0]["fom1_fit"]):
    fom1_at0 = pid_scan_results[0]["fom1_fit"]
    fom1_at10 = next(
        (r["fom1_fit"] for r in pid_scan_results if abs(r["pid_cut"] - 0.10) < 0.01), np.nan
    )
    direction = "DECREASES" if fom1_at10 < fom1_at0 else "INCREASES"
    print(f"\n  FOM1 at PID>0.00: {fom1_at0:.2f}")
    print(f"  FOM1 at PID>0.10: {fom1_at10:.2f}  → FOM1 {direction} with PID cut")
    if direction == "DECREASES":
        print("  → Fit confirms: PID_product > 0.0 (no cut) is correct optimum")
    else:
        print("  → Fit DISAGREES with proxy method: PID cuts DO help!")
        print("     The sideband proxy was misleading!")

if "Set1_noPID" in cs_results and "Set1_PID10" in cs_results:
    r0 = cs_results["Set1_noPID"]
    r1 = cs_results["Set1_PID10"]
    if r0 and r1:
        delta_jpsi = (r1["N_jpsi"] - r0["N_jpsi"]) / max(r0["N_jpsi"], 1) * 100
        delta_bkg = (r1["N_background"] - r0["N_background"]) / max(r0["N_background"], 1) * 100
        print(
            f"\n  Adding PID>0.10 to Set1: ΔN(J/ψ) = {delta_jpsi:+.1f}%,  ΔN(bkg) = {delta_bkg:+.1f}%"
        )
