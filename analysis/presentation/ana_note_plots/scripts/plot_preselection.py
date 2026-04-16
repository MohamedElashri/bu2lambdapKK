"""
Preselection validation plots for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces:
  figs/LambdaLL/L0_M_after_preselection.pdf       — Λ mass (full range), selection window marked
  figs/LambdaDD/L0_M_after_preselection.pdf
  figs/LambdaLL/Preselections/Delta_Z.pdf         — B+→Λ vertex separation
  figs/LambdaDD/Preselections/Delta_Z.pdf
  figs/LambdaLL/Preselections/Bu_M_after_preselection.pdf — B+ corrected mass, signal window
  figs/LambdaDD/Preselections/Bu_M_after_preselection.pdf
  figs/LambdaLL/logIPCHI2run2_Sig.pdf             — log(B+ IP χ²) after final selection
  figs/LambdaDD/logIPCHI2run2_Sig.pdf

Run from analysis/ directory:
    uv run python presentation/ana_note_plots/scripts/plot_preselection.py
"""

import logging
import sys
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPTS_DIR.resolve().parents[3]  # analysis/
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.resolve().parents[2]))  # analysis/ for modules.*

from modules.plot_utils import BINNING, COLORS, figs_path, plot_data, save_fig, setup_style
from modules.presentation_config import MC15_PID_BRANCHES, get_presentation_config

PRESENTATION = get_presentation_config()
DATA_BASE = PRESENTATION.data_base
MC_BASE = PRESENTATION.mc_base
M_LAMBDA_PDG = PRESENTATION.lambda_mass_pdg
YEARS = PRESENTATION.year_suffixes
MAGNETS = PRESENTATION.magnets
LAMBDA_MIN = PRESENTATION.lambda_mass_min
LAMBDA_MAX = PRESENTATION.lambda_mass_max
PID_CUT = PRESENTATION.pid_product_min
SB_LO, SB_HI = PRESENTATION.bu_sideband_windows()
SIG_LO, SIG_HI = PRESENTATION.bu_signal_window()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════════


def _open_tree(path: Path, cat: str):
    return uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]


def _trigger_mask(ev, is_mc: bool) -> ak.Array:
    if is_mc:
        l0_keys = ["Bu_L0Global_TIS", "Bu_L0HadronDecision_TIS"]
    else:
        l0_keys = ["Bu_L0GlobalDecision_TIS", "Bu_L0PhysDecision_TIS", "Bu_L0HadronDecision_TIS"]
    hlt1_keys = ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"]
    hlt2_keys = [
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    fields = ev.fields

    def _or(keys):
        m = ak.zeros_like(ev["L0_MM"], dtype=bool)
        for k in keys:
            if k in fields:
                m = m | (ev[k] > 0)
        return m

    ml0 = _or(l0_keys)
    mhlt1 = _or(hlt1_keys)
    mhlt2 = _or(hlt2_keys)
    if not any(k in fields for k in l0_keys):
        ml0 = ak.ones_like(ml0)
    if not any(k in fields for k in hlt1_keys):
        mhlt1 = ak.ones_like(mhlt1)
    if not any(k in fields for k in hlt2_keys):
        mhlt2 = ak.ones_like(mhlt2)
    return ml0 & mhlt1 & mhlt2


def _pid_key(branch: str, is_mc: bool) -> str:
    del is_mc
    if branch == "Lp":
        return MC15_PID_BRANCHES["lp"]
    return MC15_PID_BRANCHES["p"]


def _load_events(
    path: Path,
    cat: str,
    is_mc: bool,
    after_lambda_window: bool = True,
    sideband_only: bool = True,
    signal_window: bool = False,
    pid_cut: float = 0.0,
) -> ak.Array:
    """Generic loader: trigger + optional Lambda mass window + optional B+ sideband/signal."""
    lp_pid_key = _pid_key("Lp", is_mc)
    pid_branches = [MC15_PID_BRANCHES["p"], MC15_PID_BRANCHES["h1"], MC15_PID_BRANCHES["h2"]]
    want = [
        "L0_MM",
        "Bu_MM",
        "L0_ENDVERTEX_Z",
        "Bu_ENDVERTEX_Z",
        "Bu_FDCHI2_OWNPV",
        "Bu_IPCHI2_OWNPV",
        "Bu_PT",
        "L0_FDCHI2_OWNPV",
        "Bu_P",
        "nTracks",
        lp_pid_key,
    ] + pid_branches
    if is_mc:
        want += [
            "Bu_L0Global_TIS",
            "Bu_L0HadronDecision_TIS",
            "Bu_Hlt1TrackMVADecision_TOS",
            "Bu_Hlt1TwoTrackMVADecision_TOS",
            "Bu_Hlt2Topo2BodyDecision_TOS",
            "Bu_Hlt2Topo3BodyDecision_TOS",
            "Bu_Hlt2Topo4BodyDecision_TOS",
        ]
    else:
        want += [
            "Bu_L0GlobalDecision_TIS",
            "Bu_L0PhysDecision_TIS",
            "Bu_L0HadronDecision_TIS",
            "Bu_Hlt1TrackMVADecision_TOS",
            "Bu_Hlt1TwoTrackMVADecision_TOS",
            "Bu_Hlt2Topo2BodyDecision_TOS",
            "Bu_Hlt2Topo3BodyDecision_TOS",
            "Bu_Hlt2Topo4BodyDecision_TOS",
        ]

    # DTF chi2 — key name differs between data and MC files
    tree = _open_tree(path, cat)
    avail = [b for b in want if b in tree.keys()]
    dtf_key = None
    for k in ["Bu_DTF_chi2", "Bu_DTF_CHI2", "Bu_DTFFun_DTF_CHI2"]:
        if k in tree.keys():
            dtf_key = k
            if k not in avail:
                avail.append(k)
            break

    ev = tree.arrays(avail, library="ak")
    ev["Lp_ProbNNp"] = ev[lp_pid_key]
    ev["Delta_Z_mm"] = np.abs(ev["L0_ENDVERTEX_Z"] - ev["Bu_ENDVERTEX_Z"])
    ev["Bu_MM_corrected"] = ev["Bu_MM"] - ev["L0_MM"] + M_LAMBDA_PDG
    # Compute log(IPCHI2) and eta from P/PT
    ipchi2 = ev["Bu_IPCHI2_OWNPV"]
    ev["log_Bu_IPCHI2"] = ak.where(ipchi2 > 0, np.log(ipchi2), -999.0)
    P = ev["Bu_P"]
    PT = ev["Bu_PT"]
    pz = ak.where(P > PT, np.sqrt(P**2 - PT**2), 0.0)
    ev["Bu_eta"] = ak.where(
        (P + pz) > 0, 0.5 * np.log((P + pz) / ak.where(P - pz > 0, P - pz, 1e-9)), 0.0
    )
    if dtf_key:
        chi2 = ev[dtf_key]
        if "var" in str(ak.type(chi2)):
            chi2 = chi2[:, 0]
        ev["Bu_DTF_chi2"] = chi2

    # Trigger
    ev = ev[_trigger_mask(ev, is_mc)]

    # Lambda mass window (optional)
    if after_lambda_window:
        ev = ev[(ev["L0_MM"] > LAMBDA_MIN) & (ev["L0_MM"] < LAMBDA_MAX)]

    # PID product cut (optional)
    if pid_cut > 0:
        p_b = MC15_PID_BRANCHES["p"]
        h1_b = MC15_PID_BRANCHES["h1"]
        h2_b = MC15_PID_BRANCHES["h2"]
        pid = ak.ones_like(ev["Bu_PT"])
        for b in [p_b, h1_b, h2_b]:
            if b in ev.fields:
                pid = pid * ev[b]
        ev = ev[pid > pid_cut]

    # Data: B+ mass sidebands or signal window; MC: full range
    if sideband_only and not is_mc:
        mass = ak.to_numpy(ev["Bu_MM_corrected"])
        ev = ev[
            ((mass >= SB_LO[0]) & (mass <= SB_LO[1])) | ((mass >= SB_HI[0]) & (mass <= SB_HI[1]))
        ]
    elif signal_window and not is_mc:
        mass = ak.to_numpy(ev["Bu_MM_corrected"])
        ev = ev[(mass >= SIG_LO) & (mass <= SIG_HI)]
    return ev


def _collect(loader_fn, is_mc: bool, cat: str, state: str = "jpsi", **kw) -> ak.Array:
    arrs = []
    if is_mc:
        dir_name = "Jpsi" if state == "jpsi" else state
        prefix = dir_name
        for yr in YEARS:
            for mag in MAGNETS:
                p = MC_BASE / dir_name / f"{prefix}_{yr}_{mag}.root"
                if not p.exists():
                    continue
                try:
                    arrs.append(loader_fn(p, cat, is_mc=True, **kw))
                except Exception as e:
                    log.warning(f"  MC {p}: {e}")
    else:
        for yr in YEARS:
            for mag in MAGNETS:
                p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
                if not p.exists():
                    continue
                try:
                    arrs.append(loader_fn(p, cat, is_mc=False, **kw))
                except Exception as e:
                    log.warning(f"  Data {p}: {e}")
    return ak.concatenate(arrs) if arrs else ak.Array([])


# ════════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ════════════════════════════════════════════════════════════════════════════════


def _scale_mc_to_data(h_data: np.ndarray, h_mc: np.ndarray) -> float:
    """Return scale factor so MC integral matches data integral."""
    n_d = np.sum(h_data)
    n_m = np.sum(h_mc)
    return float(n_d / n_m) if n_m > 0 else 1.0


def single_plot(
    cat: str,
    branch: str,
    data_arr,
    mc_arr,
    xlabel: str,
    bins: np.ndarray,
    cut_val=None,
    cut_dir: str = "right",
    filename: str = None,
    log_y: bool = False,
    subdir: str = "Preselections",
):
    """
    Produce a single-variable comparison PDF matching the reference style.

    data_arr : numpy array of data values (sideband)
    mc_arr   : numpy array of MC values
    """
    if filename is None:
        filename = f"{branch}.pdf"

    bmin, bmax = bins[0], bins[-1]
    width = round((bmax - bmin) / len(bins), 2)

    fig, ax = plt.subplots()
    histstyle = {"range": (bmin, bmax), "bins": len(bins) - 1}

    # Data — error bars
    if len(data_arr) > 0:
        plot_data(ax, data_arr, "Sideband data", histstyle, color="black")

    # MC — step dashed, density=True then scale
    if len(mc_arr) > 0:
        h_mc, _ = np.histogram(mc_arr, range=(bmin, bmax), bins=len(bins) - 1)
        h_d, _ = (
            np.histogram(data_arr, range=(bmin, bmax), bins=len(bins) - 1)
            if len(data_arr) > 0
            else (np.zeros_like(h_mc), None)
        )
        scale = _scale_mc_to_data(h_d, h_mc)
        ax.hist(
            mc_arr,
            range=(bmin, bmax),
            bins=len(bins) - 1,
            histtype="step",
            linestyle="--",
            linewidth=4,
            weights=np.full(len(mc_arr), scale),
            label=r"$J/\psi$ MC",
            color=COLORS[0],
        )

    # Cut line
    if cut_val is not None:
        ax.axvline(cut_val, color=COLORS[3], linestyle="--", linewidth=2)
        # Efficiency on MC — placed at lower right to avoid legend collision
        if len(mc_arr) > 0:
            eff = np.mean(mc_arr > cut_val) if cut_dir == "right" else np.mean(mc_arr < cut_val)
            ax.text(
                0.97,
                0.04,
                rf"$\varepsilon_{{\rm MC}} = {100*eff:.1f}\%$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color=COLORS[3],
                fontsize=10,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(rf"Candidates / ({width} {_unit(branch)})")
    # Legend: upper left avoids colliding with the efficiency text (lower right)
    ax.legend(loc="upper left", frameon=False)
    if log_y:
        ax.set_yscale("log")

    out = figs_path(cat, subdir, filename)
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


def _unit(branch: str) -> str:
    if "PT" in branch:
        return r"MeV/$c$"
    if "MM" in branch or "M_" in branch:
        return r"MeV/$c^2$"
    if "mm" in branch.lower():
        return "mm"
    return ""


# ════════════════════════════════════════════════════════════════════════════════
# PRESELECTION VARIABLE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════


# (branch_key, xlabel, bins_array, cut_value, cut_direction, log_y, filename)
def _bins(key: str) -> np.ndarray:
    """Return a linspace bin array from the centralized BINNING dict."""
    b = BINNING[key]
    return np.linspace(b["range"][0], b["range"][1], b["bins"] + 1)


PRESEL_VARS = {
    "LL": [
        ("Delta_Z_mm", r"$\Delta Z$ [mm]", _bins("delta_z_ll"), 20.0, "right", True, "Delta_Z.pdf"),
    ],
    "DD": [
        ("Delta_Z_mm", r"$\Delta Z$ [mm]", _bins("delta_z_dd"), 5.0, "right", True, "Delta_Z.pdf"),
    ],
}


# ════════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════


def plot_presel_vars(cat: str, data_ev, mc_ev):
    """6 individual preselection variable PDFs."""
    log.info(f"  Preselection variables [{cat}]")
    for branch, xlabel, bins, cut_val, cut_dir, log_y, fname in PRESEL_VARS[cat]:
        if branch not in (data_ev.fields if len(data_ev) > 0 else []):
            log.warning(f"    Branch {branch} missing — skipping")
            continue
        d_arr = (
            np.clip(ak.to_numpy(data_ev[branch]).astype(float), bins[0], bins[-1])
            if len(data_ev) > 0
            else np.array([])
        )
        m_arr = (
            np.clip(ak.to_numpy(mc_ev[branch]).astype(float), bins[0], bins[-1])
            if len(mc_ev) > 0
            else np.array([])
        )
        single_plot(
            cat,
            branch,
            d_arr,
            m_arr,
            xlabel,
            bins,
            cut_val=cut_val,
            cut_dir=cut_dir,
            filename=fname,
            log_y=log_y,
        )


def plot_bu_mass_sideband(cat: str, data_ev):
    """Bu_M_after_preselection.pdf — B+ corrected mass sideband data after preselection."""
    log.info(f"  Bu_M after presel [{cat}]")
    if len(data_ev) == 0:
        log.warning(f"  No data for Bu_M [{cat}]")
        return
    bins = np.linspace(5100, 5600, 51)  # 10 MeV/bin
    bmin, bmax, nbins = bins[0], bins[-1], len(bins) - 1
    width = round((bmax - bmin) / nbins, 0)

    mass = np.asarray(ak.to_numpy(data_ev["Bu_MM_corrected"]))

    fig, ax = plt.subplots()
    plot_data(ax, mass, "Sideband data", {"range": (bmin, bmax), "bins": nbins}, color="black")

    # Mark signal window
    ax.axvspan(SIG_LO, SIG_HI, alpha=0.15, color=COLORS[2], label="Signal window")
    ax.axvline(SIG_LO, color=COLORS[2], linestyle="--", linewidth=1.5)
    ax.axvline(SIG_HI, color=COLORS[2], linestyle="--", linewidth=1.5)

    ax.set_xlabel(r"$m_{\rm corr}(B^+)$ [MeV/$c^2$]")
    ax.set_ylabel(rf"Candidates / ({int(width)} MeV/$c^2$)")
    ax.legend(frameon=False)
    ax.set_xlim(bmin, bmax)

    save_fig(fig, figs_path(cat, "Preselections", "Bu_M_after_preselection.pdf"))


def plot_lambda_mass_after_presel_toplevel(cat: str, data_ev, mc_ev):
    """
    L0_M_after_preselection.pdf — top-level file.
    Lambda mass after trigger (before Lambda mass window), showing where the
    current configured selection window is applied.
    """
    log.info(f"  L0_M_after_preselection (top-level) [{cat}]")
    bins = _bins("lambda_mass_full")  # 1 MeV/bin, full stripping range
    bmin, bmax, nbins = bins[0], bins[-1], len(bins) - 1
    width = round((bmax - bmin) / nbins, 2)

    d_arr = np.asarray(ak.to_numpy(data_ev["L0_MM"])) if len(data_ev) > 0 else np.array([])
    m_arr = np.asarray(ak.to_numpy(mc_ev["L0_MM"])) if len(mc_ev) > 0 else np.array([])

    fig, ax = plt.subplots()
    if len(d_arr) > 0:
        plot_data(
            ax, d_arr, "Data (B+ sidebands)", {"range": (bmin, bmax), "bins": nbins}, color="black"
        )
    if len(m_arr) > 0:
        h_d, _ = (
            np.histogram(d_arr, range=(bmin, bmax), bins=nbins)
            if len(d_arr) > 0
            else (np.ones(nbins), None)
        )
        h_m, _ = np.histogram(m_arr, range=(bmin, bmax), bins=nbins)
        scale = _scale_mc_to_data(h_d, h_m)
        ax.hist(
            m_arr,
            range=(bmin, bmax),
            bins=nbins,
            histtype="step",
            linestyle="--",
            linewidth=4,
            weights=np.full(len(m_arr), scale),
            label=r"$J/\psi$ MC (arb. scale)",
            color=COLORS[0],
        )
    # Mark the preselection mass window
    ax.axvline(LAMBDA_MIN, color=COLORS[3], linestyle="--", linewidth=1.5)
    ax.axvline(LAMBDA_MAX, color=COLORS[3], linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"$m(p\pi^-)$ [MeV/$c^2$]")
    ax.set_ylabel(rf"Candidates / ({width} MeV/$c^2$)")
    ax.legend(frameon=False)
    ax.set_xlim(bmin, bmax)

    save_fig(fig, figs_path(cat, "L0_M_after_preselection.pdf"))
    log.info(f"    Saved L0_M_after_preselection.pdf [{cat}]")


def plot_log_ipchi2(cat: str, data_ev, mc_ev):
    """
    logIPCHI2run2_Sig.pdf — log(B+ IP χ²) distribution after final selection.
    Reference: shown in §4.6 optimisation section.
    """
    log.info(f"  logIPCHI2run2_Sig [{cat}]")
    bins = _bins("log_ipchi2")  # 0.2/bin
    bmin, bmax, nbins = bins[0], bins[-1], len(bins) - 1
    width = round((bmax - bmin) / nbins, 2)

    d_arr = np.asarray(ak.to_numpy(data_ev["log_Bu_IPCHI2"])) if len(data_ev) > 0 else np.array([])
    m_arr = np.asarray(ak.to_numpy(mc_ev["log_Bu_IPCHI2"])) if len(mc_ev) > 0 else np.array([])
    # Clip to range
    d_arr = d_arr[(d_arr > bmin) & (d_arr < bmax)] if len(d_arr) > 0 else d_arr
    m_arr = m_arr[(m_arr > bmin) & (m_arr < bmax)] if len(m_arr) > 0 else m_arr

    fig, ax = plt.subplots()
    if len(d_arr) > 0:
        plot_data(
            ax, d_arr, "Signal window data", {"range": (bmin, bmax), "bins": nbins}, color="black"
        )
    if len(m_arr) > 0:
        h_d, _ = (
            np.histogram(d_arr, range=(bmin, bmax), bins=nbins)
            if len(d_arr) > 0
            else (np.ones(nbins), None)
        )
        h_m, _ = np.histogram(m_arr, range=(bmin, bmax), bins=nbins)
        scale = _scale_mc_to_data(h_d, h_m)
        ax.hist(
            m_arr,
            range=(bmin, bmax),
            bins=nbins,
            histtype="step",
            linestyle="--",
            linewidth=4,
            weights=np.full(len(m_arr), scale),
            label=r"$J/\psi$ MC (arb. scale)",
            color=COLORS[0],
        )
    ax.set_xlabel(r"$\log(B^+~\chi^2_{\rm IP})$")
    ax.set_ylabel(rf"Candidates / ({width})")
    ax.legend(frameon=False)

    save_fig(fig, figs_path(cat, "logIPCHI2run2_Sig.pdf"))
    log.info(f"    Saved logIPCHI2run2_Sig.pdf [{cat}]")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════


def main():
    for cat in ["LL", "DD"]:
        log.info(f"=== Category: {cat} ===")

        # ── Load 1: trigger only (no Lambda window) — for L0_M_after_preselection ──
        log.info("  Loading for Lambda mass plot (trigger only)...")
        data_pre = _collect(
            _load_events, is_mc=False, cat=cat, after_lambda_window=False, sideband_only=True
        )
        mc_pre = _collect(
            _load_events, is_mc=True, cat=cat, after_lambda_window=False, sideband_only=False
        )
        plot_lambda_mass_after_presel_toplevel(cat, data_pre, mc_pre)
        del data_pre, mc_pre

        # ── Load 2: trigger + Lambda window — for Delta_Z + B+ mass ──────────
        log.info("  Loading for presel vars + B+ mass...")
        data_ev = _collect(
            _load_events, is_mc=False, cat=cat, after_lambda_window=True, sideband_only=True
        )
        mc_ev = _collect(
            _load_events, is_mc=True, cat=cat, after_lambda_window=True, sideband_only=False
        )
        plot_presel_vars(cat, data_ev, mc_ev)  # Delta_Z only
        plot_bu_mass_sideband(cat, data_ev)
        del data_ev, mc_ev

        # ── Load 3: full selection (signal window + configured PID pre-cut) — for logIPCHI2 ──
        log.info("  Loading for logIPCHI2 (final selection)...")
        data_final = _collect(
            _load_events,
            is_mc=False,
            cat=cat,
            after_lambda_window=True,
            sideband_only=False,
            signal_window=True,
            pid_cut=PID_CUT,
        )
        mc_final = _collect(
            _load_events,
            is_mc=True,
            cat=cat,
            after_lambda_window=True,
            sideband_only=False,
            signal_window=False,
            pid_cut=PID_CUT,
        )
        plot_log_ipchi2(cat, data_final, mc_final)
        del data_final, mc_final

    log.info("=== Done. Figures in: " + str(SCRIPTS_DIR.parent / "figs") + " ===")


if __name__ == "__main__":
    main()
