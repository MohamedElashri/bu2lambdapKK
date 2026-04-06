"""
Kinematic reweighting comparison plots for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces (Batch 1):
  figs/LambdaLL/reweight/pt_cmp.pdf               — pT before reweighting
  figs/LambdaDD/reweight/pt_cmp.pdf
  figs/LambdaLL/reweight/eta_cmp.pdf              — η before reweighting
  figs/LambdaDD/reweight/eta_cmp.pdf
  figs/LambdaLL/reweight/IPCHI2_cmp.pdf           — log(IPCHI2) before reweighting
  figs/LambdaDD/reweight/IPCHI2_cmp.pdf
  figs/LambdaLL/reweight/ntracks_cmp.pdf          — nTracks before reweighting
  figs/LambdaDD/reweight/ntracks_cmp.pdf
  figs/LambdaLL/reweight/reweight_Bu_PT_cmp.pdf   — pT after reweighting
  figs/LambdaDD/reweight/reweight_Bu_PT_cmp.pdf
  figs/LambdaLL/reweight/reweight_Bu_IPCHI2_OWNPV_cmp.pdf
  figs/LambdaDD/reweight/reweight_Bu_IPCHI2_OWNPV_cmp.pdf
  figs/LambdaLL/reweight/reweight_nTracks_cmp.pdf
  figs/LambdaDD/reweight/reweight_nTracks_cmp.pdf
  figs/LambdaLL/reweight/sweight_Bu_PT_cmp.pdf    — J/ψ vs ηc sWeighted
  figs/LambdaDD/reweight/sweight_Bu_PT_cmp.pdf
  figs/LambdaLL/reweight/sweight_Bu_IPCHI2_OWNPV_cmp.pdf
  figs/LambdaDD/reweight/sweight_Bu_IPCHI2_OWNPV_cmp.pdf
  figs/LambdaLL/reweight/sweight_nTracks_cmp.pdf
  figs/LambdaDD/reweight/sweight_nTracks_cmp.pdf

Note: our reweighting is 1D pT-only; reference uses pT × log(IPCHI2) × nTracks.

Follows reference reweight_plot.py style exactly:
  - Three step histograms: unweighted MC, reweighted MC, sideband-subtracted data
  - density=True (normalized)
  - ScalarFormatter on y-axis
  - Colors: COLORS[0] (darkgreen), COLORS[1] (#6F4F59), COLORS[2] (#003366)

Run from analysis/ directory:
    uv run python studies/kinematic_reweighting/plot_reweighting.py
"""

import json
import logging
import sys
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

# ── paths ──────────────────────────────────────────────────────────────────────
STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_DIR.parents[1]))  # analysis/ for modules.*

from modules.plot_utils import COLORS, figs_path, make_formatter, save_fig, setup_style

DATA_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/data")
MC_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")
WEIGHT_DIR = STUDY_DIR / "output"

M_LAMBDA_PDG = 1115.683
YEARS = ["16", "17", "18"]
MAGNETS = ["MD", "MU"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()
_fmt = make_formatter()


# ════════════════════════════════════════════════════════════════════════════════
# WEIGHT LOOKUP
# ════════════════════════════════════════════════════════════════════════════════


def _load_weights(cat: str) -> tuple:
    """Load 1D pT weight map. Returns (pt_edges, weights)."""
    wfile = WEIGHT_DIR / f"kinematic_weights_{cat}.json"
    if not wfile.exists():
        # Fallback: no-op weights
        return np.array([0, 1e9]), np.array([1.0])
    with open(wfile) as f:
        w = json.load(f)
    edges = np.array(w["pt_bins"])
    weights = np.array(w["weights"])
    return edges, weights


def _apply_pt_weights(pt: np.ndarray, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return per-event weights from 1D pT lookup."""
    idx = np.digitize(pt, edges) - 1
    idx = np.clip(idx, 0, len(weights) - 1)
    return weights[idx]


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════════


def _trig_mask_ak(ev, l0_keys, hlt1_keys, hlt2_keys, ref_branch):
    """Build trigger mask from awkward array event dict."""
    avail = ev.fields
    ml0 = ak.zeros_like(ev[ref_branch], dtype=bool)
    for k in l0_keys:
        if k in avail:
            ml0 = ml0 | (ev[k] > 0)
    if not any(k in avail for k in l0_keys):
        ml0 = ak.ones_like(ml0)

    mhlt1 = ak.zeros_like(ev[ref_branch], dtype=bool)
    for k in hlt1_keys:
        if k in avail:
            mhlt1 = mhlt1 | (ev[k] > 0)
    if not any(k in avail for k in hlt1_keys):
        mhlt1 = ak.ones_like(mhlt1)

    mhlt2 = ak.zeros_like(ev[ref_branch], dtype=bool)
    for k in hlt2_keys:
        if k in avail:
            mhlt2 = mhlt2 | (ev[k] > 0)
    if not any(k in avail for k in hlt2_keys):
        mhlt2 = ak.ones_like(mhlt2)
    return ml0 & mhlt1 & mhlt2


def _load_data_vars(path: Path, cat: str) -> dict:
    """
    Load data kinematic variables with sideband assignment.
    Returns dict: branch → {"sig": array, "sb": array, "scale": float}
    """
    want = [
        "Bu_MM",
        "L0_MM",
        "Bu_PT",
        "Bu_P",
        "Bu_IPCHI2_OWNPV",
        "nTracks",
        "Bu_L0GlobalDecision_TIS",
        "Bu_L0PhysDecision_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="ak")

    mask = _trig_mask_ak(
        ev,
        ["Bu_L0GlobalDecision_TIS", "Bu_L0PhysDecision_TIS", "Bu_L0HadronDecision_TIS"],
        ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"],
        [
            "Bu_Hlt2Topo2BodyDecision_TOS",
            "Bu_Hlt2Topo3BodyDecision_TOS",
            "Bu_Hlt2Topo4BodyDecision_TOS",
        ],
        "Bu_PT",
    )
    mask = mask & (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)
    ev = ev[mask]

    bu_corr = ak.to_numpy(ev["Bu_MM"]) - ak.to_numpy(ev["L0_MM"]) + M_LAMBDA_PDG
    sig_mask = (bu_corr >= 5255) & (bu_corr <= 5305)
    sb_mask = ((bu_corr >= 5150) & (bu_corr <= 5230)) | ((bu_corr >= 5330) & (bu_corr <= 5410))
    scale = (5305 - 5255) / ((5230 - 5150) + (5410 - 5330))

    # Compute derived variables
    P = ak.to_numpy(ev["Bu_P"])
    PT = ak.to_numpy(ev["Bu_PT"])
    pz = np.sqrt(np.maximum(P**2 - PT**2, 0.0))
    eta = 0.5 * np.log(np.maximum((P + pz) / np.maximum(P - pz, 1e-9), 1e-9))
    log_ipchi2 = np.log(np.maximum(ak.to_numpy(ev["Bu_IPCHI2_OWNPV"]), 1e-9))
    pt_arr = ak.to_numpy(ev["Bu_PT"])
    nt_arr = ak.to_numpy(ev["nTracks"]).astype(float) if "nTracks" in avail else None

    result = {}
    for name, arr in [
        ("Bu_PT", pt_arr),
        ("Bu_eta", eta),
        ("log_IPCHI2", log_ipchi2),
        ("nTracks", nt_arr),
    ]:
        if arr is None:
            continue
        result[name] = {"sig": arr[sig_mask], "sb": arr[sb_mask], "scale": scale}
    return result


def _load_mc_vars(path: Path, cat: str) -> dict:
    """Load MC kinematic variables after trigger + Lambda cuts."""
    want = [
        "Bu_PT",
        "Bu_P",
        "Bu_IPCHI2_OWNPV",
        "nTracks",
        "L0_MM",
        "Bu_L0Global_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="ak")

    mask = _trig_mask_ak(
        ev,
        ["Bu_L0Global_TIS", "Bu_L0HadronDecision_TIS"],
        ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"],
        [
            "Bu_Hlt2Topo2BodyDecision_TOS",
            "Bu_Hlt2Topo3BodyDecision_TOS",
            "Bu_Hlt2Topo4BodyDecision_TOS",
        ],
        "Bu_PT",
    )
    mask = mask & (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)
    ev = ev[mask]

    P = ak.to_numpy(ev["Bu_PT"])
    Ptot = ak.to_numpy(ev["Bu_P"])
    pz = np.sqrt(np.maximum(Ptot**2 - P**2, 0.0))
    eta = 0.5 * np.log(np.maximum((Ptot + pz) / np.maximum(Ptot - pz, 1e-9), 1e-9))
    log_ipchi2 = np.log(np.maximum(ak.to_numpy(ev["Bu_IPCHI2_OWNPV"]), 1e-9))
    nt_arr = ak.to_numpy(ev["nTracks"]).astype(float) if "nTracks" in avail else None

    result = {"Bu_PT": ak.to_numpy(ev["Bu_PT"]), "Bu_eta": eta, "log_IPCHI2": log_ipchi2}
    if nt_arr is not None:
        result["nTracks"] = nt_arr
    return result


def _collect_data_vars(cat: str) -> dict:
    """Collect data variables from all years/magnets."""
    combined = {}
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                d = _load_data_vars(p, cat)
                for k, v in d.items():
                    if k not in combined:
                        combined[k] = {"sig": [], "sb": [], "scale": v["scale"]}
                    combined[k]["sig"].append(v["sig"])
                    combined[k]["sb"].append(v["sb"])
            except Exception as e:
                log.warning(f"  Skip data {p}: {e}")
    for k in combined:
        combined[k]["sig"] = np.concatenate(combined[k]["sig"])
        combined[k]["sb"] = np.concatenate(combined[k]["sb"])
    return combined


def _collect_mc_vars(cat: str) -> dict:
    """Collect MC variables from all J/ψ MC files."""
    combined = {}
    for yr in YEARS:
        for mag in MAGNETS:
            p = MC_BASE / "Jpsi" / f"Jpsi_{yr}_{mag}.root"
            if not p.exists():
                continue
            try:
                d = _load_mc_vars(p, cat)
                for k, v in d.items():
                    combined.setdefault(k, []).append(v)
            except Exception as e:
                log.warning(f"  Skip MC {p}: {e}")
    return {k: np.concatenate(v) for k, v in combined.items()}


# Legacy loaders kept for plot_reweight_pt
def _load_sideband_data(path: Path, cat: str):
    d = _load_data_vars(path, cat)
    if "Bu_PT" not in d:
        return np.array([]), np.array([]), 1.0
    return d["Bu_PT"]["sig"], d["Bu_PT"]["sb"], d["Bu_PT"]["scale"]


def _load_mc_pt(path: Path, cat: str) -> np.ndarray:
    d = _load_mc_vars(path, cat)
    return d.get("Bu_PT", np.array([]))


# ════════════════════════════════════════════════════════════════════════════════
# PLOT (legacy single-variable function, kept for reference)
# ════════════════════════════════════════════════════════════════════════════════


def plot_reweight_pt(cat: str):
    """Produce reweight_Bu_PT_cmp.pdf (legacy — now covered by plot_all_reweight_cmps)."""
    log.info(f"=== {cat} pT reweighting comparison (legacy) ===")

    pt_edges, pt_weights = _load_weights(cat)
    mc_pt_all = []
    for yr in YEARS:
        for mag in MAGNETS:
            p = MC_BASE / "Jpsi" / f"Jpsi_{yr}_{mag}.root"
            if not p.exists():
                continue
            try:
                mc_pt_all.append(_load_mc_pt(p, cat))
            except Exception as e:
                log.warning(f"  Skip MC {p}: {e}")
    if not mc_pt_all:
        log.warning(f"  No MC events for {cat}")
        return
    mc_pt = np.concatenate(mc_pt_all)
    mc_rw = _apply_pt_weights(mc_pt, pt_edges, pt_weights)

    data_sig_all, data_sb_all = [], []
    scale_global = 1.0
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                pt_sig, pt_sb, scale = _load_sideband_data(p, cat)
                data_sig_all.append(pt_sig)
                data_sb_all.append(pt_sb)
                scale_global = scale
            except Exception as e:
                log.warning(f"  Skip data {p}: {e}")
    if not data_sig_all:
        log.warning(f"  No data for {cat}")
        return

    data_sig = np.concatenate(data_sig_all)
    data_sb = np.concatenate(data_sb_all)
    log.info(f"  Data: {len(data_sig)} sig, {len(data_sb)} sideband events")

    # Plot: reference style (all density=True step histograms)
    histstyle = {"histtype": "step", "linestyle": "--", "linewidth": 4, "density": True}
    ranges = [3000, 20000]
    bins = 40

    fig, ax = plt.subplots()

    # Unweighted MC
    ax.hist(
        mc_pt,
        weights=None,
        range=ranges,
        bins=bins,
        label=r"Unweighted MC",
        color=COLORS[0],
        **histstyle,
    )

    # Reweighted MC
    ax.hist(
        mc_pt,
        weights=mc_rw,
        range=ranges,
        bins=bins,
        label=r"Reweighted MC",
        color=COLORS[1],
        histtype="step",
        linestyle="-",
        linewidth=4,
        density=True,
    )

    # Sideband-subtracted data (positive sig - negative sb)
    # Combine into one histogram with positive/negative weights
    all_pt = np.concatenate([data_sig, data_sb])
    all_w = np.concatenate([np.ones(len(data_sig)), -scale_global * np.ones(len(data_sb))])
    ax.hist(
        all_pt,
        weights=all_w,
        range=ranges,
        bins=bins,
        label=r"Sideband-subtracted data",
        color=COLORS[2],
        histtype="step",
        linestyle=":",
        linewidth=3,
        density=True,
    )

    ax.yaxis.set_major_formatter(_fmt)
    ax.set_xlabel(r"$p_T(B^+)$ [MeV/$c$]")
    ax.set_ylabel("Normalized")
    ax.set_title(rf"$\Lambda_{{{cat}}}$ sample")
    ax.legend()

    out = figs_path(cat, "reweight", "reweight_Bu_PT_cmp.pdf")
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


def _sb_subtract_hist(sig, sb, scale, x_range, bins):
    """Return sideband-subtracted normalized histogram (centers, h, err, widths)."""
    all_v = np.concatenate([sig, sb])
    all_w = np.concatenate([np.ones(len(sig)), -scale * np.ones(len(sb))])
    h, edges = np.histogram(all_v, weights=all_w, range=x_range, bins=bins)
    bw = (x_range[1] - x_range[0]) / bins
    tot = np.sum(h) * bw
    if tot > 0:
        h_sq, _ = np.histogram(all_v, weights=all_w**2, range=x_range, bins=bins)
        err = np.sqrt(np.abs(h_sq)) / tot
        h = h / tot
    else:
        err = np.ones_like(h)
    centers = (edges[1:] + edges[:-1]) / 2
    widths = (edges[1:] - edges[:-1]) / 2
    return centers, h, err, widths


def _two_panel_cmp(
    cat: str,
    var: str,
    data_vars: dict,
    mc_vars: dict,
    mc_rw: np.ndarray,
    x_range,
    bins: int,
    xlabel: str,
    outfile: str,
    before: bool = True,
    log_y: bool = False,
):
    """
    Generic comparison plot: unweighted MC vs reweighted MC vs sideband data.
    before=True  → 'pt_cmp.pdf' style (show raw MC vs data).
    before=False → 'reweight_*.pdf' style (show reweighted MC vs data).
    """
    if var not in data_vars or var not in mc_vars:
        log.warning(f"  {var} not available, skipping {outfile}")
        return

    d = data_vars[var]
    sig, sb, scale = d["sig"], d["sb"], d["scale"]
    mc = mc_vars[var]

    centers, h_data, err, widths = _sb_subtract_hist(sig, sb, scale, x_range, bins)

    histstyle = {"histtype": "step", "linewidth": 4, "density": True}
    fig, ax = plt.subplots()

    if before:
        # Before: show unweighted MC vs data
        ax.hist(
            mc,
            range=x_range,
            bins=bins,
            label="MC (unweighted)",
            color=COLORS[0],
            linestyle="--",
            **histstyle,
        )
    else:
        # After: show both unweighted and reweighted MC vs data
        ax.hist(
            mc,
            range=x_range,
            bins=bins,
            label="MC (unweighted)",
            color=COLORS[0],
            linestyle="--",
            **histstyle,
        )
        ax.hist(
            mc,
            range=x_range,
            bins=bins,
            weights=mc_rw,
            label="MC (reweighted)",
            color=COLORS[1],
            linestyle="-",
            **histstyle,
        )

    # Sideband-subtracted data as error bars
    ax.errorbar(
        centers,
        h_data,
        xerr=widths,
        yerr=err,
        label="Sideband-subtracted data",
        ecolor="black",
        mfc="black",
        color="black",
        elinewidth=2,
        markersize=5,
        marker="o",
        fmt=" ",
    )

    ax.yaxis.set_major_formatter(_fmt)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Normalized", fontsize=18)
    ax.set_title(rf"$\Lambda_{{{cat}}}$ sample", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend(frameon=False, fontsize=15)
    if log_y:
        ax.set_yscale("log")

    out = figs_path(cat, "reweight", outfile)
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


def _sweight_cmp(
    cat: str,
    var: str,
    data_jpsi: dict,
    data_etac: dict,
    x_range,
    bins: int,
    xlabel: str,
    outfile: str,
):
    """
    sWeight cross-check: J/ψ signal-window data vs ηc signal-window data.
    Uses events in B+ signal window within each charmonium mass window.
    Both normalized to unity for shape comparison.
    """
    if var not in data_jpsi or var not in data_etac:
        log.warning(f"  {var} not available for sweight cmp, skipping {outfile}")
        return

    jpsi_arr = data_jpsi[var]["sig"]
    etac_arr = data_etac[var]["sig"]

    if len(jpsi_arr) == 0 or len(etac_arr) == 0:
        log.warning(f"  Empty arrays for {outfile}, skipping")
        return

    histstyle = {"histtype": "step", "linewidth": 4, "density": True}
    fig, ax = plt.subplots()

    ax.hist(
        jpsi_arr,
        range=x_range,
        bins=bins,
        label=r"$J/\psi$ window data",
        color=COLORS[0],
        linestyle="--",
        **histstyle,
    )
    ax.hist(
        etac_arr,
        range=x_range,
        bins=bins,
        label=r"$\eta_c$ window data",
        color=COLORS[1],
        linestyle="-",
        **histstyle,
    )

    ax.yaxis.set_major_formatter(_fmt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized")
    ax.set_title(rf"$\Lambda_{{{cat}}}$ sample")
    ax.legend(frameon=False, fontsize=11)

    out = figs_path(cat, "reweight", outfile)
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


def _load_ccbar_window_data(path: Path, cat: str, window: str) -> dict:
    """
    Load data kinematic vars for events in B+ signal window AND
    charmonium mass window ('jpsi' or 'etac').
    Used for sWeight cross-check plots.
    """

    M_JPSI_LOCAL = 3096.9
    M_ETAC_LOCAL = 2980.3

    want = [
        "Bu_MM",
        "L0_MM",
        "Bu_PT",
        "Bu_P",
        "Bu_IPCHI2_OWNPV",
        "nTracks",
        "L0_PE",
        "L0_PX",
        "L0_PY",
        "L0_PZ",
        "p_PE",
        "p_PX",
        "p_PY",
        "p_PZ",
        "h1_PE",
        "h1_PX",
        "h1_PY",
        "h1_PZ",
        "Bu_L0GlobalDecision_TIS",
        "Bu_L0PhysDecision_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="ak")

    mask = _trig_mask_ak(
        ev,
        ["Bu_L0GlobalDecision_TIS", "Bu_L0PhysDecision_TIS", "Bu_L0HadronDecision_TIS"],
        ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"],
        [
            "Bu_Hlt2Topo2BodyDecision_TOS",
            "Bu_Hlt2Topo3BodyDecision_TOS",
            "Bu_Hlt2Topo4BodyDecision_TOS",
        ],
        "Bu_PT",
    )
    mask = mask & (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)
    ev = ev[mask]

    # Charmonium mass (m(ΛpK⁻))
    mom_keys = [
        "L0_PE",
        "L0_PX",
        "L0_PY",
        "L0_PZ",
        "p_PE",
        "p_PX",
        "p_PY",
        "p_PZ",
        "h1_PE",
        "h1_PX",
        "h1_PY",
        "h1_PZ",
    ]
    if all(k in avail for k in mom_keys):
        E = ev["L0_PE"] + ev["p_PE"] + ev["h1_PE"]
        px = ev["L0_PX"] + ev["p_PX"] + ev["h1_PX"]
        py = ev["L0_PY"] + ev["p_PY"] + ev["h1_PY"]
        pz = ev["L0_PZ"] + ev["p_PZ"] + ev["h1_PZ"]
        ccbar_M = np.sqrt(
            np.maximum(
                ak.to_numpy(E) ** 2
                - ak.to_numpy(px) ** 2
                - ak.to_numpy(py) ** 2
                - ak.to_numpy(pz) ** 2,
                0.0,
            )
        )
    else:
        return {}

    # B+ corrected mass
    bu_corr = ak.to_numpy(ev["Bu_MM"]) - ak.to_numpy(ev["L0_MM"]) + M_LAMBDA_PDG

    # B+ signal window
    bu_sel = (bu_corr >= 5255) & (bu_corr <= 5305)

    # Charmonium window
    M_ref = M_JPSI_LOCAL if window == "jpsi" else M_ETAC_LOCAL
    cc_sel = np.abs(ccbar_M - M_ref) < 30.0

    sel = bu_sel & cc_sel
    ev = ev[sel]
    bu_corr = bu_corr[sel]
    ccbar_M = ccbar_M[sel]

    P = ak.to_numpy(ev["Bu_P"]) if "Bu_P" in avail else np.ones(len(ev))
    PT = ak.to_numpy(ev["Bu_PT"])
    pz_v = np.sqrt(np.maximum(P**2 - PT**2, 0.0))
    eta = 0.5 * np.log(np.maximum((P + pz_v) / np.maximum(P - pz_v, 1e-9), 1e-9))
    lip = (
        np.log(np.maximum(ak.to_numpy(ev["Bu_IPCHI2_OWNPV"]), 1e-9))
        if "Bu_IPCHI2_OWNPV" in avail
        else np.zeros(len(ev))
    )
    nt = ak.to_numpy(ev["nTracks"]).astype(float) if "nTracks" in avail else None

    result = {
        "Bu_PT": {"sig": PT, "sb": np.array([]), "scale": 1.0},
        "Bu_eta": {"sig": eta, "sb": np.array([]), "scale": 1.0},
        "log_IPCHI2": {"sig": lip, "sb": np.array([]), "scale": 1.0},
    }
    if nt is not None:
        result["nTracks"] = {"sig": nt, "sb": np.array([]), "scale": 1.0}
    return result


def _collect_ccbar_vars(cat: str, window: str) -> dict:
    combined = {}
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                d = _load_ccbar_window_data(p, cat, window)
                for k, v in d.items():
                    combined.setdefault(k, {"sig": [], "sb": [], "scale": 1.0})
                    combined[k]["sig"].append(v["sig"])
            except Exception as e:
                log.warning(f"  Skip {p} [{window}]: {e}")
    for k in combined:
        combined[k]["sig"] = np.concatenate(combined[k]["sig"])
    return combined


# Variable specs: (var_key, xlabel, x_range, bins, log_y)
VAR_SPECS = [
    ("Bu_PT", r"$p_T(B^+)$ [MeV/$c$]", [3000, 20000], 40, False),
    ("Bu_eta", r"$\eta(B^+)$", [2.0, 5.0], 40, False),
    ("log_IPCHI2", r"$\log(\chi^2_{\rm IP}(B^+))$", [-1.0, 4.0], 40, False),
    ("nTracks", r"Number of tracks", [0, 500], 40, False),
]
VAR_OUTFILES = {
    "Bu_PT": ("pt_cmp.pdf", "reweight_Bu_PT_cmp.pdf", "sweight_Bu_PT_cmp.pdf"),
    "Bu_eta": ("eta_cmp.pdf", None, None),
    "log_IPCHI2": (
        "IPCHI2_cmp.pdf",
        "reweight_Bu_IPCHI2_OWNPV_cmp.pdf",
        "sweight_Bu_IPCHI2_OWNPV_cmp.pdf",
    ),
    "nTracks": ("ntracks_cmp.pdf", "reweight_nTracks_cmp.pdf", "sweight_nTracks_cmp.pdf"),
}


def plot_all_reweight_cmps(cat: str):
    """Produce all reweighting comparison plots for one category."""
    log.info(f"  Loading data and MC variables [{cat}]...")

    # Load combined data and MC
    data_vars = _collect_data_vars(cat)
    mc_vars = _collect_mc_vars(cat)

    if not data_vars or not mc_vars:
        log.warning(f"  Missing data or MC for {cat}")
        return

    # PT reweighting weights
    pt_edges, pt_weights = _load_weights(cat)
    mc_pt = mc_vars.get("Bu_PT", np.array([]))
    mc_rw = (
        _apply_pt_weights(mc_pt, pt_edges, pt_weights) if len(mc_pt) > 0 else np.ones(len(mc_pt))
    )

    # For other variables, use PT weights applied to MC
    mc_rw_full = {}
    for var in mc_vars:
        mc_rw_full[var] = mc_rw  # same pT weights for all variables

    for var, xlabel, x_range, bins, log_y in VAR_SPECS:
        outfiles = VAR_OUTFILES.get(var, (None, None, None))

        # Before reweighting
        if outfiles[0]:
            _two_panel_cmp(
                cat,
                var,
                data_vars,
                mc_vars,
                mc_rw_full.get(var, np.ones(len(mc_vars.get(var, [])))),
                x_range,
                bins,
                xlabel,
                outfiles[0],
                before=True,
                log_y=log_y,
            )

        # After reweighting (only for vars with output file)
        if outfiles[1]:
            _two_panel_cmp(
                cat,
                var,
                data_vars,
                mc_vars,
                mc_rw_full.get(var, np.ones(len(mc_vars.get(var, [])))),
                x_range,
                bins,
                xlabel,
                outfiles[1],
                before=False,
                log_y=log_y,
            )

    # sWeight cross-checks (J/ψ window vs ηc window data)
    log.info(f"  Loading J/ψ and ηc window data [{cat}]...")
    data_jpsi = _collect_ccbar_vars(cat, "jpsi")
    data_etac = _collect_ccbar_vars(cat, "etac")

    for var, xlabel, x_range, bins, log_y in VAR_SPECS:
        outfiles = VAR_OUTFILES.get(var, (None, None, None))
        if outfiles[2]:
            _sweight_cmp(cat, var, data_jpsi, data_etac, x_range, bins, xlabel, outfiles[2])


def main():
    for cat in ("LL", "DD"):
        log.info(f"=== Category: Lambda{cat} ===")
        plot_all_reweight_cmps(cat)
    log.info("=== Done. ===")


if __name__ == "__main__":
    main()
