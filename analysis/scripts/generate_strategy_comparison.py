"""
Generate strategy-comparison figures: signal retained vs background removed
or signal retained vs background kept
for every combination of (optimisation method) × (PID variant).

Six strategies per category (LL / DD):
  Box — no PID cut        (PID_prod at 0.0, first grid point)
  Box — PID product cut   (PID_prod > fit-based optimal)
  Box — individual PID    (p_ProbNNp + h1 + h2 each at proxy-optimal, product approx.)
  MVA A — baseline        (BDT on topology only, no PID features)
  MVA B — individual PID  (BDT + p_ProbNNp + h1_ProbNNk + h2_ProbNNk)
  MVA C — PID product     (BDT + PID_product)

Signal efficiency   = fraction of MC signal events passing the cut / threshold.
Background rejection = fraction of sideband events failing the cut / threshold.
Background kept      = fraction of sideband events passing the cut / threshold.

Box strategies → proxy_scan_results (eps_sig, eps_bkg) at the optimal PID cut.
MVA strategies → reload saved XGBoost models, score signal MC and sideband data,
                 evaluate at the stored opt_bdt_cut (High_Yield FOM1 threshold).

Output: figs/strategy_comparison.pdf
"""

import json
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import awkward as ak
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from _paths import ANALYSIS_DIR as PROJECT_ROOT
from _paths import SLIDES_DIR, resolve_pid_study_dir, resolve_pipeline_cache_dir

STUDY_DIR = resolve_pid_study_dir()
PROXY_DIR = STUDY_DIR / "output" / "box_proxy"
FIT_DIR = STUDY_DIR / "output" / "fit_based"
MVA_DIR = STUDY_DIR / "output" / "mva"
FIGS_DIR = SLIDES_DIR / "figs"

# Add analysis root so we can import cache/config helpers
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
        "axes.linewidth": 1.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)

VARIANTS = {
    "A_baseline": ["Bu_DTF_chi2", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV", "Bu_PT"],
    "B_individual": [
        "Bu_DTF_chi2",
        "Bu_FDCHI2_OWNPV",
        "Bu_IPCHI2_OWNPV",
        "Bu_PT",
        "p_ProbNNp",
        "h1_ProbNNk",
        "h2_ProbNNk",
    ],
    "C_product": ["Bu_DTF_chi2", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV", "Bu_PT", "PID_product"],
}

STRATEGY_META = {
    "box_nopid": {"label": "Box: no PID cut", "color": "#aaaaaa", "marker": "D", "ms": 10},
    "box_product": {"label": "Box: PID product cut", "color": "#333333", "marker": "D", "ms": 10},
    "box_indiv": {"label": "Box: individual PID cuts", "color": "#888888", "marker": "D", "ms": 10},
    "mva_A": {"label": "MVA: no PID features", "color": "#1f77b4", "marker": "o", "ms": 12},
    "mva_B": {"label": "MVA: individual PID feat.", "color": "#d62728", "marker": "o", "ms": 12},
    "mva_C": {"label": "MVA: PID product feat.", "color": "#ff7f0e", "marker": "o", "ms": 12},
}

# ── Cache loader (mirrors mva_pid_study.py) ───────────────────────────────────


def _load_cache():
    def _load_compatible(cache: CacheManager, name: str, deps: dict, description: str):
        cached = cache.load(name, dependencies=deps)
        if cached is not None:
            return cached

        fallback_meta = []
        for meta_path in cache.metadata_dir.glob("*.json"):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            if meta.get("description") == description:
                fallback_meta.append((meta.get("created_at", ""), meta.get("key")))

        for _, key in sorted(fallback_meta, reverse=True):
            data_path = cache.data_dir / f"{key}.pkl"
            if not data_path.exists():
                continue
            with open(data_path, "rb") as f:
                return pickle.load(f)
        return None

    cache_dir = resolve_pipeline_cache_dir()
    cfg = StudyConfig.from_dir(PROJECT_ROOT / "config", output_dir=str(cache_dir.parent))
    magnets = cfg.get_input_magnets() or ["MD", "MU"]
    states = cfg.get_input_mc_states() or ["Jpsi", "etac", "chic0", "chic1"]
    years = cfg.get_input_years() or ["2016", "2017", "2018"]
    cache = CacheManager(cache_dir=str(cache_dir))
    deps = cache.compute_dependencies(
        config_files=cfg.config_paths(),
        code_files=[
            PROJECT_ROOT / "modules" / "clean_data_loader.py",
            PROJECT_ROOT / "scripts" / "load_data.py",
        ],
        extra_params={
            "years": years,
            "track_types": ["LL", "DD"],
            "magnets": magnets,
            "states": states,
        },
    )
    data_full = _load_compatible(cache, "preprocessed_data", deps, "Data Pre-processed")
    mc_full = _load_compatible(cache, "preprocessed_mc", deps, "MC Pre-processed")
    if data_full is None or mc_full is None:
        raise RuntimeError("Cache entries missing. Rebuild with 'snakemake load_data -j1'.")
    return data_full, mc_full


def _events_to_df(events, features):
    d = {}
    for f in features:
        br = events[f]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        d[f] = ak.to_numpy(br)
    return pd.DataFrame(d).dropna()


def _sideband_mask(bu_mm):
    return (bu_mm > 5330.0) | ((bu_mm >= 5150.0) & (bu_mm <= 5230.0))


# ── MVA operating-point evaluation ───────────────────────────────────────────


def mva_efficiencies(data_full, mc_full):
    """
    For each (variant, category) load the saved XGBoost model, score signal MC
    and sideband background, then return (sig_eff, bkg_rej) at the stored
    opt_bdt_cut (High_Yield, S/sqrt(B)).

    Returns: dict[(variant_id, cat)] = (sig_eff_pct, bkg_rej_pct)
    """
    results = {}
    for cat in ["LL", "DD"]:
        with open(MVA_DIR / f"mva_pid_summary_{cat}.json") as f:
            summary = json.load(f)

        # Build per-variant optimal thresholds (High_Yield, FOM1 = S/sqrt(B))
        opt_cuts = {}
        for entry in summary["fom_summary"]:
            if entry["group"] == "High_Yield" and entry["fom_type"] == "S/sqrt(B)":
                opt_cuts[entry["variant"]] = entry["opt_bdt_cut"]

        # Prepare combined signal MC and sideband background
        years = sorted(data_full.keys())
        data_cat = ak.concatenate([data_full[yr][cat] for yr in years if cat in data_full[yr]])
        mc_cat = ak.concatenate([mc_full[st][cat] for st in mc_full if cat in mc_full[st]])

        bkg_mask = _sideband_mask(data_cat["Bu_MM_corrected"])
        bkg_ev = data_cat[bkg_mask]

        for vid, features in VARIANTS.items():
            model_path = MVA_DIR / f"xgboost_{vid}_{cat}.json"
            if not model_path.exists():
                continue

            model = xgb.XGBClassifier()
            model.load_model(str(model_path))

            df_sig = _events_to_df(mc_cat, features)
            df_bkg = _events_to_df(bkg_ev, features)

            if df_sig.empty or df_bkg.empty:
                continue

            scores_sig = model.predict_proba(df_sig[features].values)[:, 1]
            scores_bkg = model.predict_proba(df_bkg[features].values)[:, 1]

            thr = opt_cuts.get(vid, 0.5)
            sig_eff = float(np.mean(scores_sig >= thr)) * 100.0
            bkg_rej = float(np.mean(scores_bkg < thr)) * 100.0

            results[(vid, cat)] = (sig_eff, bkg_rej)
            print(f"  MVA {vid} {cat}: sig={sig_eff:.1f}%  bkg_rej={bkg_rej:.1f}%  (thr={thr:.2f})")

    return results


# ── Box operating-point evaluation ───────────────────────────────────────────


def box_efficiencies(mode: str = "fit"):
    """
    Read proxy_scan_results and fit_scan_results to extract (sig_eff, bkg_rej)
    at the chosen PID cut for each variable.

    mode="fit"   → use fit-based optimal box PID cuts (historical deck)
    mode="proxy" → use proxy-based optimal box PID cuts (fully blinded inputs)

    Returns: dict[(strategy_id, cat)] = (sig_eff_pct, bkg_rej_pct)
    """
    results = {}
    for cat in ["LL", "DD"]:
        with open(PROXY_DIR / f"proxy_scan_results_{cat}.json") as f:
            proxy = json.load(f)
        with open(FIT_DIR / f"fit_scan_results_{cat}.json") as f:
            fit = json.load(f)

        def _at_cut(vname, opt_cut):
            pr = proxy.get(vname, {})
            if not pr.get("cuts"):
                return None
            cuts = np.array(pr["cuts"])
            eps_sig = np.array(pr["eps_sig"])
            eps_bkg = np.array(pr["eps_bkg"])
            idx = int(np.argmin(np.abs(cuts - opt_cut)))
            return float(eps_sig[idx]) * 100.0, float(1 - eps_bkg[idx]) * 100.0

        # Box: no PID cut (cut = 0)
        pt = _at_cut("pid_product", 0.0)
        if pt:
            results[("box_nopid", cat)] = pt

        # Box: PID product at selected optimal
        if mode == "proxy":
            opt = proxy.get("pid_product", {}).get("best_cut_fom1", 0.0)
        else:
            fi_pp = fit.get("pid_product", {})
            opt = fi_pp.get("best_cut_fom1", 0.25)
        pt = _at_cut("pid_product", opt)
        if pt:
            results[("box_product", cat)] = pt

        # Box: independent individual PID cuts (product-efficiency approximation)
        indiv = ["p_probnnp", "h1_probnnk", "h2_probnnk"]
        sig_list, bkg_list, ok = [], [], True
        for vname in indiv:
            if mode == "proxy":
                pr_v = proxy.get(vname, {})
                if not pr_v.get("cuts"):
                    ok = False
                    break
                opt_v = pr_v.get("best_cut_fom1", 0.0)
            else:
                fi = fit.get(vname, {})
                if not fi.get("cuts"):
                    ok = False
                    break
                opt_v = fi.get("best_cut_fom1", 0.0)
            pt_v = _at_cut(vname, opt_v)
            if pt_v is None:
                ok = False
                break
            sig_list.append(pt_v[0] / 100.0)
            bkg_list.append(1 - pt_v[1] / 100.0)  # convert back to eps_bkg
        if ok:
            combined_sig = float(np.prod(sig_list)) * 100.0
            combined_bkg = (1 - float(np.prod(bkg_list))) * 100.0
            results[("box_indiv", cat)] = (combined_sig, combined_bkg)

        for sid, (s, b) in {k: v for k, v in results.items() if k[1] == cat}.items():
            print(f"  Box  {sid[0]:<12} {cat}: sig={s:.1f}%  bkg_rej={b:.1f}%")

    return results


# ── Main figure ───────────────────────────────────────────────────────────────


def make_figure(
    mode: str = "fit",
    output_name: str = "strategy_comparison.pdf",
    y_mode: str = "removed",
):
    print("Loading pipeline cache for MVA scoring...")
    data_full, mc_full = _load_cache()
    print("Scoring MVA models...")
    mva_pts = mva_efficiencies(data_full, mc_full)
    print(f"Reading box scan results ({mode} mode)...")
    box_pts = box_efficiencies(mode=mode)

    if y_mode not in {"removed", "kept"}:
        raise ValueError(f"Unsupported y_mode={y_mode!r}")

    all_pts = {}
    for (vid, cat), v in mva_pts.items():
        sid = {"A_baseline": "mva_A", "B_individual": "mva_B", "C_product": "mva_C"}[vid]
        all_pts[(sid, cat)] = v
    all_pts.update(box_pts)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.0))
    if y_mode == "removed":
        title = "Signal Retained vs Background Removed: All Optimisation Strategies"
    else:
        title = "Signal Retained vs Background Kept: All Optimisation Strategies"
    if mode == "proxy":
        title += "\n(proxy-defined operating points only)"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    shared_legend_handles = []
    for sid, meta in STRATEGY_META.items():
        shared_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=meta["color"],
                marker=meta["marker"],
                linestyle="None",
                markeredgecolor="black",
                markeredgewidth=1.0,
                ms=13.0,
                label=meta["label"],
            )
        )

    for ax, cat in zip(axes, ["LL", "DD"]):

        for sid, meta in STRATEGY_META.items():
            key = (sid, cat)
            if key not in all_pts:
                continue
            x, y_removed = all_pts[key]
            y = y_removed if y_mode == "removed" else max(100.0 - y_removed, 1e-3)

            if y_mode == "removed":
                # Lift the point slightly off the zero line so it is fully visible
                y_plot = max(y, 1.5)
            else:
                y_plot = y

            handle = ax.scatter(
                x,
                y_plot,
                color=meta["color"],
                marker=meta["marker"],
                s=meta["ms"] ** 2,
                zorder=5,
                edgecolors="black",
                linewidths=1.0,
            )

        if y_mode == "removed":
            ax.axhline(50, color="gray", lw=0.7, ls=":", alpha=0.45)
            ax.axvline(50, color="gray", lw=0.7, ls=":", alpha=0.45)
        else:
            for y_ref in [0.1, 1, 10]:
                ax.axhline(y_ref, color="gray", lw=0.7, ls=":", alpha=0.35)
            ax.axvline(50, color="gray", lw=0.7, ls=":", alpha=0.45)

        ax.set_xlabel("Signal retained (% of MC signal events passing cut/threshold)", fontsize=12)
        if y_mode == "removed":
            ax.set_ylabel("Background removed (% of sideband events rejected)", fontsize=12)
        else:
            ax.set_ylabel("Background kept (% of sideband events passing)", fontsize=12)
        ax.set_title(rf"$\Lambda$ {cat}", fontsize=13)
        ax.set_xlim(-5, 115)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        if y_mode == "removed":
            ax.set_ylim(-5, 115)  # start below 0 so points on y=0 are not clipped
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.text(
                111, 111, "ideal", fontsize=9, color="gray", ha="right", va="top", style="italic"
            )
        else:
            ax.set_yscale("log")
            ax.set_ylim(0.05, 120)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
            ax.text(
                111,
                0.06,
                "ideal",
                fontsize=9,
                color="gray",
                ha="right",
                va="bottom",
                style="italic",
            )

    fig.legend(
        handles=shared_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=13.0,
        framealpha=0.92,
        frameon=True,
        edgecolor="#aaaaaa",
        handlelength=1.2,
        handletextpad=0.55,
        borderpad=0.7,
        labelspacing=0.4,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.23, wspace=0.15)
    out = FIGS_DIR / output_name
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    make_figure()
