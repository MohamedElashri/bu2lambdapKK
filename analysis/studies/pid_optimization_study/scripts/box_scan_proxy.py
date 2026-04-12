"""
Proxy-Based PID Box Scan
========================
Performs 1-D cut scans over individual PID variables (p_ProbNNp, h1_ProbNNk,
h2_ProbNNk) and over PID_product using the traditional sideband-proxy FOM.

Two FOM definitions are evaluated at every cut point:

  Proxy FOM 1 (eff ratio):
      FOM1 = eps_sig / sqrt(eps_bkg)
    where
      eps_sig = fraction of MC signal events passing the cut
      eps_bkg = fraction of data sideband events passing the cut

  Proxy FOM 2 (signal significance):
      FOM2 = S_proxy / sqrt(B_proxy)
    where
      S_proxy = eps_sig * N_data_charmonium_signalregion (charmonium-window, B+ window)
      B_proxy = N_sideband_passing_cut * (B+_signal_width / B+_sideband_width)

KNOWN LIMITATION
----------------
Both proxy FOMs are ANTI-CORRELATED with the true fit-based FOM (Pearson r ≈ -0.9).
Root cause: data sidebands consist of real, well-identified particles whose PID quality
exceeds that of the true combinatorial background in the signal window.  The proxy
therefore overestimates background PID efficiency → optimizer always prefers no PID cut.
This is a systematic bias, not a statistical effect.  Results from this script must be
interpreted in light of fit_based_scan.py, which gives the correct answer.

Usage
-----
  uv run python scripts/box_scan_proxy.py [--category LL|DD|both]
"""

import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup: resolve project root (analysis/) and add to sys.path
# ---------------------------------------------------------------------------
STUDY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = STUDY_DIR.parent.parent  # analysis/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

OUTPUT_DIR = STUDY_DIR / "output" / "box_proxy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cache(project_root: Path):
    """Load preprocessed data and MC from the main pipeline cache."""
    cache_dir = None
    for method in ["mva", "box"]:
        candidate = project_root / "analysis_output" / method / "cache"
        if candidate.exists():
            cache_dir = candidate
            break
    if cache_dir is None:
        raise RuntimeError(
            "Pipeline cache not found. Run 'uv run snakemake load_data -j1' "
            "from the analysis/ directory first."
        )

    main_config = StudyConfig(
        config_file=str(project_root / "config" / "selection.toml"),
        output_dir=str(cache_dir.parent),
    )
    cache = CacheManager(cache_dir=str(cache_dir))
    deps = cache.compute_dependencies(
        config_files=list((project_root / "config").glob("*.toml")),
        code_files=[
            project_root / "modules" / "clean_data_loader.py",
            project_root / "scripts" / "load_data.py",
        ],
        extra_params={"years": ["2016", "2017", "2018"], "track_types": ["LL", "DD"]},
    )
    data_full = cache.load("preprocessed_data", dependencies=deps)
    mc_full = cache.load("preprocessed_mc", dependencies=deps)
    if data_full is None or mc_full is None:
        raise RuntimeError("Cache entries missing. Rebuild with 'snakemake load_data -j1'.")
    return data_full, mc_full


def _get_arrays(data_full, mc_full, category: str):
    """Concatenate data and MC arrays for the requested category."""
    years = list(data_full.keys())
    data_list = [data_full[yr][category] for yr in years if category in data_full[yr]]
    data_cat = ak.concatenate(data_list)

    mc_list = []
    for state in mc_full:
        if category in mc_full[state]:
            mc_list.append(mc_full[state][category])
    mc_cat = ak.concatenate(mc_list) if mc_list else ak.Array([])

    return data_cat, mc_cat


def _sideband_mask(bu_mass, cfg: dict):
    """Boolean mask for B+ mass sideband region."""
    lo_min, lo_max = cfg["b_low_sideband"]
    hi_min, hi_max = cfg["b_high_sideband"]
    return ((bu_mass > lo_min) & (bu_mass < lo_max)) | ((bu_mass > hi_min) & (bu_mass < hi_max))


def _signal_proxy_mask(bu_mass, cc_mass, state_cfg: dict, b_sig: list):
    """Boolean mask for charmonium signal region within B+ signal window."""
    c, w = state_cfg["center"], state_cfg["window"]
    b_min, b_max = b_sig
    return (bu_mass > b_min) & (bu_mass < b_max) & (cc_mass > c - w) & (cc_mass < c + w)


# ---------------------------------------------------------------------------
# Core scan function
# ---------------------------------------------------------------------------


def run_proxy_scan(
    var_name: str,
    branch: str,
    grid: np.ndarray,
    data: ak.Array,
    mc: ak.Array,
    study_cfg: dict,
    label: str,
) -> dict:
    """
    Perform a 1-D proxy-based scan over ``branch``.

    Returns a dict with cut-point arrays and FOM arrays, suitable for
    JSON serialisation and plotting.
    """
    bu_mass = data["Bu_MM_corrected"]
    cc_mass = data["M_LpKm_h2"]

    b_sig = study_cfg["mass_windows"]["bu_corrected"]
    b_sig_width = b_sig[1] - b_sig[0]

    lo_min, lo_max = study_cfg["mass_windows"]["b_low_sideband"]
    hi_min, hi_max = study_cfg["mass_windows"]["b_high_sideband"]
    sb_width = (lo_max - lo_min) + (hi_max - hi_min)

    sb_mask = _sideband_mask(bu_mass, study_cfg["mass_windows"])
    n_sb_total = float(ak.sum(sb_mask))

    # High-yield proxy signal count (J/ψ + ηc) at zero PID cut
    high_yield_states = ["jpsi", "etac"]
    n_sig_proxy_base = 0.0
    for st in high_yield_states:
        sc = study_cfg["signal_regions"][st]
        smask = _signal_proxy_mask(bu_mass, cc_mass, sc, b_sig)
        n_sr = float(ak.sum(smask))
        # background estimate from sidebands scaled to signal window width
        n_sb_in_cc = float(
            ak.sum(
                sb_mask
                & (
                    (cc_mass > sc["center"] - sc["window"])
                    & (cc_mass < sc["center"] + sc["window"])
                )
            )
        )
        b_est = n_sb_in_cc * (b_sig_width / sb_width) if sb_width > 0 else 0.0
        n_sig_proxy_base += max(n_sr - b_est, 0.0)

    mc_branch = ak.to_numpy(mc[branch]) if len(mc) > 0 else np.array([])
    n_mc_total = len(mc_branch)

    cuts, eps_sig, eps_bkg, fom1, fom2 = [], [], [], [], []

    for cut in grid:
        # Signal efficiency from MC
        if n_mc_total > 0:
            mc_pass = float(np.sum(mc_branch > cut))
            e_sig = mc_pass / n_mc_total
        else:
            e_sig = 1.0

        # Background efficiency from data sidebands
        data_branch = ak.to_numpy(data[branch])
        sb_pass = float(np.sum((ak.to_numpy(sb_mask)) & (data_branch > cut)))
        e_bkg = sb_pass / n_sb_total if n_sb_total > 0 else 0.0

        # FOM 1: efficiency ratio
        f1 = e_sig / np.sqrt(e_bkg) if e_bkg > 0 else 0.0

        # FOM 2: signal significance proxy
        s_proxy = e_sig * n_sig_proxy_base
        b_proxy = sb_pass * (b_sig_width / sb_width) if sb_width > 0 else 0.0
        f2 = s_proxy / np.sqrt(b_proxy) if b_proxy > 0 else 0.0

        cuts.append(float(cut))
        eps_sig.append(e_sig)
        eps_bkg.append(e_bkg)
        fom1.append(f1)
        fom2.append(f2)

    cuts = np.array(cuts)
    fom1 = np.array(fom1)
    fom2 = np.array(fom2)
    eps_sig = np.array(eps_sig)
    eps_bkg = np.array(eps_bkg)

    best1_idx = int(np.argmax(fom1))
    best2_idx = int(np.argmax(fom2))

    return {
        "var_name": var_name,
        "branch": branch,
        "label": label,
        "cuts": cuts.tolist(),
        "eps_sig": eps_sig.tolist(),
        "eps_bkg": eps_bkg.tolist(),
        "fom1": fom1.tolist(),
        "fom2": fom2.tolist(),
        "best_cut_fom1": cuts[best1_idx],
        "best_fom1": fom1[best1_idx],
        "eps_sig_at_best1": eps_sig[best1_idx],
        "eps_bkg_at_best1": eps_bkg[best1_idx],
        "best_cut_fom2": cuts[best2_idx],
        "best_fom2": fom2[best2_idx],
        "n_mc_total": n_mc_total,
        "n_sb_total": n_sb_total,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scan(result: dict, outdir: Path, cat: str) -> None:
    cuts = np.array(result["cuts"])
    fom1 = np.array(result["fom1"])
    fom2 = np.array(result["fom2"])
    eps_sig = np.array(result["eps_sig"])
    eps_bkg = np.array(result["eps_bkg"])
    var = result["var_name"]
    lbl = result["label"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Proxy-Based PID Scan: {lbl}  [{cat}]\n"
        r"WARNING: proxy FOM is anti-correlated with true fit-based FOM (see fit_based_scan.py)",
        fontsize=10,
    )

    ax = axes[0, 0]
    ax.plot(cuts, eps_sig, "b-o", ms=4, label=r"$\varepsilon_{\rm sig}$ (MC)")
    ax.plot(cuts, eps_bkg, "r-s", ms=4, label=r"$\varepsilon_{\rm bkg}$ (sideband proxy)")
    ax.axvline(
        result["best_cut_fom1"],
        color="gray",
        ls="--",
        lw=1,
        label=f"opt FOM1 cut = {result['best_cut_fom1']:.3f}",
    )
    ax.set_xlabel(f"Cut on {lbl} (> value)")
    ax.set_ylabel("Efficiency")
    ax.set_title("Signal and Background Efficiency")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    ax = axes[0, 1]
    ratio = eps_sig / np.sqrt(np.maximum(eps_bkg, 1e-9))
    ax.plot(cuts, ratio, "g-^", ms=4)
    ax.axvline(
        result["best_cut_fom1"],
        color="gray",
        ls="--",
        lw=1,
        label=f"opt cut = {result['best_cut_fom1']:.3f}",
    )
    ax.set_xlabel(f"Cut on {lbl} (> value)")
    ax.set_ylabel(r"$\varepsilon_{\rm sig}/\sqrt{\varepsilon_{\rm bkg}}$")
    ax.set_title("FOM 1 (efficiency ratio)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(cuts, fom2, "m-D", ms=4)
    ax.axvline(
        result["best_cut_fom2"],
        color="gray",
        ls="--",
        lw=1,
        label=f"opt cut = {result['best_cut_fom2']:.3f}",
    )
    ax.set_xlabel(f"Cut on {lbl} (> value)")
    ax.set_ylabel(r"$S_{\rm proxy}/\sqrt{B_{\rm proxy}}$")
    ax.set_title("FOM 2 (proxy significance)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    sig_rej = 1.0 - eps_bkg  # background rejection
    ax.plot(eps_sig, sig_rej, "k-o", ms=4)
    ax.set_xlabel(r"$\varepsilon_{\rm sig}$")
    ax.set_ylabel(r"$1 - \varepsilon_{\rm bkg}$ (bkg rejection)")
    ax.set_title("Efficiency curve (signal vs bkg rejection)")
    # Mark optimal points
    best1 = int(np.argmax(fom1))
    ax.plot(eps_sig[best1], sig_rej[best1], "r*", ms=12, label="opt FOM1")
    best2 = int(np.argmax(fom2))
    ax.plot(eps_sig[best2], sig_rej[best2], "b*", ms=12, label="opt FOM2")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = outdir / f"proxy_scan_{var}_{cat}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Proxy-based PID box scan")
    parser.add_argument("--category", choices=["LL", "DD", "both"], default="both")
    args = parser.parse_args()

    # Load study config
    cfg_path = STUDY_DIR / "config.toml"
    import tomllib

    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    print("Loading pipeline cache...")
    data_full, mc_full = _load_cache(PROJECT_ROOT)

    categories = ["LL", "DD"] if args.category == "both" else [args.category]

    pid_grids_cfg = cfg["pid_grids"]

    variables = [
        {
            "var_name": "pid_product",
            "branch": "PID_product",
            "label": "PID product (p × h1 × h2)",
            "grid": np.arange(
                pid_grids_cfg["pid_product_min"],
                pid_grids_cfg["pid_product_max"] + pid_grids_cfg["pid_product_step"] / 2,
                pid_grids_cfg["pid_product_step"],
            ),
        },
        {
            "var_name": "p_probnnp",
            "branch": "p_ProbNNp",
            "label": "p ProbNNp",
            "grid": np.arange(
                pid_grids_cfg["p_probnnp_min"],
                pid_grids_cfg["p_probnnp_max"] + pid_grids_cfg["p_probnnp_step"] / 2,
                pid_grids_cfg["p_probnnp_step"],
            ),
        },
        {
            "var_name": "h1_probnnk",
            "branch": "h1_ProbNNk",
            "label": "h1 ProbNNk (K+)",
            "grid": np.arange(
                pid_grids_cfg["h1_probnnk_min"],
                pid_grids_cfg["h1_probnnk_max"] + pid_grids_cfg["h1_probnnk_step"] / 2,
                pid_grids_cfg["h1_probnnk_step"],
            ),
        },
        {
            "var_name": "h2_probnnk",
            "branch": "h2_ProbNNk",
            "label": "h2 ProbNNk (K-)",
            "grid": np.arange(
                pid_grids_cfg["h2_probnnk_min"],
                pid_grids_cfg["h2_probnnk_max"] + pid_grids_cfg["h2_probnnk_step"] / 2,
                pid_grids_cfg["h2_probnnk_step"],
            ),
        },
    ]

    all_results = {}

    for cat in categories:
        print(f"\n{'='*70}")
        print(f"Category: {cat}")
        print(f"{'='*70}")
        data_cat, mc_cat = _get_arrays(data_full, mc_full, cat)
        print(f"  Data events : {len(data_cat)}")
        print(f"  MC events   : {len(mc_cat)}")

        cat_results = {}
        for v in variables:
            print(f"\n  Scanning {v['label']}  ({len(v['grid'])} points)...")
            if v["branch"] not in data_cat.fields:
                print(f"  WARNING: branch {v['branch']} not in data — skipping")
                continue
            if len(mc_cat) > 0 and v["branch"] not in mc_cat.fields:
                print(f"  WARNING: branch {v['branch']} not in MC — signal eff will be 1.0")

            result = run_proxy_scan(
                var_name=v["var_name"],
                branch=v["branch"],
                grid=v["grid"],
                data=data_cat,
                mc=mc_cat,
                study_cfg=cfg,
                label=v["label"],
            )

            print(
                f"    FOM1 optimal cut: {result['best_cut_fom1']:.3f}  "
                f"(eps_sig={result['eps_sig_at_best1']:.3f}, "
                f"eps_bkg={result['eps_bkg_at_best1']:.3f})"
            )
            print(f"    FOM2 optimal cut: {result['best_cut_fom2']:.3f}")

            plot_scan(result, OUTPUT_DIR, cat)
            cat_results[v["var_name"]] = result

        # Save JSON
        out_json = OUTPUT_DIR / f"proxy_scan_results_{cat}.json"
        with open(out_json, "w") as fp:
            json.dump(cat_results, fp, indent=2)
        print(f"\n  Results saved: {out_json}")

        all_results[cat] = cat_results

    # Summary table
    print("\n" + "=" * 70)
    print("PROXY SCAN SUMMARY")
    print("=" * 70)
    header = f"{'Variable':<28} {'Cat':<4} {'Opt FOM1 cut':<14} {'eps_sig':<10} {'eps_bkg':<10} {'Opt FOM2 cut':<14}"
    print(header)
    print("-" * len(header))
    for cat, cat_res in all_results.items():
        for vname, r in cat_res.items():
            print(
                f"{r['label']:<28} {cat:<4} "
                f"{r['best_cut_fom1']:<14.3f} "
                f"{r['eps_sig_at_best1']:<10.3f} "
                f"{r['eps_bkg_at_best1']:<10.3f} "
                f"{r['best_cut_fom2']:<14.3f}"
            )

    print("\nNOTE: All proxy-based optimal cuts are expected to cluster near zero.")
    print("This is consistent with prior studies (pid_proxy_comparison).")
    print("The proxy systematically overestimates background PID efficiency.")
    print("Compare with fit_based_scan.py for the true optimal cut.")


if __name__ == "__main__":
    main()
