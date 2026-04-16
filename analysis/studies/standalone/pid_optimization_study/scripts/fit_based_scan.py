"""
Fit-Based PID Scan
==================
Performs 1-D cut scans over individual PID variables (p_ProbNNp, h1_ProbNNk,
h2_ProbNNk) and over PID_product by running an actual RooFit mass fit for
every grid point.

This is the methodologically correct approach for PID optimization because:
  - Signal (S) and background (B) are measured directly from the mass fit,
    not from a proxy that has systematic biases.
  - The sideband-proxy approach consistently overestimates background PID
    efficiency → proxy FOM always peaks at zero cut (wrong).
  - Fit-based FOM correctly captures the trade-off between signal efficiency
    and background reduction.

FOM definitions (identical to the fit_based_optimizer study):
  FOM1 = (N_jpsi + N_etac) / sqrt(N_bkg_at_jpsi)  — high-yield group
  FOM2 = (N_chic0 + N_chic1) / sqrt(N_chic_bkg)   — low-yield group

For each PID variable a separate 1-D scan is run while all other selection
cuts are held fixed at their pipeline baseline values (the pre-cut applied
during data loading: PID_product > 0 here, replaced scan-by-scan below).

Runtime estimate: ~30 fits × ~0.3 s/fit ≈ 10 s per variable, ~40 s total.

Usage
-----
  uv run python scripts/fit_based_scan.py [--category LL|DD|both] [--variable all|pid_product|p_probnnp|h1_probnnk|h2_probnnk]
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
# Path setup
# ---------------------------------------------------------------------------
STUDY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = STUDY_DIR.parent.parent  # analysis/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig
from modules.mass_fitter import MassFitter

OUTPUT_DIR = STUDY_DIR / "output" / "fit_based"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cache loader (shared with box_scan_proxy.py)
# ---------------------------------------------------------------------------


def _load_cache(project_root: Path):
    cache_dir = None
    for method in ["mva", "box"]:
        candidate = project_root / "generated" / "cache" / "pipeline" / method
        if candidate.exists():
            cache_dir = candidate
            break
    if cache_dir is None:
        raise RuntimeError("Pipeline cache not found. Run 'uv run snakemake load_data -j1' first.")
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


# ---------------------------------------------------------------------------
# Single fit helper
# ---------------------------------------------------------------------------


def _run_single_fit(
    events: ak.Array,
    config: StudyConfig,
    plot_dir: Path | None = None,
    label: str = "",
) -> dict | None:
    """
    Run one mass fit on ``events`` (already filtered by the caller).

    Returns dict with yields and background count, or None on fit failure.
    Keys: jpsi, etac, chic0, chic1, etac_2s (yield values), bkg (total bkg).
    """
    fitter = MassFitter(config)
    try:
        results = fitter.perform_fit(
            data_by_year={"scan": events},
            fit_combined=False,
            plot_dir=plot_dir,
            fit_label=label,
        )
    except Exception as exc:
        print(f"    [WARN] Fit failed ({exc}); skipping this point.")
        return None

    yields_raw = results.get("yields", {}).get("scan", {})
    if not yields_raw:
        return None

    state_map = {
        "jpsi": "jpsi",
        "etac_1s": "etac",
        "chic0": "chic0",
        "chic1": "chic1",
        "etac_2s": "etac_2s",
    }
    out = {}
    for fit_key, analysis_key in state_map.items():
        pair = yields_raw.get(fit_key, (0.0, 0.0))
        out[analysis_key] = float(pair[0]) if isinstance(pair, (list, tuple)) else float(pair)

    # Total background: sum of per-state background estimates from the fit
    bkg_raw = results.get("background", {}).get("scan", {})
    if bkg_raw:
        out["bkg"] = float(bkg_raw.get("total", 0.0))
    else:
        # Fall back: estimate from the ARGUS yield if background key absent
        out["bkg"] = float(yields_raw.get("bkg", (0.0,))[0])

    return out


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------


def run_fit_scan(
    var_name: str,
    branch: str,
    label_str: str,
    grid: np.ndarray,
    data_full: dict,
    category: str,
    config: StudyConfig,
) -> dict:
    """
    Scan ``branch`` > cut over ``grid``, running a mass fit at each point.

    data_full: {year: {category: ak.Array}} — the full pipeline cache.
    """
    years = sorted(data_full.keys())

    cuts, n_jpsi, n_etac, n_chic0, n_chic1, n_bkg = [], [], [], [], [], []
    fom1, fom2 = [], []

    print(f"\n  Fit-based scan: {label_str}  ({len(grid)} points)")

    for cut in grid:
        # Assemble events: all years merged, category filtered, PID cut applied
        slices = []
        for yr in years:
            if category not in data_full[yr]:
                continue
            ev = data_full[yr][category]
            br = ak.to_numpy(ev[branch])
            ev = ev[br > cut]
            slices.append(ev)

        if not slices:
            print(f"    cut={cut:.3f}  → no data")
            continue

        events = ak.concatenate(slices)
        n_ev = len(events)
        print(f"    cut={cut:.3f}  N={n_ev}", end="  ")

        if n_ev < 50:
            print("(too few events — skipping)")
            continue

        fit_res = _run_single_fit(events, config, plot_dir=None, label=f"{var_name}_cut{cut:.3f}")
        if fit_res is None:
            continue

        nj = max(fit_res.get("jpsi", 0.0), 0.0)
        ne = max(fit_res.get("etac", 0.0), 0.0)
        nc0 = max(fit_res.get("chic0", 0.0), 0.0)
        nc1 = max(fit_res.get("chic1", 0.0), 0.0)
        nb = max(fit_res.get("bkg", 1.0), 1.0)

        f1 = (nj + ne) / np.sqrt(nb) if nb > 0 else 0.0
        f2 = (nc0 + nc1) / np.sqrt(nc0 + nc1 + nb) if (nc0 + nc1 + nb) > 0 else 0.0

        print(f"J/ψ={nj:.0f}  ηc={ne:.0f}  B={nb:.0f}  FOM1={f1:.2f}  FOM2={f2:.3f}")

        cuts.append(float(cut))
        n_jpsi.append(nj)
        n_etac.append(ne)
        n_chic0.append(nc0)
        n_chic1.append(nc1)
        n_bkg.append(nb)
        fom1.append(f1)
        fom2.append(f2)

    if not cuts:
        print(f"  WARNING: no valid fit points for {label_str}")
        return {
            "var_name": var_name,
            "branch": branch,
            "label": label_str,
            "cuts": [],
            "fom1": [],
            "fom2": [],
        }

    cuts = np.array(cuts)
    fom1 = np.array(fom1)
    fom2 = np.array(fom2)

    best1 = int(np.argmax(fom1))
    best2 = int(np.argmax(fom2))

    return {
        "var_name": var_name,
        "branch": branch,
        "label": label_str,
        "cuts": cuts.tolist(),
        "n_jpsi": n_jpsi,
        "n_etac": n_etac,
        "n_chic0": n_chic0,
        "n_chic1": n_chic1,
        "n_bkg": n_bkg,
        "fom1": fom1.tolist(),
        "fom2": fom2.tolist(),
        "best_cut_fom1": float(cuts[best1]),
        "best_fom1": float(fom1[best1]),
        "n_jpsi_at_best1": float(n_jpsi[best1]),
        "n_etac_at_best1": float(n_etac[best1]),
        "n_bkg_at_best1": float(n_bkg[best1]),
        "best_cut_fom2": float(cuts[best2]),
        "best_fom2": float(fom2[best2]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_fit_scan(result: dict, outdir: Path, cat: str) -> None:
    cuts = np.array(result["cuts"])
    if len(cuts) == 0:
        return

    fom1 = np.array(result["fom1"])
    fom2 = np.array(result["fom2"])
    n_jpsi = np.array(result["n_jpsi"])
    n_etac = np.array(result["n_etac"])
    n_bkg = np.array(result["n_bkg"])
    lbl = result["label"]
    var = result["var_name"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Fit-Based PID Scan: {lbl}  [{cat}]", fontsize=11)

    ax = axes[0, 0]
    ax.plot(cuts, n_jpsi, "b-o", ms=5, label=r"$N_{J/\psi}$")
    ax.plot(cuts, n_etac, "g-s", ms=5, label=r"$N_{\eta_c}$")
    ax.set_xlabel(f"Cut on {lbl} (> value)")
    ax.set_ylabel("Fitted yield")
    ax.set_title("Signal yields vs PID cut")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(cuts, n_bkg, "r-^", ms=5)
    ax.set_xlabel(f"Cut on {lbl} (> value)")
    ax.set_ylabel("Background (ARGUS yield)")
    ax.set_title("Background yield vs PID cut")

    ax = axes[1, 0]
    ax.plot(cuts, fom1, "b-o", ms=5)
    ax.axvline(
        result["best_cut_fom1"],
        color="gray",
        ls="--",
        lw=1.2,
        label=f"opt = {result['best_cut_fom1']:.3f}",
    )
    ax.set_xlabel(f"Cut on {lbl} (> value)")
    ax.set_ylabel(r"$(N_{J/\psi}+N_{\eta_c})/\sqrt{B}$")
    ax.set_title("FOM 1 — high-yield group (fit-based)")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(cuts, fom2, "m-D", ms=5)
    ax.axvline(
        result["best_cut_fom2"],
        color="gray",
        ls="--",
        lw=1.2,
        label=f"opt = {result['best_cut_fom2']:.3f}",
    )
    ax.set_xlabel(f"Cut on {lbl} (> value)")
    ax.set_ylabel(r"$(N_{\chi_c0}+N_{\chi_c1})/\sqrt{S+B}$")
    ax.set_title("FOM 2 — low-yield group (fit-based)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fname = outdir / f"fit_scan_{var}_{cat}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fit-based PID scan")
    parser.add_argument("--category", choices=["LL", "DD", "both"], default="both")
    parser.add_argument(
        "--variable",
        choices=["all", "pid_product", "p_probnnp", "h1_probnnk", "h2_probnnk"],
        default="all",
    )
    args = parser.parse_args()

    import tomllib

    with open(STUDY_DIR / "config.toml", "rb") as f:
        cfg_dict = tomllib.load(f)

    # Build a StudyConfig pointing at the main selection.toml
    config = StudyConfig(
        config_file=str(PROJECT_ROOT / "config" / "selection.toml"),
        output_dir=str(STUDY_DIR / "output" / "fit_based"),
    )

    print("Loading pipeline cache...")
    data_full, mc_full = _load_cache(PROJECT_ROOT)

    pg = cfg_dict["pid_grids"]
    all_variables = {
        "pid_product": {
            "branch": "PID_product",
            "label": "PID product (p × h1 × h2)",
            "grid": np.arange(
                pg["pid_product_min"],
                pg["pid_product_max"] + pg["pid_product_step"] / 2,
                pg["pid_product_step"],
            ),
        },
        "p_probnnp": {
            "branch": "p_ProbNNp",
            "label": "p ProbNNp",
            "grid": np.arange(
                pg["p_probnnp_min"],
                pg["p_probnnp_max"] + pg["p_probnnp_step"] / 2,
                pg["p_probnnp_step"],
            ),
        },
        "h1_probnnk": {
            "branch": "h1_ProbNNk",
            "label": "h1 ProbNNk (K+)",
            "grid": np.arange(
                pg["h1_probnnk_min"],
                pg["h1_probnnk_max"] + pg["h1_probnnk_step"] / 2,
                pg["h1_probnnk_step"],
            ),
        },
        "h2_probnnk": {
            "branch": "h2_ProbNNk",
            "label": "h2 ProbNNk (K-)",
            "grid": np.arange(
                pg["h2_probnnk_min"],
                pg["h2_probnnk_max"] + pg["h2_probnnk_step"] / 2,
                pg["h2_probnnk_step"],
            ),
        },
    }

    vars_to_run = list(all_variables.keys()) if args.variable == "all" else [args.variable]
    categories = ["LL", "DD"] if args.category == "both" else [args.category]

    for cat in categories:
        print(f"\n{'='*70}")
        print(f"Category: {cat}")
        print(f"{'='*70}")

        cat_results = {}
        for vname in vars_to_run:
            v = all_variables[vname]
            result = run_fit_scan(
                var_name=vname,
                branch=v["branch"],
                label_str=v["label"],
                grid=v["grid"],
                data_full={yr: data_full[yr] for yr in data_full},
                category=cat,
                config=config,
            )
            plot_fit_scan(result, OUTPUT_DIR, cat)
            cat_results[vname] = result

        out_json = OUTPUT_DIR / f"fit_scan_results_{cat}.json"
        with open(out_json, "w") as fp:
            json.dump(cat_results, fp, indent=2)
        print(f"\n  Results saved: {out_json}")

    # Summary
    print("\n" + "=" * 70)
    print("FIT-BASED SCAN SUMMARY")
    print("=" * 70)
    for cat in categories:
        json_path = OUTPUT_DIR / f"fit_scan_results_{cat}.json"
        if not json_path.exists():
            continue
        with open(json_path) as fp:
            res = json.load(fp)
        print(f"\n  [{cat}]")
        hdr = f"  {'Variable':<28} {'Opt FOM1 cut':<15} {'FOM1':<10} {'Opt FOM2 cut':<15} {'FOM2':<10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for vname, r in res.items():
            if not r.get("cuts"):
                continue
            print(
                f"  {r['label']:<28} "
                f"{r['best_cut_fom1']:<15.3f} "
                f"{r['best_fom1']:<10.2f} "
                f"{r['best_cut_fom2']:<15.3f} "
                f"{r['best_fom2']:<10.3f}"
            )


if __name__ == "__main__":
    main()
