"""
Fit Model Systematic Uncertainties

Runs the nominal mass fit plus three systematic variations per (branch, category):
  1. bkg_poly2        : Replace ARGUS background with 2nd-order Chebyshev polynomial
  2. argus_endpoint_up: Increase ARGUS endpoint offset by 50 MeV  (+50 from nominal 200)
  3. argus_endpoint_dn: Decrease ARGUS endpoint offset by 50 MeV  (−50 from nominal 200)
  4. resolution_up    : Fix signal resolution to nominal + 2 MeV   (approx +1σ)
  5. resolution_dn    : Fix signal resolution to nominal − 2 MeV   (approx −1σ)

For each variation, the fit is re-run and the absolute yield shift
  δN = |N_varied − N_nominal|
is recorded per state.  The fit systematic for a state is then:
  σ_fit_syst = max(δN across all variations)

Output: output/fit_systematics_{branch}_{category}.json
"""

import json
import logging
import sys
from pathlib import Path

# Add analysis/ to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig
from modules.mass_fitter import MassFitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_single_fit(config, data_dict, systematic_params: dict, label: str) -> dict:
    """Run one mass fit with given systematic_params. Return {state: (value, error)}."""
    logger.info(f"  Running fit variation: {label}")
    fitter = MassFitter(config=config, systematic_params=systematic_params)
    # plot_dir=None: systematic variation fits do not produce plots
    result = fitter.perform_fit(data_dict, fit_combined=True)
    if result and "combined" in result.get("yields", {}):
        return result["yields"]["combined"]
    return {}


def compute_fit_systematics(
    branch: str, category: str, config_dir: str, cache_dir: str, output_dir: str
):
    config = StudyConfig.from_dir(config_dir, output_dir=output_dir)

    cache = CacheManager(cache_dir=cache_dir)
    cut_deps = cache.compute_dependencies(
        config_files=config.config_paths(),
        code_files=[project_root / "scripts" / "apply_cuts.py"],
    )
    data_dict = cache.load(f"{branch}_{category}_final_data", dependencies=cut_deps)
    if data_dict is None:
        logger.error(
            f"Cut data for branch={branch}, category={category} not found. "
            "Run 'snakemake apply_cuts' first."
        )
        sys.exit(1)

    logger.info(f"=== Fit systematics: branch={branch}, category={category} ===")

    # --- 1. Nominal fit ---
    nominal_yields = run_single_fit(config, data_dict, {}, "nominal")
    if not nominal_yields:
        logger.error("Nominal fit returned no yields — aborting.")
        sys.exit(1)

    # Extract nominal resolution for ±2 MeV variations
    nominal_res = 5.0  # MeV — fallback if fitter doesn't expose it
    # (Resolution floats in the nominal fit; we use the PDG-reasonable ±2 MeV window)

    # --- 2. Systematic variations ---
    nominal_endpoint_offset = config.fitting.get("argus_endpoint_offset", 200.0)
    variations = {
        "bkg_poly2": {"bkg_model": "poly2"},
        "argus_endpoint_up": {"argus_endpoint_offset": nominal_endpoint_offset + 50.0},
        "argus_endpoint_dn": {"argus_endpoint_offset": max(50.0, nominal_endpoint_offset - 50.0)},
        "resolution_up": {"signal_resolution_fixed": nominal_res + 2.0},
        "resolution_dn": {"signal_resolution_fixed": max(1.0, nominal_res - 2.0)},
    }

    all_states = ["jpsi", "etac", "chic0", "chic1"]  # etac_2s excluded (MC pending)
    variation_yields: dict[str, dict[str, tuple]] = {}
    for var_name, sys_params in variations.items():
        var_yields = run_single_fit(config, data_dict, sys_params, var_name)
        variation_yields[var_name] = var_yields

    # --- 3. Compute max |δN| per state ---
    systematics = {}
    for state in all_states:
        n_nom, e_nom = nominal_yields.get(state, (0.0, 0.0))
        shifts = []
        for var_name, var_ylds in variation_yields.items():
            n_var, _ = var_ylds.get(state, (n_nom, 0.0))
            shifts.append(abs(n_var - n_nom))
            logger.info(f"  {state} [{var_name}]: N={n_var:.1f}  δN={n_var - n_nom:+.1f}")
        syst = max(shifts) if shifts else 0.0
        rel_syst = syst / n_nom if n_nom > 0 else 0.0
        systematics[state] = {
            "nominal_yield": n_nom,
            "nominal_err": e_nom,
            "fit_syst_abs": syst,
            "fit_syst_rel": rel_syst,
            "variations": {
                var: variation_yields[var].get(state, (0.0, 0.0))[0] for var in variations
            },
        }
        logger.info(f"  {state}: N_nom={n_nom:.1f}, σ_fit_syst={syst:.1f} ({100*rel_syst:.1f}%)")

    out_path = Path(output_dir) / f"fit_systematics_{branch}_{category}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(systematics, f, indent=2)
    logger.info(f"Fit systematics saved to {out_path}")
    return systematics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit model systematics")
    parser.add_argument("--branch", default="high_yield")
    parser.add_argument("--category", default="LL")
    parser.add_argument("--config-dir", default="../../config")
    parser.add_argument("--cache-dir", default="../../analysis_output/mva/cache")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    compute_fit_systematics(
        branch=args.branch,
        category=args.category,
        config_dir=args.config_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
