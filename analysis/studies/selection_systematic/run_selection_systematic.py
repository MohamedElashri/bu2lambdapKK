"""
Selection Systematic Uncertainty

Method:
  For the MVA-based selection, the nominal BDT threshold is shifted by ±1 step of the
  threshold scan grid (nominally step = 0.01).  For each shifted threshold, the full
  selection is re-applied to the cached data, the mass fit is re-run, and the yields
  are extracted.  The systematic is the maximum absolute yield shift:

    σ_sel(state) = max( |N(thr+δ) − N_nom| , |N(thr−δ) − N_nom| )

  Since this uses the already-existing pipeline cache (apply_cuts output), it avoids a
  full re-run of the data loading and cut steps — only the BDT threshold changes.

Output: output/selection_systematics_{branch}_{category}.json
"""

import json
import logging
import sys
from pathlib import Path

import awkward as ak

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig
from modules.mass_fitter import MassFitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

THRESHOLD_STEP = 0.01  # one grid step for BDT threshold variation


def load_cut_data(branch: str, category: str, cache_dir: str, config_dir: str) -> dict | None:
    """Load the post-cut (but pre-MVA-threshold) cached data."""
    config = StudyConfig.from_dir(config_dir)
    cache = CacheManager(cache_dir=cache_dir)
    cut_deps = cache.compute_dependencies(
        config_files=config.config_paths(),
        code_files=[project_root / "scripts" / "apply_cuts.py"],
    )
    return cache.load(f"{branch}_{category}_final_data", dependencies=cut_deps)


def apply_mva_threshold(
    data_dict: dict, model_path: Path, features: list, threshold: float
) -> dict:
    """Re-apply a given MVA threshold to already-loaded data."""
    if not model_path.exists():
        logger.warning(f"MVA model not found at {model_path} — skipping threshold re-application.")
        return data_dict

    import pandas as pd
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    filtered = {}
    for key, events in data_dict.items():
        if len(events) == 0:
            filtered[key] = events
            continue
        df_dict = {}
        for feat in features:
            arr = events[feat]
            if "var" in str(ak.type(arr)):
                arr = ak.firsts(arr)
            df_dict[feat] = ak.to_numpy(arr)
        df = pd.DataFrame(df_dict)
        preds = model.predict_proba(df)[:, 1]
        filtered[key] = events[preds > threshold]
    return filtered


def fit_and_extract(config, data_dict: dict) -> dict:
    """Run mass fit and return {state: (value, error)} for the combined dataset."""
    fitter = MassFitter(config=config)
    result = fitter.perform_fit(data_dict, fit_combined=True)  # plot_dir=None: no plots
    if result and "combined" in result.get("yields", {}):
        return result["yields"]["combined"]
    return {}


def compute_selection_systematics(
    branch: str, category: str, config_dir: str, cache_dir: str, output_dir: str
):
    config = StudyConfig.from_dir(config_dir, output_dir=output_dir)

    data_dict = load_cut_data(branch, category, cache_dir, config_dir)
    if data_dict is None:
        logger.error("Cached cut data not found — run main pipeline first.")
        sys.exit(1)

    # Load nominal BDT threshold and features
    mva_dir = project_root / "analysis_output" / "mva"
    cuts_path = mva_dir / branch / category / "models" / "optimized_cuts.json"
    mva_features = config.xgboost.get(
        "features", ["Bu_DTF_chi2", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV", "Bu_PT"]
    )

    threshold_nom = 0.89  # conservative default
    if cuts_path.exists():
        with open(cuts_path) as f:
            cuts_data = json.load(f)
        threshold_nom = cuts_data.get("mva_threshold_high", threshold_nom)
    logger.info(f"Nominal MVA threshold = {threshold_nom:.4f}")

    model_path = mva_dir / branch / category / "models" / "mva_model.cbm"

    logger.info(f"=== Selection systematics: branch={branch}, category={category} ===")

    # Nominal fit (data already at nominal threshold from apply_cuts)
    nominal_yields = fit_and_extract(config, data_dict)

    systematics = {}
    all_states = ["jpsi", "etac", "chic0", "chic1"]

    variations = {
        "thr_up": threshold_nom + THRESHOLD_STEP,
        "thr_dn": max(0.01, threshold_nom - THRESHOLD_STEP),
    }

    var_yields = {}
    for var_name, thr in variations.items():
        logger.info(f"  Applying threshold {thr:.4f} ({var_name})...")
        data_var = apply_mva_threshold(data_dict, model_path, mva_features, thr)
        var_yields[var_name] = fit_and_extract(config, data_var)

    for state in all_states:
        n_nom, e_nom = nominal_yields.get(state, (0.0, 0.0))
        shifts = []
        for var_name, ylds in var_yields.items():
            n_var, _ = ylds.get(state, (n_nom, 0.0))
            shift = abs(n_var - n_nom)
            shifts.append(shift)
            logger.info(
                f"  {state} [{var_name}: thr={variations[var_name]:.4f}]: N={n_var:.1f}  δN={n_var - n_nom:+.1f}"
            )

        syst = max(shifts) if shifts else 0.0
        rel_syst = syst / n_nom if n_nom > 0 else 0.0
        systematics[state] = {
            "nominal_yield": n_nom,
            "nominal_err": e_nom,
            "sel_syst_abs": syst,
            "sel_syst_rel": rel_syst,
            "threshold_nominal": threshold_nom,
            "threshold_step": THRESHOLD_STEP,
        }
        logger.info(f"  {state}: σ_sel={syst:.1f} ({100*rel_syst:.1f}%)")

    out_path = Path(output_dir) / f"selection_systematics_{branch}_{category}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(systematics, f, indent=2)
    logger.info(f"Selection systematics saved to {out_path}")
    return systematics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Selection systematic")
    parser.add_argument("--branch", default="high_yield")
    parser.add_argument("--category", default="LL")
    parser.add_argument("--config-dir", default="../../config")
    parser.add_argument("--cache-dir", default="../../analysis_output/mva/cache")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    compute_selection_systematics(
        branch=args.branch,
        category=args.category,
        config_dir=args.config_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
