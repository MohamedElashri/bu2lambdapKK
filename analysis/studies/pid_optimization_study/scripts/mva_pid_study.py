"""
MVA PID Study
=============
Trains an XGBoost BDT under three different feature configurations and
compares performance, to determine whether including PID variables improves
the classifier:

  Variant A — Baseline (no PID):
      Bu_DTF_chi2, Bu_FDCHI2_OWNPV, Bu_IPCHI2_OWNPV, Bu_PT

  Variant B — Individual PID variables:
      same as A + p_ProbNNp, h1_ProbNNk, h2_ProbNNk

  Variant C — PID product:
      same as A + PID_product (= p_ProbNNp × h1_ProbNNk × h2_ProbNNk)

For each variant the script produces:
  - ROC curve (AUC comparison)
  - Overtraining check (KS test, train/test overlap)
  - Feature importance plot
  - Proxy FOM scan (BDT score threshold, identical methodology to the main
    MVA study — inherits the same known proxy bias for PID features)
  - A summary JSON with AUC, optimal BDT threshold, and FOM at that threshold

IMPORTANT — Known Limitations of Variants B and C
--------------------------------------------------
The BDT training background is drawn from the data B+ mass sideband.
Sideband events are real, well-identified particles with PID quality
comparable to or better than the signal MC.  As a result:
  - The BDT cannot distinguish signal from background using PID alone
    (both have good PID by construction of the sideband sample).
  - Including PID features may REDUCE classifier AUC if the BDT learns
    a spurious PID boundary that degrades performance on true combinatorial
    background (which has lower PID than the sideband proxy).
  - A higher AUC for Variant B/C vs Variant A therefore does NOT imply
    that PID inclusion is beneficial for the actual analysis.

The correct evaluation is in fit_based_scan.py, which applies PID cuts
to data and measures S and B from a mass fit.

Usage
-----
  uv run python scripts/mva_pid_study.py [--category LL|DD|both]
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
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
STUDY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = STUDY_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

OUTPUT_DIR = STUDY_DIR / "output" / "mva"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "A_baseline": {
        "label": "Baseline (no PID)",
        "features": ["Bu_DTF_chi2", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV", "Bu_PT"],
    },
    "B_individual": {
        "label": "Individual PID (p_ProbNNp, h1_ProbNNk, h2_ProbNNk)",
        "features": [
            "Bu_DTF_chi2",
            "Bu_FDCHI2_OWNPV",
            "Bu_IPCHI2_OWNPV",
            "Bu_PT",
            "p_ProbNNp",
            "h1_ProbNNk",
            "h2_ProbNNk",
        ],
    },
    "C_product": {
        "label": "PID product (PID_product = p × h1 × h2)",
        "features": ["Bu_DTF_chi2", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV", "Bu_PT", "PID_product"],
    },
}

# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------


def _load_cache(project_root: Path):
    cache_dir = None
    for method in ["mva", "box"]:
        candidate = project_root / "analysis_output" / method / "cache"
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
# Data preparation for one category
# ---------------------------------------------------------------------------


def prepare_data(
    data_full: dict,
    mc_full: dict,
    features: list,
    category: str,
    b_high_min: float = 5330.0,
    b_low_min: float = 5150.0,
    b_low_max: float = 5230.0,
    random_state: int = 42,
    test_size: float = 0.3,
):
    """
    Build signal (MC) and background (sideband data) arrays for training.

    Returns dict with X_train, X_test, y_train, y_test, data_combined, mc_combined.
    """
    years = sorted(data_full.keys())
    data_list = [data_full[yr][category] for yr in years if category in data_full[yr]]
    data_combined = ak.concatenate(data_list)

    mc_list = []
    for state in mc_full:
        if category in mc_full[state]:
            mc_list.append(mc_full[state][category])
    mc_combined = ak.concatenate(mc_list) if mc_list else ak.Array([])

    bu_mm = data_combined["Bu_MM_corrected"]
    bkg_mask = (bu_mm > b_high_min) | ((bu_mm >= b_low_min) & (bu_mm <= b_low_max))
    bkg_ev = data_combined[bkg_mask]

    def to_df(events, feat_list):
        d = {}
        for f in feat_list:
            br = events[f]
            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            d[f] = ak.to_numpy(br)
        return pd.DataFrame(d)

    df_bkg = to_df(bkg_ev, features)
    df_bkg["label"] = 0

    df_sig = to_df(mc_combined, features)
    df_sig["label"] = 1

    df = pd.concat([df_sig, df_bkg], ignore_index=True).dropna()
    X = df[features].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": features,
        "data_combined": data_combined,
        "mc_combined": mc_combined,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_xgboost(ml_data: dict, cfg_dict: dict, random_state: int = 42) -> xgb.XGBClassifier:
    hp = cfg_dict["mva"]["hyperparameters"]
    n_neg = int(np.sum(ml_data["y_train"] == 0))
    n_pos = int(np.sum(ml_data["y_train"] == 1))
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        scale_pos_weight=spw,
        random_state=random_state,
        **hp,
    )
    model.fit(
        ml_data["X_train"],
        ml_data["y_train"],
        eval_set=[(ml_data["X_train"], ml_data["y_train"]), (ml_data["X_test"], ml_data["y_test"])],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Plots: ROC, overtraining, feature importance
# ---------------------------------------------------------------------------


def plot_roc(results: dict, variant_id: str, outdir: Path, cat: str) -> None:
    plt.figure(figsize=(7, 6))
    for vid, r in results.items():
        plt.plot(r["fpr"], r["tpr"], label=f"{VARIANTS[vid]['label']}  AUC={r['auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve Comparison  [{cat}]")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / f"roc_comparison_{cat}.pdf")
    plt.close()


def plot_overtraining(ml_data: dict, model, variant_id: str, outdir: Path, cat: str) -> None:
    feats = ml_data["features"]

    def _score(ev, feat_list):
        df = {}
        for f in feat_list:
            br = ev[f]
            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            df[f] = ak.to_numpy(br)
        return model.predict_proba(pd.DataFrame(df)[feat_list].values)[:, 1]

    y_tr_pred = model.predict_proba(ml_data["X_train"])[:, 1]
    y_te_pred = model.predict_proba(ml_data["X_test"])[:, 1]
    y_tr = ml_data["y_train"]
    y_te = ml_data["y_test"]

    bins = np.linspace(0, 1, 40)
    fig, ax = plt.subplots(figsize=(9, 6))

    for sig_mask, te_mask, col, lbl in [
        (y_tr == 1, y_te == 1, "blue", "Signal"),
        (y_tr == 0, y_te == 0, "red", "Background"),
    ]:
        ax.hist(
            y_tr_pred[sig_mask],
            bins=bins,
            alpha=0.4,
            color=col,
            density=True,
            label=f"{lbl} (Train)",
        )
        hist_te, _ = np.histogram(y_te_pred[te_mask], bins=bins, density=True)
        bc = 0.5 * (bins[1:] + bins[:-1])
        n = max(np.sum(te_mask), 1)
        ax.errorbar(
            bc,
            hist_te,
            yerr=np.sqrt(hist_te / n),
            fmt="o",
            color=col,
            markersize=4,
            label=f"{lbl} (Test)",
        )

    ks_sig = ks_2samp(y_tr_pred[y_tr == 1], y_te_pred[y_te == 1]).pvalue
    ks_bkg = ks_2samp(y_tr_pred[y_tr == 0], y_te_pred[y_te == 0]).pvalue
    ax.set_title(
        f"Overtraining Check — {VARIANTS[variant_id]['label']}  [{cat}]\n"
        f"KS p-value  Sig={ks_sig:.3f}  Bkg={ks_bkg:.3f}"
    )
    ax.set_xlabel("BDT score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"overtraining_{variant_id}_{cat}.pdf")
    plt.close()


def plot_feature_importance(model, features: list, variant_id: str, outdir: Path, cat: str) -> None:
    imp = model.feature_importances_
    idx = np.argsort(imp)
    pos = np.arange(len(idx)) + 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(pos, imp[idx], align="center", color="steelblue")
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(features)[idx])
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"Feature Importance — {VARIANTS[variant_id]['label']}  [{cat}]")
    plt.tight_layout()
    plt.savefig(outdir / f"feature_importance_{variant_id}_{cat}.pdf")
    plt.close()


def plot_fom_scan(
    fom_history: dict, best_cuts: dict, variant_id: str, outdir: Path, cat: str
) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"FOM Scan (proxy) — {VARIANTS[variant_id]['label']}  [{cat}]\n"
        r"WARNING: proxy FOM biased for PID features (see fit_based_scan.py)",
        fontsize=9,
    )
    for ax, group in zip(axes, ["High_Yield", "Low_Yield"]):
        ax.plot(thresholds, fom_history[group]["S/sqrt(B)"], label="S/sqrt(B)")
        ax.plot(thresholds, fom_history[group]["S/sqrt(S+B)"], label="S/sqrt(S+B)")
        ax.axvline(
            best_cuts[group]["S/sqrt(B)"]["cut"],
            color="blue",
            ls="--",
            label=f"opt S/sqrt(B) = {best_cuts[group]['S/sqrt(B)']['cut']:.2f}",
        )
        ax.axvline(
            best_cuts[group]["S/sqrt(S+B)"]["cut"],
            color="orange",
            ls="--",
            label=f"opt S/sqrt(S+B) = {best_cuts[group]['S/sqrt(S+B)']['cut']:.2f}",
        )
        ax.set_xlabel("BDT score threshold")
        ax.set_ylabel("FOM")
        ax.set_title(group.replace("_", " "))
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"fom_scan_{variant_id}_{cat}.pdf")
    plt.close()


# ---------------------------------------------------------------------------
# Proxy FOM scan over BDT threshold
# ---------------------------------------------------------------------------


def proxy_fom_scan(model, ml_data: dict, cfg_dict: dict) -> tuple[dict, dict]:
    """
    Scan BDT score threshold, compute proxy FOM (identical to main MVA study).

    Returns (fom_history, best_cuts).
    """

    mw = cfg_dict["mass_windows"]
    sr = cfg_dict["signal_regions"]

    data = ml_data["data_combined"]
    feats = ml_data["features"]

    def _score_data():
        d = {}
        for f in feats:
            br = data[f]
            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            d[f] = ak.to_numpy(br)
        return model.predict_proba(pd.DataFrame(d)[feats].values)[:, 1]

    bdt_score = _score_data()
    bu_mass = ak.to_numpy(data["Bu_MM_corrected"])
    cc_mass = ak.to_numpy(data["M_LpKm_h2"])

    b_sig_min, b_sig_max = mw["bu_corrected"]
    lo_min, lo_max = mw["b_low_sideband"]
    hi_min, hi_max = mw["b_high_sideband"]
    b_sig_w = b_sig_max - b_sig_min
    lo_w = lo_max - lo_min
    hi_w = hi_max - hi_min

    in_b_sig = (bu_mass > b_sig_min) & (bu_mass < b_sig_max)
    in_b_lo = (bu_mass > lo_min) & (bu_mass < lo_max)
    in_b_hi = (bu_mass > hi_min) & (bu_mass < hi_max)

    # Pre-compute signal region masks per state
    state_masks = {}
    for state, sc in sr.items():
        c, w = sc["center"], sc["window"]
        in_cc = (cc_mass > c - w) & (cc_mass < c + w)
        state_masks[state] = {
            "sig": in_cc & in_b_sig,
            "lo": in_cc & in_b_lo,
            "hi": in_cc & in_b_hi,
        }

    # Baseline yields (no BDT cut)
    n_expected = {}
    for state, masks in state_masks.items():
        n_sr = float(np.sum(masks["sig"]))
        n_lo = float(np.sum(masks["lo"])) / lo_w if lo_w > 0 else 0.0
        n_hi = float(np.sum(masks["hi"])) / hi_w if hi_w > 0 else 0.0
        b_est = ((n_lo + n_hi) / 2.0) * b_sig_w
        n_expected[state] = max(n_sr - b_est, 1.0)

    HIGH_YIELD = ["jpsi", "etac"]
    LOW_YIELD = ["chic0", "chic1"]

    thresholds = np.linspace(0.01, 0.99, 99)
    history = {g: {"S/sqrt(B)": [], "S/sqrt(S+B)": []} for g in ["High_Yield", "Low_Yield"]}
    best_cuts = {
        g: {"S/sqrt(B)": {"cut": 0.0, "fom": -np.inf}, "S/sqrt(S+B)": {"cut": 0.0, "fom": -np.inf}}
        for g in ["High_Yield", "Low_Yield"]
    }

    mc_combined = ml_data["mc_combined"]
    n_mc_total = len(mc_combined)
    mc_feats = {}
    for f in feats:
        br = mc_combined[f]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        mc_feats[f] = ak.to_numpy(br)
    mc_scores = model.predict_proba(pd.DataFrame(mc_feats)[feats].values)[:, 1]

    for thr in thresholds:
        data_mask = bdt_score > thr
        mc_eps = float(np.sum(mc_scores > thr)) / n_mc_total if n_mc_total > 0 else 0.0

        state_sb = {}
        for state, masks in state_masks.items():
            n_lo_a = float(np.sum(data_mask & masks["lo"]))
            n_hi_a = float(np.sum(data_mask & masks["hi"]))
            b_est = (
                ((n_lo_a / lo_w + n_hi_a / hi_w) / 2.0) * b_sig_w
                if (lo_w > 0 and hi_w > 0)
                else 0.0
            )
            s_est = mc_eps * n_expected[state]
            state_sb[state] = (s_est, b_est)

        for group, states in [("High_Yield", HIGH_YIELD), ("Low_Yield", LOW_YIELD)]:
            s = sum(state_sb[st][0] for st in states if st in state_sb)
            b = sum(state_sb[st][1] for st in states if st in state_sb)
            f1 = s / np.sqrt(b) if b > 0 else 0.0
            f2 = s / np.sqrt(s + b) if (s + b) > 0 else 0.0
            history[group]["S/sqrt(B)"].append(f1)
            history[group]["S/sqrt(S+B)"].append(f2)
            for fom_key, fval in [("S/sqrt(B)", f1), ("S/sqrt(S+B)", f2)]:
                if fval > best_cuts[group][fom_key]["fom"]:
                    best_cuts[group][fom_key] = {
                        "cut": float(thr),
                        "fom": float(fval),
                        "s": s,
                        "b": b,
                    }

    return history, best_cuts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MVA PID study — 3 feature variants")
    parser.add_argument("--category", choices=["LL", "DD", "both"], default="both")
    args = parser.parse_args()

    import tomllib

    with open(STUDY_DIR / "config.toml", "rb") as f:
        cfg_dict = tomllib.load(f)

    print("Loading pipeline cache...")
    data_full, mc_full = _load_cache(PROJECT_ROOT)

    categories = ["LL", "DD"] if args.category == "both" else [args.category]

    for cat in categories:
        print(f"\n{'='*70}")
        print(f"Category: {cat}")
        print(f"{'='*70}")

        roc_collector = {}
        summary_rows = []

        for variant_id, variant in VARIANTS.items():
            print(f"\n  -- Variant {variant_id}: {variant['label']} --")

            ml_data = prepare_data(
                data_full,
                mc_full,
                variant["features"],
                cat,
                random_state=cfg_dict["mva"]["random_state"],
                test_size=cfg_dict["mva"]["test_size"],
            )

            print(f"     Train: {ml_data['X_train'].shape}  Test: {ml_data['X_test'].shape}")

            model = train_xgboost(ml_data, cfg_dict, random_state=cfg_dict["mva"]["random_state"])

            y_pred = model.predict_proba(ml_data["X_test"])[:, 1]
            fpr, tpr, _ = roc_curve(ml_data["y_test"], y_pred)
            roc_auc = auc(fpr, tpr)
            print(f"     AUC = {roc_auc:.4f}")

            roc_collector[variant_id] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

            # Plots
            plot_overtraining(ml_data, model, variant_id, OUTPUT_DIR, cat)
            plot_feature_importance(model, variant["features"], variant_id, OUTPUT_DIR, cat)

            # Save model
            model.save_model(OUTPUT_DIR / f"xgboost_{variant_id}_{cat}.json")

            # Proxy FOM scan
            fom_hist, best_cuts = proxy_fom_scan(model, ml_data, cfg_dict)
            plot_fom_scan(fom_hist, best_cuts, variant_id, OUTPUT_DIR, cat)

            for group in ["High_Yield", "Low_Yield"]:
                for fom_key in ["S/sqrt(B)", "S/sqrt(S+B)"]:
                    bc = best_cuts[group][fom_key]
                    summary_rows.append(
                        {
                            "variant": variant_id,
                            "label": variant["label"],
                            "category": cat,
                            "group": group,
                            "fom_type": fom_key,
                            "opt_bdt_cut": bc["cut"],
                            "fom_value": bc["fom"],
                            "auc": roc_auc,
                        }
                    )

        # ROC comparison plot (all variants on one canvas)
        plot_roc(roc_collector, "", OUTPUT_DIR, cat)

        # Save summary
        summary_path = OUTPUT_DIR / f"mva_pid_summary_{cat}.json"
        with open(summary_path, "w") as fp:
            json.dump({"roc": roc_collector, "fom_summary": summary_rows}, fp, indent=2)
        print(f"\n  Summary saved: {summary_path}")

    # Print final table
    print("\n" + "=" * 70)
    print("MVA PID STUDY SUMMARY (AUC)")
    print("=" * 70)
    for cat in categories:
        sp = OUTPUT_DIR / f"mva_pid_summary_{cat}.json"
        if not sp.exists():
            continue
        with open(sp) as f:
            data = json.load(f)
        print(f"\n  [{cat}]")
        print(f"  {'Variant':<40} {'AUC':<8}")
        print("  " + "-" * 48)
        seen = set()
        for row in data["fom_summary"]:
            if row["variant"] not in seen:
                print(f"  {row['label']:<40} {row['auc']:.4f}")
                seen.add(row["variant"])

    print("\nNOTE: Higher AUC for variants B/C vs A does NOT imply PID helps.")
    print("Sideband background has similar PID quality to signal MC → AUC")
    print("comparison cannot diagnose whether PID improves analysis sensitivity.")
    print("Use fit_based_scan.py results for the correct evaluation.")


if __name__ == "__main__":
    main()
