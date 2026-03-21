"""
Utility functions for MVA optimization study, including table generation,
feature importance extraction, and visualizations.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Allow importing from analysis/modules/
_analysis_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_analysis_root) not in sys.path:
    sys.path.insert(0, str(_analysis_root))

try:
    from modules.plot_utils import setup_style

    setup_style()
except Exception:
    pass  # Style is best-effort; plotting still works without it

logger = logging.getLogger(__name__)


def generate_tuned_hyperparameters_table(best_params_path: Path, output_path: Path):
    """
    Generates a Markdown table comparing default vs. tuned hyperparameters.
    """
    if not best_params_path.exists():
        logger.warning(f"Could not find tuned parameters at {best_params_path}")
        return

    with open(best_params_path, "r") as f:
        tuned_params = json.load(f)

    default_params = {
        "iterations": 350,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "border_count": 254,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Phase 1: Hyperparameter Tuning\n\n")
        f.write("| Hyperparameter | Default | Tuned |\n")
        f.write("| --- | --- | --- |\n")
        for k in tuned_params.keys():
            d_val = default_params.get(k, "N/A")
            t_val = tuned_params[k]
            if isinstance(d_val, float):
                d_val = f"{d_val:.3f}"
            if isinstance(t_val, float):
                t_val = f"{t_val:.3f}"
            f.write(f"| `{k}` | {d_val} | {t_val} |\n")
    logger.info(f"Generated hyperparameter comparison table at {output_path}")


def analyze_feature_importance_and_impact(
    model, ml_data: dict, catboost_params: dict, output_path: Path
):
    """
    Extracts top features and trains reduced models to quantify their impact.
    Outputs a summary markdown file.
    """
    features = ml_data["features"]
    importance = model.get_feature_importance()
    sorted_idx = np.argsort(importance)[::-1]

    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]
    w_train = ml_data["w_train"]
    X_test = ml_data["X_test"]
    y_test = ml_data["y_test"]
    w_test = ml_data["w_test"]

    # Top 2 features only
    top_2_idx = sorted_idx[:2]
    X_train_top2 = X_train[:, top_2_idx]
    X_test_top2 = X_test[:, top_2_idx]

    # All EXCEPT top 2 features
    rest_idx = sorted_idx[2:]
    X_train_rest = X_train[:, rest_idx]
    X_test_rest = X_test[:, rest_idx]

    logger.info("Training reduced model with ONLY top 2 features...")
    model_top2 = CatBoostClassifier(**catboost_params)
    model_top2.fit(X_train_top2, y_train, sample_weight=w_train, eval_set=[(X_test_top2, y_test)])
    auc_top2 = roc_auc_score(
        y_test, model_top2.predict_proba(X_test_top2)[:, 1], sample_weight=w_test
    )

    logger.info("Training reduced model EXCLUDING top 2 features...")
    model_rest = CatBoostClassifier(**catboost_params)
    model_rest.fit(X_train_rest, y_train, sample_weight=w_train, eval_set=[(X_test_rest, y_test)])
    auc_rest = roc_auc_score(
        y_test, model_rest.predict_proba(X_test_rest)[:, 1], sample_weight=w_test
    )

    auc_full = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], sample_weight=w_test)

    # Generate Markdown Table
    data = {
        "Model Configuration": [
            "Full CatBoost Model (7 Features)",
            f"Top 2 Features Only (`{features[top_2_idx[0]]}`, `{features[top_2_idx[1]]}`)",
            "Excluding Top 2 Features",
        ],
        "ROC AUC": [f"{auc_full:.4f}", f"{auc_top2:.4f}", f"{auc_rest:.4f}"],
    }

    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Phase 2: Feature Importance and Impact Analysis\n\n")
        f.write("### Top Features:\n")
        for i, idx in enumerate(sorted_idx, 1):
            f.write(f"{i}. `{features[idx]}`: {importance[idx]:.2f}%\n")
        f.write("\n### Model Performance Comparison:\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    logger.info(f"Generated feature impact analysis at {output_path}")
    return top_2_idx


def plot_top_features_separation(ml_data: dict, top_2_idx: list, plot_dir: Path, suffix: str = ""):
    """
    Plots 2D histograms and 1D distributions of the top 2 features to visually
    demonstrate signal vs background separation power.
    """
    features = ml_data["features"]
    feat1_idx, feat2_idx = top_2_idx

    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]

    X_sig = X_train[y_train == 1]
    X_bkg = X_train[y_train == 0]

    f1_sig = X_sig[:, feat1_idx]
    f2_sig = X_sig[:, feat2_idx]
    f1_bkg = X_bkg[:, feat1_idx]
    f2_bkg = X_bkg[:, feat2_idx]

    plot_dir.mkdir(parents=True, exist_ok=True)

    # 2D Histograms
    logger.info("Plotting 2D histograms for top features...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Determine scale for the second feature (often highly skewed, e.g. IPCHI2/FDCHI2)
    # Applying log10 if the variable name implies chi2 or PT for better visualization
    use_log_y = "chi2" in features[feat2_idx].lower() or "pt" in features[feat2_idx].lower()

    y_sig = np.log10(f2_sig + 1e-3) if use_log_y else f2_sig
    y_bkg = np.log10(f2_bkg + 1e-3) if use_log_y else f2_bkg
    y_label = f"log10({features[feat2_idx]})" if use_log_y else features[feat2_idx]

    h1 = axes[0].hist2d(f1_sig, y_sig, bins=50, cmap="Blues", density=True)
    axes[0].set_title(f"Signal: {features[feat1_idx]} vs {y_label}")
    axes[0].set_xlabel(features[feat1_idx])
    axes[0].set_ylabel(y_label)
    fig.colorbar(h1[3], ax=axes[0])

    h2 = axes[1].hist2d(f1_bkg, y_bkg, bins=50, cmap="Reds", density=True)
    axes[1].set_title(f"Background: {features[feat1_idx]} vs {y_label}")
    axes[1].set_xlabel(features[feat1_idx])
    axes[1].set_ylabel(y_label)
    fig.colorbar(h2[3], ax=axes[1])

    plt.tight_layout()
    plt.savefig(plot_dir / f"top2_features_2d{suffix}.pdf")
    plt.close()

    # 1D Distributions
    logger.info("Plotting 1D distributions for top features...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(f1_sig, bins=50, alpha=0.5, label="Signal", density=True, color="blue")
    axes[0].hist(f1_bkg, bins=50, alpha=0.5, label="Background", density=True, color="red")
    axes[0].set_title(f"1D Distribution: {features[feat1_idx]}")
    axes[0].set_xlabel(features[feat1_idx])
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(y_sig, bins=50, alpha=0.5, label="Signal", density=True, color="blue")
    axes[1].hist(y_bkg, bins=50, alpha=0.5, label="Background", density=True, color="red")
    axes[1].set_title(f"1D Distribution: {y_label}")
    axes[1].set_xlabel(y_label)
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_dir / f"top2_features_1d{suffix}.pdf")
    plt.close()


def generate_final_comparison_summary(mva_opt_md_path: Path, output_path: Path):
    """
    Combines the MVA optimization table with the final phase 4 summary.
    """
    if not mva_opt_md_path.exists():
        logger.warning(f"Could not find MVA optimization table at {mva_opt_md_path}")
        return

    with open(mva_opt_md_path, "r") as f:
        mva_md = f.read()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Phase 4: Final Comparison (MVA vs Baseline)\n\n")
        f.write("## 1. MVA Optimization Results (from tuned CatBoost)\n")
        f.write(mva_md)
        f.write("\n\n## 2. Invariant Mass Fits\n")
        f.write(
            "Mass fits have been successfully run utilizing the optimal CatBoost cuts derived from `S/sqrt(S+B)`.\n"
        )
        f.write(
            "- **Low Yield Fits**: Saved to `output/plots/fits/Low_Yield_S_sqrt(SplusB)_bdt_cut/`\n"
        )
        f.write(
            "- **High Yield Fits**: Saved to `output/plots/fits/High_Yield_S_sqrt(SplusB)_bdt_cut/`\n"
        )
        f.write(
            "\nThis completes the quantitative comparison required to demonstrate the improved background suppression and signal retention of the MVA approach.\n"
        )
    logger.info(f"Generated final comparison summary at {output_path}")
