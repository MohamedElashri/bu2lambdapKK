"""
MVA Trainer for XGBoost
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from config_loader import StudyConfig
from scipy.stats import ks_2samp
from sklearn.metrics import auc, roc_curve

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def plot_feature_importance(model, features, output_dir):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    plt.figure(figsize=(10, 6))
    plt.barh(pos, importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.title("Feature Importance (Weight)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.pdf")
    plt.close()


def plot_roc_curve(model, X_train, y_train, X_test, y_test, output_dir):
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)

    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_train, tpr_train, label=f"Train AUC = {auc_train:.3f}", color="blue")
    plt.plot(fpr_test, tpr_test, label=f"Test AUC = {auc_test:.3f}", color="red", linestyle="--")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.pdf")
    plt.close()


def plot_overtraining(model, X_train, y_train, X_test, y_test, output_dir):
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    sig_train = y_train_pred[y_train == 1]
    bkg_train = y_train_pred[y_train == 0]
    sig_test = y_test_pred[y_test == 1]
    bkg_test = y_test_pred[y_test == 0]

    plt.figure(figsize=(10, 7))
    bins = np.linspace(0, 1, 40)

    # Train histograms (filled)
    plt.hist(sig_train, bins=bins, alpha=0.4, color="blue", density=True, label="Signal (Train)")
    plt.hist(bkg_train, bins=bins, alpha=0.4, color="red", density=True, label="Background (Train)")

    # Test histograms (points)
    hist_sig_test, _ = np.histogram(sig_test, bins=bins, density=True)
    hist_bkg_test, _ = np.histogram(bkg_test, bins=bins, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Avoid division by zero in error bars
    err_sig = np.where(len(sig_test) > 0, np.sqrt(hist_sig_test / len(sig_test)), 0)
    err_bkg = np.where(len(bkg_test) > 0, np.sqrt(hist_bkg_test / len(bkg_test)), 0)

    plt.errorbar(
        bin_centers, hist_sig_test, yerr=err_sig, fmt="o", color="darkblue", label="Signal (Test)"
    )
    plt.errorbar(
        bin_centers,
        hist_bkg_test,
        yerr=err_bkg,
        fmt="o",
        color="darkred",
        label="Background (Test)",
    )

    ks_sig = (
        ks_2samp(sig_train, sig_test).pvalue if len(sig_train) > 0 and len(sig_test) > 0 else 1.0
    )
    ks_bkg = (
        ks_2samp(bkg_train, bkg_test).pvalue if len(bkg_train) > 0 and len(bkg_test) > 0 else 1.0
    )

    plt.title(
        f"Overtraining Check\nKS p-value (Sig): {ks_sig:.3f} | KS p-value (Bkg): {ks_bkg:.3f}"
    )
    plt.xlabel("XGBoost Probability")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "overtraining_check.pdf")
    plt.close()


def train_and_evaluate_bdt(config: StudyConfig, ml_data: dict):
    logger.info("Initializing XGBoost Classifier...")

    hyperparams = config.xgboost.get("hyperparameters", {})

    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]
    X_test = ml_data["X_test"]
    y_test = ml_data["y_test"]

    # Calculate scale_pos_weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=config.xgboost.get("random_state", 42),
        **hyperparams,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=10)

    plot_dir = Path("analysis_output/plots/mva")
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation plots...")
    features = ml_data["features"]
    plot_feature_importance(model, features, plot_dir)
    plot_roc_curve(model, X_train, y_train, X_test, y_test, plot_dir)
    plot_overtraining(model, X_train, y_train, X_test, y_test, plot_dir)

    model_dir = Path("analysis_output/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "xgboost_bdt.json")
    logger.info(f"Model saved to {model_dir / 'xgboost_bdt.json'}")

    return model
