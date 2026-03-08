"""
MVA Trainer with XGBoost, LightGBM, CatBoost
"""

import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from config_loader import StudyConfig
from scipy.stats import ks_2samp
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def plot_feature_importance(model, features, output_dir, model_name="xgboost"):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
    else:
        return

    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    plt.figure(figsize=(10, 6))
    plt.barh(pos, importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.title(f"Feature Importance ({model_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_{model_name}.pdf")
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, output_dir, model_name="xgboost"):
    plt.figure(figsize=(8, 8))

    plt.plot(
        fpr,
        tpr,
        color="b",
        label=f"Test ROC (AUC = {roc_auc:.3f})",
        lw=2,
    )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{model_name}.pdf")
    plt.close()


def plot_overtraining(y_train, y_train_pred, y_test, y_test_pred, output_dir, model_name="xgboost"):
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
        bin_centers,
        hist_sig_test,
        yerr=err_sig,
        fmt="o",
        color="darkblue",
        label="Signal (Test)",
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
        f"Overtraining Check ({model_name})\nKS p-value (Sig): {ks_sig:.3f} | KS p-value (Bkg): {ks_bkg:.3f}"
    )
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / f"overtraining_check_{model_name}.pdf")
    plt.close()


def train_and_evaluate_model(config: StudyConfig, ml_data: dict, model_type="xgboost"):
    # Using single Train/Test split like TMVA to make it apples-to-apples comparison
    logger.info(f"Initializing {model_type.upper()} Classifier...")

    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]
    w_train = ml_data["w_train"]

    X_test = ml_data["X_test"]
    y_test = ml_data["y_test"]
    w_test = ml_data["w_test"]

    features = ml_data["features"]

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    if model_type == "xgboost":
        hyperparams = config.xgboost.get("hyperparameters", {})
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=config.xgboost.get("random_state", 42),
            **hyperparams,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )

    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=config.xgboost.get("random_state", 42),
            learning_rate=0.05,
            n_estimators=350,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
        model.fit(
            X_train, y_train, sample_weight=w_train, eval_set=[(X_train, y_train), (X_test, y_test)]
        )

    elif model_type == "catboost":
        model = CatBoostClassifier(
            scale_pos_weight=scale_pos_weight,
            random_seed=config.xgboost.get("random_state", 42),
            learning_rate=0.05,
            iterations=350,
            depth=6,
            verbose=False,
        )
        # Note: CatBoost takes sample_weight inside fit
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_test, y_test)])

    elif model_type == "adaboost":
        dt = DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=0.025, random_state=config.xgboost.get("random_state", 42)
        )
        model = AdaBoostClassifier(
            estimator=dt,
            n_estimators=350,
            learning_rate=0.5,
            random_state=config.xgboost.get("random_state", 42),
        )
        model.fit(X_train, y_train, sample_weight=w_train)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Predictions
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_test_pred, sample_weight=w_test)
    roc_auc = auc(fpr, tpr)

    logger.info(f"{model_type.upper()} Test AUC: {roc_auc:.4f}")

    plot_dir = Path("analysis_output/plots/mva")
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation plots...")
    plot_feature_importance(model, features, plot_dir, model_name=model_type)
    plot_roc_curve(fpr, tpr, roc_auc, plot_dir, model_name=model_type)
    plot_overtraining(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        plot_dir,
        model_name=model_type,
    )

    model_dir = Path("analysis_output/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "xgboost":
        model.save_model(model_dir / "xgboost_bdt.json")
    elif model_type == "catboost":
        model.save_model(model_dir / "catboost_bdt.cbm")

    return model, roc_auc


def train_and_evaluate_bdt(config: StudyConfig, ml_data: dict):
    # Train and compare XGBoost, LightGBM, CatBoost, and AdaBoost
    logger.info("Starting Multi-Algorithm evaluation...")

    xgb_model, xgb_auc = train_and_evaluate_model(config, ml_data, model_type="xgboost")
    lgb_model, lgb_auc = train_and_evaluate_model(config, ml_data, model_type="lightgbm")
    cb_model, cb_auc = train_and_evaluate_model(config, ml_data, model_type="catboost")
    ab_model, ab_auc = train_and_evaluate_model(config, ml_data, model_type="adaboost")

    logger.info("=" * 50)
    logger.info("Algorithm Comparison (Single Train/Test Split AUC):")
    logger.info(f"XGBoost  : {xgb_auc:.4f}")
    logger.info(f"LightGBM : {lgb_auc:.4f}")
    logger.info(f"CatBoost : {cb_auc:.4f}")
    logger.info(f"AdaBoost : {ab_auc:.4f}")
    logger.info("=" * 50)

    # We return the CatBoost model as primary, keeping XGBoost as backup
    return cb_model
