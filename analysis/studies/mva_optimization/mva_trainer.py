"""
MVA Trainer with XGBoost, LightGBM, CatBoost and K-Fold Cross Validation
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
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

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


def plot_roc_curve_cv(tprs, aucs, mean_fpr, output_dir, model_name="xgboost"):
    plt.figure(figsize=(8, 8))

    for i, (tpr, roc_auc) in enumerate(zip(tprs, aucs)):
        plt.plot(mean_fpr, tpr, alpha=0.3, label=f"ROC fold {i+1} (AUC = {roc_auc:.3f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=f"Mean ROC (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})",
        lw=2,
        alpha=0.8,
    )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Cross-Validated ROC Curve ({model_name})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_cv_{model_name}.pdf")
    plt.close()


def plot_overtraining_cv(
    y_train_list, y_train_pred_list, y_test_list, y_test_pred_list, output_dir, model_name="xgboost"
):
    # Aggregate predictions across all folds
    y_train = np.concatenate(y_train_list)
    y_train_pred = np.concatenate(y_train_pred_list)
    y_test = np.concatenate(y_test_list)
    y_test_pred = np.concatenate(y_test_pred_list)

    sig_train = y_train_pred[y_train == 1]
    bkg_train = y_train_pred[y_train == 0]
    sig_test = y_test_pred[y_test == 1]
    bkg_test = y_test_pred[y_test == 0]

    plt.figure(figsize=(10, 7))
    bins = np.linspace(0, 1, 40)

    # Train histograms (filled)
    plt.hist(sig_train, bins=bins, alpha=0.4, color="blue", density=True, label="Signal (Train CV)")
    plt.hist(
        bkg_train, bins=bins, alpha=0.4, color="red", density=True, label="Background (Train CV)"
    )

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
        label="Signal (Test CV)",
    )
    plt.errorbar(
        bin_centers,
        hist_bkg_test,
        yerr=err_bkg,
        fmt="o",
        color="darkred",
        label="Background (Test CV)",
    )

    ks_sig = (
        ks_2samp(sig_train, sig_test).pvalue if len(sig_train) > 0 and len(sig_test) > 0 else 1.0
    )
    ks_bkg = (
        ks_2samp(bkg_train, bkg_test).pvalue if len(bkg_train) > 0 and len(bkg_test) > 0 else 1.0
    )

    plt.title(
        f"Overtraining Check CV ({model_name})\nKS p-value (Sig): {ks_sig:.3f} | KS p-value (Bkg): {ks_bkg:.3f}"
    )
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / f"overtraining_check_cv_{model_name}.pdf")
    plt.close()


def train_and_evaluate_model(config: StudyConfig, ml_data: dict, model_type="xgboost"):
    logger.info(f"Initializing {model_type.upper()} Classifier with 5-Fold CV...")

    X = np.concatenate([ml_data["X_train"], ml_data["X_test"]])
    y = np.concatenate([ml_data["y_train"], ml_data["y_test"]])
    features = ml_data["features"]

    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=config.xgboost.get("random_state", 42)
    )

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    y_train_list = []
    y_train_pred_list = []
    y_test_list = []
    y_test_pred_list = []

    models = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        logger.info(f"Training Fold {fold+1}/5...")
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]

        if model_type == "xgboost":
            hyperparams = config.xgboost.get("hyperparameters", {})
            model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=config.xgboost.get("random_state", 42),
                **hyperparams,
            )
            model.fit(
                X_train_cv,
                y_train_cv,
                eval_set=[(X_train_cv, y_train_cv), (X_test_cv, y_test_cv)],
                verbose=False,
            )

        elif model_type == "lightgbm":
            # Using similar hyperparameters adapted for LGBM
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
                X_train_cv, y_train_cv, eval_set=[(X_train_cv, y_train_cv), (X_test_cv, y_test_cv)]
            )

        elif model_type == "catboost":
            # CatBoost handles weights differently
            model = CatBoostClassifier(
                scale_pos_weight=scale_pos_weight,
                random_seed=config.xgboost.get("random_state", 42),
                learning_rate=0.05,
                iterations=350,
                depth=6,
                verbose=False,
            )
            model.fit(X_train_cv, y_train_cv, eval_set=[(X_test_cv, y_test_cv)])

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        models.append(model)

        # Predictions
        if model_type == "catboost":
            y_train_pred = model.predict_proba(X_train_cv)[:, 1]
            y_test_pred = model.predict_proba(X_test_cv)[:, 1]
        else:
            y_train_pred = model.predict_proba(X_train_cv)[:, 1]
            y_test_pred = model.predict_proba(X_test_cv)[:, 1]

        y_train_list.append(y_train_cv)
        y_train_pred_list.append(y_train_pred)
        y_test_list.append(y_test_cv)
        y_test_pred_list.append(y_test_pred)

        fpr, tpr, _ = roc_curve(y_test_cv, y_test_pred)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_auc = np.mean(aucs)
    logger.info(f"{model_type.upper()} Mean CV AUC: {mean_auc:.4f}")

    plot_dir = Path("analysis_output/plots/mva")
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation plots...")
    # Use the best model (last fold) for feature importance
    plot_feature_importance(models[-1], features, plot_dir, model_name=model_type)
    plot_roc_curve_cv(tprs, aucs, mean_fpr, plot_dir, model_name=model_type)
    plot_overtraining_cv(
        y_train_list,
        y_train_pred_list,
        y_test_list,
        y_test_pred_list,
        plot_dir,
        model_name=model_type,
    )

    model_dir = Path("analysis_output/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model = models[-1]  # Save the last fold's model as a proxy for the entire pipeline
    if model_type == "xgboost":
        best_model.save_model(model_dir / "xgboost_bdt.json")
    elif model_type == "catboost":
        best_model.save_model(model_dir / "catboost_bdt.cbm")

    return best_model, mean_auc


def train_and_evaluate_bdt(config: StudyConfig, ml_data: dict):
    # Train and compare XGBoost, LightGBM, and CatBoost
    logger.info("Starting Multi-Algorithm evaluation...")

    xgb_model, xgb_auc = train_and_evaluate_model(config, ml_data, model_type="xgboost")
    lgb_model, lgb_auc = train_and_evaluate_model(config, ml_data, model_type="lightgbm")
    cb_model, cb_auc = train_and_evaluate_model(config, ml_data, model_type="catboost")

    logger.info("=" * 50)
    logger.info("Algorithm Comparison (5-Fold CV AUC):")
    logger.info(f"XGBoost  : {xgb_auc:.4f}")
    logger.info(f"LightGBM : {lgb_auc:.4f}")
    logger.info(f"CatBoost : {cb_auc:.4f}")
    logger.info("=" * 50)

    # We return the CatBoost model as primary, keeping XGBoost as backup
    return cb_model
