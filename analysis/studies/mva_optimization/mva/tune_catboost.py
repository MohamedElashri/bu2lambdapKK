"""
Optuna Tuning script for CatBoost MVA  [FROZEN — MVA optimisation complete]
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostClassifier

# Make the mva/ sub-directory importable from any working directory
_MVA_DIR = Path(__file__).resolve().parent
if str(_MVA_DIR) not in sys.path:
    sys.path.insert(0, str(_MVA_DIR))

from config_loader import StudyConfig  # local mva config (mva_config.toml)
from data_preparation import load_and_prepare_data  # uses modules.* internally
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def optimize_catboost_hyperparameters(config: StudyConfig, ml_data: dict, n_trials: int = 50):
    logger.info("Initializing Optuna Hyperparameter Optimization for CatBoost...")

    X = np.vstack([ml_data["X_train"], ml_data["X_test"]])
    y = np.concatenate([ml_data["y_train"], ml_data["y_test"]])
    w = np.concatenate([ml_data["w_train"], ml_data["w_test"]])

    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train, w_train = X[train_idx], y[train_idx], w[train_idx]
            X_val, y_val, w_val = X[val_idx], y[val_idx], w[val_idx]

            model = CatBoostClassifier(
                scale_pos_weight=scale_pos_weight,
                random_seed=42,
                early_stopping_rounds=20,
                eval_metric="Logloss",
                verbose=False,
                **params,
            )

            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                use_best_model=True,
            )

            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred, sample_weight=w_val)
            cv_scores.append(score)

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value (CV ROC AUC): {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Output directory relative to where the script is run
    output_dir = Path("../output/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "optuna_catboost_best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)

    return trial.params


if __name__ == "__main__":
    config = StudyConfig()
    ml_data = load_and_prepare_data(config)
    optimize_catboost_hyperparameters(config, ml_data, n_trials=50)
