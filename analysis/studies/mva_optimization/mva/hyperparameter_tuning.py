"""
Optuna Tuning script for XGBoost MVA
"""

import json
import logging
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from config_loader import StudyConfig
from data_preparation import load_and_prepare_data
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def optimize_hyperparameters(config: StudyConfig, ml_data: dict, n_trials: int = 50):
    logger.info("Initializing Optuna Hyperparameter Optimization...")

    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]
    X_test = ml_data["X_test"]
    y_test = ml_data["y_test"]

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        # We also need early stopping to prevent overtraining
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=config.xgboost.get("random_state", 42),
            early_stopping_rounds=10,
            eval_metric="logloss",
            **params,
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value (ROC AUC on Test): {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Write the best params down to a JSON file to be injected into the main pipeline
    output_dir = Path("analysis_output/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "optuna_best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)

    return trial.params


if __name__ == "__main__":
    config = StudyConfig()
    ml_data = load_and_prepare_data(config)
    optimize_hyperparameters(config, ml_data, n_trials=50)
