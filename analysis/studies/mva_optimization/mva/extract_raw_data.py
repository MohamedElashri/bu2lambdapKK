from pathlib import Path

import numpy as np

# Add project root to sys.path to allow correct module resolution
project_root = Path(__file__).resolve().parent.parent.parent
import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from catboost import CatBoostClassifier
from config_loader import StudyConfig
from data_preparation import load_and_prepare_data
from sklearn.metrics import auc, roc_curve


def extract_catboost():
    # Make sure we run from mva dir so paths match
    import os

    script_dir = Path(__file__).resolve().parent
    os.chdir(str(script_dir))

    config = StudyConfig()
    ml_data = load_and_prepare_data(config)

    model_path = Path("analysis_output/models/catboost_bdt.cbm")
    if not model_path.exists():
        print(f"Catboost model not found at {model_path}")
        return

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    X_test = ml_data["X_test"]
    y_test = ml_data["y_test"]
    w_test = ml_data["w_test"]

    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]
    w_train = ml_data["w_train"]

    y_test_pred = model.predict_proba(X_test)[:, 1]
    y_train_pred = model.predict_proba(X_train)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred, sample_weight=w_test)
    roc_auc = auc(fpr, tpr)

    importance = model.get_feature_importance()
    features = ml_data["features"]

    out_dir = Path("../comparison/raw_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "catboost_raw.npz",
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        roc_auc=np.array([roc_auc]),
        y_test=y_test,
        y_test_pred=y_test_pred,
        w_test=w_test,
        y_train=y_train,
        y_train_pred=y_train_pred,
        w_train=w_train,
        importance=importance,
        features=np.array(features),
    )

    print(f"Saved Python CatBoost raw data to {out_dir / 'catboost_raw.npz'}")


if __name__ == "__main__":
    extract_catboost()
