import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_loader import StudyConfig
from data_preparation import load_and_prepare_data


def extract_correlations():
    import os

    script_dir = Path(__file__).resolve().parent
    os.chdir(str(script_dir))

    config = StudyConfig()
    ml_data = load_and_prepare_data(config)

    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]
    features = ml_data["features"]

    # We combine train and test to get the full correlation matrix
    X = np.vstack((ml_data["X_train"], ml_data["X_test"]))
    y = np.concatenate((ml_data["y_train"], ml_data["y_test"]))
    w = np.concatenate((ml_data["w_train"], ml_data["w_test"]))

    # Signal
    X_sig = X[y == 1]
    # Background
    X_bkg = X[y == 0]

    df_sig = pd.DataFrame(X_sig, columns=features)
    df_bkg = pd.DataFrame(X_bkg, columns=features)

    corr_sig = df_sig.corr().values
    corr_bkg = df_bkg.corr().values

    out_dir = Path("../comparison/raw_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "mva_correlations.npz", corr_sig=corr_sig, corr_bkg=corr_bkg, features=features
    )

    print(f"Saved Python MVA correlation data to {out_dir / 'mva_correlations.npz'}")


if __name__ == "__main__":
    extract_correlations()
