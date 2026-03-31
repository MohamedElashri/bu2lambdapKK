import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from utils.config_loader import StudyConfig
from utils.data_preparation import load_and_prepare_data

try:
    from analysis.modules.plot_utils import setup_style

    setup_style()
except Exception:
    pass


def plot_kinematics():
    config = StudyConfig(
        config_file=str(project_root / "analysis" / "config" / "selection.toml"),
        output_dir="output",
    )

    # Load ML data (combines 2016-2018 Data sidebands and MC signal)
    print("Loading data...")
    # By default, load_and_prepare_data handles the cache and feature extraction
    ml_data = load_and_prepare_data(config, test_size=0.3, random_state=42)

    X = ml_data["X_train"]
    y = ml_data["y_train"]
    features = ml_data["features"]

    X_sig = X[y == 1]
    X_bkg = X[y == 0]

    out_dir = Path("output/plots/")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        # log scale features
        if "chi2" in feature.lower() or "pt" in feature.lower():
            sig_data = np.log10(X_sig[:, i] + 1e-3)
            bkg_data = np.log10(X_bkg[:, i] + 1e-3)
            xlabel = f"log10({feature})"
        else:
            sig_data = X_sig[:, i]
            bkg_data = X_bkg[:, i]
            xlabel = feature

        bins = np.linspace(
            min(np.min(sig_data), np.min(bkg_data)), max(np.max(sig_data), np.max(bkg_data)), 50
        )

        ax.hist(sig_data, bins=bins, color="tab:blue", alpha=0.5, density=True, label="MC Signal")
        ax.hist(
            bkg_data, bins=bins, color="tab:red", alpha=0.5, density=True, label="Data Sidebands"
        )

        ax.set_title(f"Normalized Distribution: {feature}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized Density")
        ax.legend(loc="upper right")

    plt.savefig(out_dir / "kinematic_comparisons.pdf", bbox_inches="tight")
    print("Saved output/plots/kinematic_comparisons.pdf")


if __name__ == "__main__":
    plot_kinematics()
