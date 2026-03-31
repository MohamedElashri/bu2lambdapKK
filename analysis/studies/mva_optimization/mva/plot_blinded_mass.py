from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from config_loader import StudyConfig
from data_preparation import load_and_prepare_data


def generate_blinded_mass_plots():
    config = StudyConfig("mva_config.toml")

    # We will generate for both LL and DD
    for category in ["LL", "DD"]:
        print(f"Loading data to plot blinded mass for {category}...")
        ml_data = load_and_prepare_data(config, category)

        # Load the corrected mass values for DATA
        data_combined = ml_data["data_combined"]
        bu_mm = data_combined["Bu_MM_corrected"]

        cache_dir = Path("../../analysis_output/mva/cache")
        try:
            from mva.utils.cache import load_cache

            mc_data = load_cache("preprocessed_mc", cache_dir)
            jpsi_mc = mc_data.get(category, {}).get("jpsi", None)
            if jpsi_mc is not None:
                jpsi_mass = jpsi_mc["Bu_MM_corrected"]
            else:
                jpsi_mass = []
        except Exception:
            jpsi_mass = []

        fig, ax = plt.subplots(figsize=(8, 6))

        # Define blinding region
        b_min = 5255
        b_max = 5305

        # We histogram the data outside the blinding region
        bins = np.linspace(5150, 5600, 90)

        # Separate data into low and high sidebands to prevent connecting lines over blinded region
        data_m = ak.to_numpy(bu_mm)

        # Plot full data but NaN out the blinded zone to leave a gap in the error bars
        counts, bin_edges = np.histogram(data_m, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        y_err = np.sqrt(counts)
        # Apply the mask to counts and bin_centers
        blind_mask = (bin_centers > b_min) & (bin_centers < b_max)

        y_plot = np.where(blind_mask, np.nan, counts)
        y_err_plot = np.where(blind_mask, np.nan, y_err)

        ax.errorbar(
            bin_centers,
            y_plot,
            yerr=y_err_plot,
            fmt="ko",
            markersize=4,
            label="Data (Blinded)",
            capsize=2,
        )

        # If we have J/psi MC, we can plot it in the center. Let's scale it so it's visible.
        if len(jpsi_mass) > 0:
            mc_counts, _ = np.histogram(ak.to_numpy(jpsi_mass), bins=bins)
            # Scale MC to match data sideband height approx or just normalize
            scale_factor = (
                np.max(counts[~blind_mask]) / (np.max(mc_counts) + 1e-9)
                if np.max(counts[~blind_mask]) > 0
                else 1.0
            )
            ax.hist(
                bin_centers,
                bins=bin_edges,
                weights=mc_counts * scale_factor,
                histtype="step",
                color="blue",
                alpha=0.8,
                linewidth=1.5,
                label=r"Expected $J/\psi$ Signal (Scaled)",
            )

        # Highlight blinded region visually
        ax.axvspan(b_min, b_max, color="gray", alpha=0.15, hatch="//", label="Blinded Region")

        ax.set_title(f"$B^+$ Mass Distribution After Pre-selection ({category})")
        ax.set_xlabel("$m(\\bar{\\Lambda}pK^-K^+) - m(p\\pi^-) + m_{PDG}(\\Lambda^0)$ [MeV/$c^2$]")
        ax.set_ylabel("Candidates / 5 MeV/$c^2$")
        ax.legend(loc="upper right")

        out_dir = Path("../output/plots/mva")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_plot = out_dir / f"Bu_M_blinded_presel_{category}.pdf"
        plt.tight_layout()
        plt.savefig(out_plot)
        plt.close()
        print(f"Saved {out_plot}")


if __name__ == "__main__":
    generate_blinded_mass_plots()
