from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier

# Make sure we can import local mva modules
from config_loader import StudyConfig
from data_preparation import load_and_prepare_data


def generate_mass_sculpting_plot(category="LL", cut_threshold=0.44):
    print(f"Generating Mass Sculpting Plot for {category} with cut {cut_threshold}")

    # 1. Load config and data
    config_path = Path("mva_config.toml")
    if not config_path.exists():
        print("Error: mva_config.toml not found in current directory.")
        return

    config = StudyConfig(config_file=str(config_path))
    ml_data = load_and_prepare_data(config, category)

    # 2. Get the background events directly from the data_preparation extraction logic
    data_combined = ml_data["data_combined"]

    b_high_min = config.optimization.get("b_high_sideband", [5330.0, 5410.0])[0]
    b_low_min = config.optimization.get("b_low_sideband", [5150.0, 5230.0])[0]
    b_low_max = config.optimization.get("b_low_sideband", [5150.0, 5230.0])[1]

    bu_mm = data_combined["Bu_MM_corrected"]
    bkg_mask = (bu_mm > b_high_min) | ((bu_mm >= b_low_min) & (bu_mm <= b_low_max))
    bkg_events = data_combined[bkg_mask]

    # B-mass variable
    mass_before = bkg_events["Bu_MM_corrected"]

    # 3. Load model
    model_path = Path(f"../output/models/catboost_bdt_{category}.cbm")
    model = CatBoostClassifier()
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    model.load_model(str(model_path))

    # 4. Predict
    features = ml_data["features"]
    import awkward as ak

    df_dict = {}
    for feat in features:
        br = bkg_events[feat]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        df_dict[feat] = ak.to_numpy(br)

    X_bkg = pd.DataFrame(df_dict)[features].values
    scores = model.predict_proba(X_bkg)[:, 1]

    # Apply cut
    pass_cut = scores > cut_threshold
    mass_after = mass_before[pass_cut]

    # 5. Plot normalized overlays
    import matplotlib

    matplotlib.use("Agg")
    plt.figure(figsize=(8, 6))

    counts_b, bins, _ = plt.hist(
        mass_before,
        bins=50,
        range=(5100, 5600),
        histtype="step",
        color="black",
        lw=2,
        density=True,
        label="Before MVA Cut (Normalized)",
    )

    counts_a, _, _ = plt.hist(
        mass_after,
        bins=bins,
        histtype="stepfilled",
        color="dodgerblue",
        alpha=0.5,
        density=True,
        label=f"After MVA > {cut_threshold} (Normalized)",
    )

    plt.axvspan(5255, 5305, color="gray", alpha=0.2, label="B+ Signal Window (Blinded)")

    plt.title(f"$B^+$ Mass Combinatorial Sidebands - Sculpting Check ({category})")
    plt.xlabel("$M(B^+)$ corrected [MeV/$c^2$]")
    plt.ylabel("Normalized Events")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    out_dir = Path("../output/plots/mva")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mass_sculpting_{category}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved mass sculpting plot to {out_path}")


if __name__ == "__main__":
    generate_mass_sculpting_plot("LL", 0.44)
