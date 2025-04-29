import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from loaders import load_data
from selections import trigger_mask
from branches import canonical

# Load configuration
CFG = yaml.safe_load(open("config.yml"))

# Define the normalization decay mode and path
# Use the specific data mode name if available, otherwise guess from MC pattern
NORM_MODE = CFG.get("norm_mode_data", CFG["patterns"]["norm_mc"])
NORM_DATA_PATH = CFG["norm_data_dir"]

# Ensure plots directory exists
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Starting normalization mass exploration...")

# Loop through years and tracks
for year in CFG["years"]:
    for track in CFG["tracks"]:
        print(f"Processing: Year={year}, Track={track}")

        # Load normalization data
        print(f"\tLoading data...")
        data = load_data(data_path=NORM_DATA_PATH,
                         decay_mode=NORM_MODE,
                         years=[year],
                         tracks=[track],
                         sample="norm") # Changed sample to norm

        if data is None or len(data) == 0:
            print(f"\tNo data loaded for {year}_{track}. Skipping.")
            continue
        print(f"\tLoaded {len(data)} initial events.")

        # Apply trigger selection
        print(f"\tApplying trigger mask...")
        triggered_data = data[trigger_mask(data, "norm")] # Changed sample to norm

        if len(triggered_data) == 0:
            print(f"\tNo events passed trigger for {year}_{track}. Skipping plot.")
            continue
        print(f"\t{len(triggered_data)} events passed trigger.")

        # Get mass data
        mass_col = canonical("norm", ["mass"])[0] # Changed sample to norm
        mass_data = triggered_data[mass_col].to_numpy()

        # --- Plotting ---
        print(f"\tGenerating plot...")
        plt.figure(figsize=(10, 6))
        # Define plot range and bins (same as signal for consistency, adjust as needed)
        plot_range = (5100, 5800)
        num_bins = 100

        plt.hist(mass_data, bins=num_bins, range=plot_range, histtype='step', label=f'{year} {track} (Triggered)')

        plt.xlabel(f"{mass_col} [MeV]")
        plt.ylabel(f"Events / {(plot_range[1]-plot_range[0])/num_bins:.1f} MeV")
        plt.title(f"Norm. Channel Mass Distribution ({year} {track} - After Trigger)") # Updated title
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(PLOTS_DIR, f"norm_mass_{year}_{track}.pdf") # Updated filename
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free memory
        print(f"\tSaved plot -> {plot_filename}")

print("Normalization mass exploration finished.")
