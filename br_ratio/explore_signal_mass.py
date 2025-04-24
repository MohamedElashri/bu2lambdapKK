import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from loaders import load_data
from selections import trigger_mask
from branches import canonical

# Load configuration
CFG = yaml.safe_load(open("config.yml"))

# Define the signal decay mode
SIGNAL_MODE = CFG["patterns"]["signal_mc"]
SIGNAL_DATA_PATH = CFG["signal_data_dir"]

# Ensure plots directory exists
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Starting signal mass exploration...")

# Loop through years and tracks
for year in CFG["years"]:
    for track in CFG["tracks"]:
        print(f"Processing: Year={year}, Track={track}")

        # Load signal data
        print(f"\tLoading data...")
        data = load_data(data_path=SIGNAL_DATA_PATH,
                         decay_mode=SIGNAL_MODE,
                         years=[year],
                         tracks=[track],
                         sample="signal")

        if data is None or len(data) == 0:
            print(f"\tNo data loaded for {year}_{track}. Skipping.")
            continue
        print(f"\tLoaded {len(data)} initial events.")

        # Apply trigger selection
        print(f"\tApplying trigger mask...")
        triggered_data = data[trigger_mask(data, "signal")]

        if len(triggered_data) == 0:
            print(f"\tNo events passed trigger for {year}_{track}. Skipping plot.")
            continue
        print(f"\t{len(triggered_data)} events passed trigger.")

        # Get mass data
        mass_col = canonical("signal", ["mass"])[0]
        mass_data = triggered_data[mass_col].to_numpy()

        # --- Plotting ---
        print(f"\tGenerating plot...")
        plt.figure(figsize=(10, 6))
        # Define plot range and bins
        plot_range = (5100, 5800)
        num_bins = 100

        plt.hist(mass_data, bins=num_bins, range=plot_range, histtype='step', label=f'{year} {track} (Triggered)')

        plt.xlabel(f"{mass_col} [MeV]")
        plt.ylabel(f"Events / {(plot_range[1]-plot_range[0])/num_bins:.1f} MeV")
        plt.title(f"Signal Mass Distribution ({year} {track} - After Trigger)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(PLOTS_DIR, f"signal_mass_{year}_{track}.pdf")
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free memory
        print(f"\tSaved plot -> {plot_filename}")

print("Signal mass exploration finished.")
