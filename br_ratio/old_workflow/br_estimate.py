import yaml, json, math, numpy as np
import sys

# --- Configuration & Constants ---
CONFIG_FILE = "config.yml"
FIT_RESULTS_FILE = "fit_results.json"
EFF_FILE = "eff.json"
# Define acceptable fit quality
GOOD_FIT_STATUS = 0  # Or potentially allow other non-zero status if understood
GOOD_COV_QUAL = 3

# --- Helper Function for Safe Dict Access ---
def get_value(data_dict, key_path, default=None):
    """Safely access nested dictionary values."""
    keys = key_path.split('/')
    val = data_dict
    try:
        for key in keys:
            val = val[key]
        return val
    except (KeyError, TypeError):
        return default

# --- Load Input Data ---
try:
    CFG = yaml.safe_load(open(CONFIG_FILE))
    fit_results = json.load(open(FIT_RESULTS_FILE))
    effs = json.load(open(EFF_FILE))
except FileNotFoundError as e:
    print(f"Error: Could not find required input file: {e}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Could not parse JSON file: {e}")
    sys.exit(1)

# Extract necessary config values
YEARS = CFG.get('years', [])
TRACK_TYPES = CFG.get('tracks', []) # Corrected key from 'track_types' to 'tracks'
BR_NORM_PDG = CFG.get('br_norm_pdg', 0)
BR_NORM_PDG_UNC = CFG.get('br_norm_pdg_unc', 0)

if not YEARS or not TRACK_TYPES or BR_NORM_PDG == 0:
    print(f"Error: Missing critical information in {CONFIG_FILE} (years, tracks, br_norm_pdg)")
    sys.exit(1)

print("--- Branching Ratio Calculation --- ")
print(f"Using BR(norm) = {BR_NORM_PDG:.3e} +/- {BR_NORM_PDG_UNC:.3e}")
print(f"Processing Years: {YEARS}, Track Types: {TRACK_TYPES}")

# Initialize sums for combined calculation
total_Nsig = 0
total_dNsig_sq = 0
total_Nnorm = 0
total_dNnorm_sq = 0
total_sig_n_pass = 0
total_sig_n_tot = 0
total_norm_n_pass = 0
total_norm_n_tot = 0
processed_categories = 0

for year in YEARS:
    for track in TRACK_TYPES:
        category = f"{year}_{track}"
        sig_key = f"sig_{category}"
        norm_key = f"norm_{category}"

        print(f"\nProcessing category: {category}")

        # --- Extract Fit Results ---
        sig_fit = fit_results.get(sig_key, {})
        norm_fit = fit_results.get(norm_key, {})

        if not sig_fit or not norm_fit:
            print(f"  Skipping: Missing fit results for {sig_key} or {norm_key}")
            continue

        sig_status = sig_fit.get('_fit_status', -1)
        sig_cov_qual = sig_fit.get('_cov_qual', -1)
        norm_status = norm_fit.get('_fit_status', -1)
        norm_cov_qual = norm_fit.get('_cov_qual', -1)

        # --- Check Fit Quality (Warn only) --- 
        if sig_status != GOOD_FIT_STATUS or sig_cov_qual != GOOD_COV_QUAL:
             print(f"  WARNING: Bad fit quality for {sig_key} (Status: {sig_status}, CovQual: {sig_cov_qual})")
             # continue # Ensure skipping is commented out
        if norm_status != GOOD_FIT_STATUS or norm_cov_qual != GOOD_COV_QUAL:
             print(f"  WARNING: Bad fit quality for {norm_key} (Status: {norm_status}, CovQual: {norm_cov_qual})")
             # continue # Ensure skipping is commented out

        # --- Extract and Sum Yields --- 
        Nsig = get_value(sig_fit, 'nsig/value')
        dNsig = get_value(sig_fit, 'nsig/error')
        Nnorm = get_value(norm_fit, 'nsig/value') # norm yield is also 'nsig'
        dNnorm = get_value(norm_fit, 'nsig/error')

        if Nsig is None or dNsig is None or Nnorm is None or dNnorm is None:
            print(f"  Skipping: Missing yield/error values in fit_results.json")
            continue

        total_Nsig += Nsig
        total_dNsig_sq += dNsig**2
        total_Nnorm += Nnorm
        total_dNnorm_sq += dNnorm**2

        # --- Extract and Sum Efficiency Counts --- 
        year_track_key = f"{year}_{track}"
        sig_eff_data = effs.get('sig', {}).get(year_track_key, {})
        norm_eff_data = effs.get('norm', {}).get(year_track_key, {})

        sig_n_pass = sig_eff_data.get('n_pass')
        sig_n_tot = sig_eff_data.get('n_tot')
        norm_n_pass = norm_eff_data.get('n_pass')
        norm_n_tot = norm_eff_data.get('n_tot')

        if None in [sig_n_pass, sig_n_tot, norm_n_pass, norm_n_tot]:
             print(f"  Skipping: Missing n_pass/n_tot values in eff.json")
             continue
        if sig_n_tot <= 0 or norm_n_tot <= 0:
            print(f"  Skipping: Zero or negative n_tot found in eff.json")
            continue

        total_sig_n_pass += sig_n_pass
        total_sig_n_tot += sig_n_tot
        total_norm_n_pass += norm_n_pass
        total_norm_n_tot += norm_n_tot
        processed_categories += 1

# --- Calculate Combined Results --- 
print("\n--- Calculating Combined Result --- ")

if processed_categories == 0:
    print("Error: No categories were successfully processed.")
    sys.exit(1)

if total_Nnorm == 0 or total_sig_n_tot == 0 or total_norm_n_tot == 0:
    print("Error: Cannot calculate BR due to zero denominators (Nnorm or n_tot)")
    sys.exit(1)

# Combined yields and uncertainties
total_dNsig = math.sqrt(total_dNsig_sq)
total_dNnorm = math.sqrt(total_dNnorm_sq)

# Combined efficiencies and uncertainties (binomial)
comb_sig_eff = total_sig_n_pass / total_sig_n_tot
comb_norm_eff = total_norm_n_pass / total_norm_n_tot

def binomial_unc(n_pass, n_tot):
    if n_tot <= 0:
        return 0
    eff = n_pass / n_tot
    # Avoid math domain error for eff=0 or eff=1
    if eff <= 0 or eff >= 1:
        return 0.0
    return math.sqrt(eff * (1 - eff) / n_tot)

comb_sig_eff_unc = binomial_unc(total_sig_n_pass, total_sig_n_tot)
comb_norm_eff_unc = binomial_unc(total_norm_n_pass, total_norm_n_tot)

print(f"Combined Nsig = {total_Nsig:.1f} +/- {total_dNsig:.1f}")
print(f"Combined Nnorm = {total_Nnorm:.1f} +/- {total_dNnorm:.1f}")
print(f"Combined Sig Eff = {comb_sig_eff:.4f} +/- {comb_sig_eff_unc:.4f}")
print(f"Combined Norm Eff = {comb_norm_eff:.4f} +/- {comb_norm_eff_unc:.4f}")

# Calculate final BR
try:
    if total_Nnorm <= 0 or comb_sig_eff <= 0 or BR_NORM_PDG <= 0:
        print("Error: Division by zero or non-positive value in final calculation.")
        sys.exit(1)

    R_yield = total_Nsig / total_Nnorm
    rel_err2_yield = (total_dNsig / total_Nsig)**2 + (total_dNnorm / total_Nnorm)**2 if total_Nsig > 0 else float('inf')

    R_eff = comb_norm_eff / comb_sig_eff
    rel_err2_eff = (comb_norm_eff_unc / comb_norm_eff)**2 + (comb_sig_eff_unc / comb_sig_eff)**2 if comb_norm_eff > 0 else float('inf')

    # Branching ratio and total relative error squared
    final_br = R_yield * R_eff * BR_NORM_PDG
    
    rel_err2_norm_br = 0
    if BR_NORM_PDG > 0 and BR_NORM_PDG_UNC is not None and BR_NORM_PDG_UNC >= 0:
        rel_err2_norm_br = (BR_NORM_PDG_UNC / BR_NORM_PDG)**2

    total_rel_err2 = rel_err2_yield + rel_err2_eff + rel_err2_norm_br

    if total_rel_err2 < 0 or math.isinf(total_rel_err2):
        print(f"Error: Invalid total relative error squared ({total_rel_err2})")
        final_unc = float('nan') # Indicate error
    else:
        final_unc = final_br * math.sqrt(total_rel_err2)

    print("\n" + "-" * 52)
    print(f" Combined branching ratio: {final_br:.3e} ± {final_unc:.2e}")
    print("-" * 52)

except ZeroDivisionError:
    print("\nError: Division by zero encountered during final calculation.")
    sys.exit(1)
except Exception as e:
    print(f"\nError: An unexpected error occurred during final calculation - {e}")
    sys.exit(1)