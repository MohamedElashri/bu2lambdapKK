# File: br_ratio/report_results.py
import json
import yaml
import pandas as pd
import math
import sys
import os

# --- Configuration & Constants ---
CONFIG_FILE = "config.yml"
FIT_RESULTS_FILE = "fit_results.json"
EFF_FILE = "eff.json"
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define acceptable fit quality
GOOD_FIT_STATUS = [0] # Allow only status 0 for "OK"
GOOD_COV_QUAL = [3] # Allow only covQual 3 for "OK"

# --- Helper Functions ---
def binomial_unc(n_pass, n_tot):
    """Calculate binomial uncertainty."""
    if n_tot is None or n_tot <= 0 or n_pass is None:
        return 0.0
    # Ensure n_pass is not negative and not greater than n_tot
    n_pass = max(0, min(n_pass, n_tot))
    eff = n_pass / n_tot
    # Avoid math domain error for eff=0 or eff=1, return 0 uncertainty
    if eff <= 0 or eff >= 1:
        return 0.0
    return math.sqrt(eff * (1 - eff) / n_tot)

def format_value_error(value, error, precision_val=2, precision_err=2, scientific=False):
    """Formats value ± error."""
    if value is None: return "N/A"
    if error is None: error = 0.0 # Treat missing error as zero for formatting
    try:
        if scientific:
            return f"{value:.{precision_val}e} ± {error:.{precision_err}e}"
        else:
            return f"{value:.{precision_val}f} ± {error:.{precision_err}f}"
    except (TypeError, ValueError):
        return "Error" # Handle non-numeric inputs gracefully

def get_fit_status_label(status, cov_qual):
    """Provides a label based on fit status and covariance quality."""
    status_ok = status in GOOD_FIT_STATUS
    cov_ok = cov_qual in GOOD_COV_QUAL

    if status is None or cov_qual is None: return "UNKNOWN"
    if status == -999: return "FAILED (No Data/Fit)"
    if status == -998: return "FAILED (Exception)"

    if status_ok and cov_ok: return "OK"
    if not status_ok and cov_ok: return f"WARN (Status={status})"
    if status_ok and not cov_ok: return f"WARN (CovQual={cov_qual})"
    return f"FAIL (Status={status}, CovQual={cov_qual})"

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

# --- 1. Efficiency Table ---
print("\n--- Generating Efficiency Report ---")
eff_data = []
for sample_type in effs.keys(): # 'sig', 'norm'
    if not isinstance(effs[sample_type], dict): continue
    for category, data in effs[sample_type].items():
        n_pass = data.get('n_pass')
        n_tot = data.get('n_tot')
        eff_val = data.get('eff') # Use pre-calculated if available

        if n_tot is not None and n_tot > 0 and n_pass is not None:
            eff_calc = n_pass / n_tot
            eff_unc = binomial_unc(n_pass, n_tot)
            if eff_val is None: eff_val = eff_calc # Use calculated if not present
            eff_str = format_value_error(eff_val, eff_unc, precision_val=4, precision_err=4)
        else:
            eff_str = "N/A"
            n_pass = n_pass if n_pass is not None else "N/A"
            n_tot = n_tot if n_tot is not None else "N/A"

        eff_data.append({
            "Sample": sample_type,
            "Category": category,
            "N Passed": n_pass,
            "N Total": n_tot,
            "Efficiency": eff_str
        })

if eff_data:
    eff_df = pd.DataFrame(eff_data)
    eff_df.sort_values(by=["Sample", "Category"], inplace=True)
    print(eff_df.to_string(index=False))
    eff_df.to_csv(os.path.join(OUTPUT_DIR, "efficiency_report.csv"), index=False)
    print(f"\nSaved efficiency report to {OUTPUT_DIR}/efficiency_report.csv")
else:
    print("No efficiency data found.")


# --- 2. Fit Quality and Results Table ---
print("\n--- Generating Fit Results Report ---")
fit_data = []
param_keys = ["nsig", "nbkg", "mean", "sigma", "alpha", "n", "c1", "c2"] # All possible params

for fit_id, results in fit_results.items():
    if not isinstance(results, dict): # Handle potential errors stored as strings
        fit_data.append({"Fit ID": fit_id, "Status": "ERROR (Invalid JSON entry)", **{k: "N/A" for k in param_keys}})
        continue

    row = {"Fit ID": fit_id}
    status = results.get('_fit_status')
    cov_qual = results.get('_cov_qual')
    row["Status"] = get_fit_status_label(status, cov_qual)

    for param in param_keys:
        val = results.get(param, {}).get('value')
        err = results.get(param, {}).get('error')
        # Adjust precision based on parameter type
        prec_val, prec_err = (1, 1) if param in ["nsig", "nbkg"] else (3, 3)
        if param in ["alpha", "n", "c1", "c2"]: prec_val, prec_err = (4, 4)

        row[f"{param}"] = format_value_error(val, err, prec_val, prec_err) if val is not None else "N/A"

    fit_data.append(row)

if fit_data:
    fit_df = pd.DataFrame(fit_data)
    # Sort for consistency
    fit_df['Sample'] = fit_df['Fit ID'].apply(lambda x: x.split('_')[0])
    fit_df['Year'] = fit_df['Fit ID'].apply(lambda x: x.split('_')[1])
    fit_df['Track'] = fit_df['Fit ID'].apply(lambda x: x.split('_')[2])
    fit_df.sort_values(by=["Sample", "Year", "Track"], inplace=True)
    fit_df.drop(columns=['Sample', 'Year', 'Track'], inplace=True)

    # Select columns to display (can customize)
    display_cols = ["Fit ID", "Status", "nsig", "nbkg", "mean", "sigma"]
    # Add other params if they exist in the data
    for p in ["alpha", "n", "c1", "c2"]:
         if p in fit_df.columns and fit_df[p].nunique() > 1: # Only add if param used/varied
              display_cols.append(p)

    print(fit_df[display_cols].to_string(index=False))
    fit_df.to_csv(os.path.join(OUTPUT_DIR, "fit_results_report.csv"), index=False)
    print(f"\nSaved fit results report to {OUTPUT_DIR}/fit_results_report.csv")
else:
    print("No fit results found.")

# --- 3. Combined Branching Ratio Summary (Optional - repeats br_estimate logic) ---
# This part essentially recalculates based on the *combined* results,
# mimicking br_estimate.py but potentially using the 'all_all' entries if they exist
# Or summing individual good fits as br_estimate.py does.
# Sticking to the br_estimate.py logic of summing individual good fits:

print("\n--- Branching Ratio Calculation Summary (based on individual fits) ---")

BR_NORM_PDG = CFG.get('br_norm_pdg', 0)
BR_NORM_PDG_UNC = CFG.get('br_norm_pdg_unc', 0)
YEARS = CFG.get('years', [])
TRACKS = CFG.get('tracks', [])

total_Nsig = 0
total_dNsig_sq = 0
total_Nnorm = 0
total_dNnorm_sq = 0
total_sig_n_pass = 0
total_sig_n_tot = 0
total_norm_n_pass = 0
total_norm_n_tot = 0
processed_categories = 0
skipped_categories = 0

for year in YEARS:
    for track in TRACKS:
        category = f"{year}_{track}"
        sig_key = f"sig_{category}"
        norm_key = f"norm_{category}"

        sig_fit = fit_results.get(sig_key, {})
        norm_fit = fit_results.get(norm_key, {})
        sig_eff_data = effs.get('sig', {}).get(category, {})
        norm_eff_data = effs.get('norm', {}).get(category, {})

        # Check if all data is present and fits are good quality
        sig_status = sig_fit.get('_fit_status')
        sig_cov_qual = sig_fit.get('_cov_qual')
        norm_status = norm_fit.get('_fit_status')
        norm_cov_qual = norm_fit.get('_cov_qual')

        sig_status_ok = sig_status in GOOD_FIT_STATUS
        sig_cov_ok = sig_cov_qual in GOOD_COV_QUAL
        norm_status_ok = norm_status in GOOD_FIT_STATUS
        norm_cov_ok = norm_cov_qual in GOOD_COV_QUAL

        Nsig = sig_fit.get('nsig', {}).get('value')
        dNsig = sig_fit.get('nsig', {}).get('error')
        Nnorm = norm_fit.get('nsig', {}).get('value') # Norm signal yield is also 'nsig'
        dNnorm = norm_fit.get('nsig', {}).get('error')

        sig_n_pass = sig_eff_data.get('n_pass')
        sig_n_tot = sig_eff_data.get('n_tot')
        norm_n_pass = norm_eff_data.get('n_pass')
        norm_n_tot = norm_eff_data.get('n_tot')

        # Stricter check: only include if fits are OK and all numbers are present
        if (sig_status_ok and sig_cov_ok and norm_status_ok and norm_cov_ok and
            all(v is not None for v in [Nsig, dNsig, Nnorm, dNnorm,
                                        sig_n_pass, sig_n_tot, norm_n_pass, norm_n_tot]) and
            sig_n_tot > 0 and norm_n_tot > 0):

            total_Nsig += Nsig
            total_dNsig_sq += dNsig**2
            total_Nnorm += Nnorm
            total_dNnorm_sq += dNnorm**2
            total_sig_n_pass += sig_n_pass
            total_sig_n_tot += sig_n_tot
            total_norm_n_pass += norm_n_pass
            total_norm_n_tot += norm_n_tot
            processed_categories += 1
        else:
            skipped_categories += 1
            print(f"  Skipping category {category} due to bad fit or missing data.")

print(f"\nProcessed {processed_categories} categories, Skipped {skipped_categories} categories.")

if processed_categories == 0:
    print("\nError: No categories suitable for combined BR calculation.")
    sys.exit(1)

# --- Calculate Combined Results ---
try:
    total_dNsig = math.sqrt(total_dNsig_sq)
    total_dNnorm = math.sqrt(total_dNnorm_sq)

    comb_sig_eff = total_sig_n_pass / total_sig_n_tot
    comb_sig_eff_unc = binomial_unc(total_sig_n_pass, total_sig_n_tot)

    comb_norm_eff = total_norm_n_pass / total_norm_n_tot
    comb_norm_eff_unc = binomial_unc(total_norm_n_pass, total_norm_n_tot)

    print(f"\nCombined Inputs (from {processed_categories} categories):")
    print(f"  Nsig          = {format_value_error(total_Nsig, total_dNsig, 1, 1)}")
    print(f"  Nnorm         = {format_value_error(total_Nnorm, total_dNnorm, 1, 1)}")
    print(f"  Sig Efficiency= {format_value_error(comb_sig_eff, comb_sig_eff_unc, 4, 4)}")
    print(f"  Norm Efficiency={format_value_error(comb_norm_eff, comb_norm_eff_unc, 4, 4)}")
    print(f"  BR(Norm) PDG  = {format_value_error(BR_NORM_PDG, BR_NORM_PDG_UNC, 3, 3, scientific=True)}")

    # Calculate final BR
    if total_Nnorm <= 0 or comb_sig_eff <= 0 or BR_NORM_PDG <= 0:
        raise ValueError("Division by zero or negative value in final calculation.")

    R_yield = total_Nsig / total_Nnorm
    rel_err2_yield = (total_dNsig / total_Nsig)**2 + (total_dNnorm / total_Nnorm)**2 if total_Nsig > 0 and total_Nnorm > 0 else float('inf')

    R_eff = comb_norm_eff / comb_sig_eff
    rel_err2_eff = (comb_norm_eff_unc / comb_norm_eff)**2 + (comb_sig_eff_unc / comb_sig_eff)**2 if comb_norm_eff > 0 and comb_sig_eff > 0 else float('inf')

    rel_err2_norm_br = (BR_NORM_PDG_UNC / BR_NORM_PDG)**2 if BR_NORM_PDG > 0 and BR_NORM_PDG_UNC is not None else 0

    final_br = R_yield * R_eff * BR_NORM_PDG
    total_rel_err2 = rel_err2_yield + rel_err2_eff + rel_err2_norm_br

    if total_rel_err2 < 0 or math.isinf(total_rel_err2):
        final_unc = float('nan')
    else:
        final_unc = final_br * math.sqrt(total_rel_err2)

    print("\n" + "=" * 52)
    print(f" Combined Branching Ratio Result:")
    print(f"   BR = {final_br:.3e} ± {final_unc:.2e}")
    print("=" * 52)

    # Save summary to text file
    with open(os.path.join(OUTPUT_DIR, "br_summary_report.txt"), "w") as f:
        f.write("Branching Ratio Calculation Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Processed {processed_categories} categories, Skipped {skipped_categories} categories.\n\n")
        f.write("Combined Inputs:\n")
        f.write(f"  Nsig          = {format_value_error(total_Nsig, total_dNsig, 1, 1)}\n")
        f.write(f"  Nnorm         = {format_value_error(total_Nnorm, total_dNnorm, 1, 1)}\n")
        f.write(f"  Sig Efficiency= {format_value_error(comb_sig_eff, comb_sig_eff_unc, 4, 4)}\n")
        f.write(f"  Norm Efficiency={format_value_error(comb_norm_eff, comb_norm_eff_unc, 4, 4)}\n")
        f.write(f"  BR(Norm) PDG  = {format_value_error(BR_NORM_PDG, BR_NORM_PDG_UNC, 3, 3, scientific=True)}\n\n")
        f.write("Final Result:\n")
        f.write(f"  BR = {final_br:.3e} ± {final_unc:.2e}\n")
    print(f"\nSaved BR summary report to {OUTPUT_DIR}/br_summary_report.txt")


except ZeroDivisionError:
    print("\nError: Division by zero encountered during final calculation.")
except ValueError as e:
     print(f"\nError: Calculation error - {e}")
except Exception as e:
    print(f"\nError: An unexpected error occurred during final calculation - {e}")
    import traceback
    traceback.print_exc()