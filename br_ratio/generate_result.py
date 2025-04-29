import yaml
import json
import math
import sys
from pathlib import Path

# --- Configuration & Constants ---
CONFIG_FILE = Path("config.yml")
FIT_RESULTS_FILE = Path("results/fit_results.json")
EFF_FILE = Path("results/eff.json")  # adjust path if needed

GOOD_FIT_STATUS = 0
MIN_GOOD_COV_QUAL = 2

# --- Utility Functions ---
def safe_get(dct, *keys, default=None):
    """Navigate nested dictionaries safely."""
    val = dct
    try:
        for key in keys:
            val = val[key]
        return val
    except (KeyError, TypeError):
        return default


def binomial_uncertainty(n_pass, n_tot):
    """Compute binomial uncertainty for an efficiency."""
    if n_tot <= 0:
        return 0.0
    eff = n_pass / n_tot
    if eff <= 0 or eff >= 1:
        return 0.0
    return math.sqrt(eff * (1 - eff) / n_tot)


def load_json(path):
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {path}: {e}")
        sys.exit(1)


def load_yaml(path):
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in {path}: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Load inputs
    cfg = load_yaml(CONFIG_FILE)
    fits = load_json(FIT_RESULTS_FILE)
    effs = load_json(EFF_FILE)

    years = cfg.get('years', [])
    tracks = cfg.get('tracks', [])
    br_norm = cfg.get('br_norm_pdg')
    br_norm_unc = cfg.get('br_norm_pdg_unc', 0)

    if not years or not tracks or not br_norm:
        print("ERROR: Missing 'years', 'tracks', or 'br_norm_pdg' in config.yml")
        sys.exit(1)

    # Accumulators
    sum_Nsig = 0.0
    sum_dNsig2 = 0.0
    sum_Nnorm = 0.0
    sum_dNnorm2 = 0.0

    sum_sig_pass = 0
    sum_sig_tot = 0
    sum_norm_pass = 0
    sum_norm_tot = 0

    count_categories = 0

    print("Calculating branching fraction...")
    print(f"Using BR(norm) = {br_norm:.3e} +/- {br_norm_unc:.3e}\n")

    for year in years:
        for track in tracks:
            key = f"{year}_{track}"
            sig_key = f"sig_{key}"
            norm_key = f"norm_{key}"
            print(f"Category: {year}, {track}")

            sig_res = fits.get(sig_key)
            norm_res = fits.get(norm_key)
            if not sig_res or not norm_res:
                print("  Skipping: missing fit results")
                continue

            # Fit quality check
            fs = sig_res.get('_fit_status', -1)
            fq = sig_res.get('_cov_qual', -1)
            ns = norm_res.get('_fit_status', -1)
            nq = norm_res.get('_cov_qual', -1)
            if fs != GOOD_FIT_STATUS or fq < MIN_GOOD_COV_QUAL:
                print(f"  WARNING: signal fit quality poor (status={fs}, cov={fq})")
            if ns != GOOD_FIT_STATUS or nq < MIN_GOOD_COV_QUAL:
                print(f"  WARNING: norm fit quality poor (status={ns}, cov={nq})")

            # Yields
            Nsig = safe_get(sig_res, 'nsig', 'value')
            dNsig = safe_get(sig_res, 'nsig', 'error')
            Nnorm = safe_get(norm_res, 'nsig', 'value')
            dNnorm = safe_get(norm_res, 'nsig', 'error')
            if None in (Nsig, dNsig, Nnorm, dNnorm):
                print("  Skipping: missing yield or error")
                continue

            # Efficiencies
            eff_sig = effs.get('sig', {}).get(key, {})
            eff_norm = effs.get('norm', {}).get(key, {})
            nsp = eff_sig.get('n_pass'); nst = eff_sig.get('n_tot')
            nnp = eff_norm.get('n_pass'); nnt = eff_norm.get('n_tot')
            if None in (nsp, nst, nnp, nnt) or nst <= 0 or nnt <= 0:
                print("  Skipping: invalid efficiency data")
                continue

            # Accumulate
            sum_Nsig += Nsig
            sum_dNsig2 += dNsig**2
            sum_Nnorm += Nnorm
            sum_dNnorm2 += dNnorm**2

            sum_sig_pass += nsp
            sum_sig_tot += nst
            sum_norm_pass += nnp
            sum_norm_tot += nnt

            count_categories += 1
            print(f"  Nsig = {Nsig:.1f} ± {dNsig:.1f}, Nnorm = {Nnorm:.1f} ± {dNnorm:.1f}")

    if count_categories == 0:
        print("No valid categories processed. Exiting.")
        sys.exit(1)

    # Combined yields and errors
    total_dNsig = math.sqrt(sum_dNsig2)
    total_dNnorm = math.sqrt(sum_dNnorm2)

    # Combined efficiencies
    eff_sig_comb = sum_sig_pass / sum_sig_tot
    eff_norm_comb = sum_norm_pass / sum_norm_tot
    unc_eff_sig = binomial_uncertainty(sum_sig_pass, sum_sig_tot)
    unc_eff_norm = binomial_uncertainty(sum_norm_pass, sum_norm_tot)

    print("\n--- Combined Numbers ---")
    print(f"Total Nsig    = {sum_Nsig:.1f} ± {total_dNsig:.1f}")
    print(f"Total Nnorm   = {sum_Nnorm:.1f} ± {total_dNnorm:.1f}")
    print(f"Eff_sig_comb  = {eff_sig_comb:.4f} ± {unc_eff_sig:.4f}")
    print(f"Eff_norm_comb = {eff_norm_comb:.4f} ± {unc_eff_norm:.4f}\n")

    # Calculate branching fraction
    try:
        R_yield = sum_Nsig / sum_Nnorm
        unc_R_yield = R_yield * math.sqrt((total_dNsig / sum_Nsig)**2 + (total_dNnorm / sum_Nnorm)**2)

        R_eff = eff_norm_comb / eff_sig_comb
        unc_R_eff = R_eff * math.sqrt((unc_eff_norm / eff_norm_comb)**2 + (unc_eff_sig / eff_sig_comb)**2)

        BR = R_yield * R_eff * br_norm
        # combine uncertainties
        rel2 = (unc_R_yield / R_yield)**2 + (unc_R_eff / R_eff)**2
        if br_norm_unc and br_norm > 0:
            rel2 += (br_norm_unc / br_norm)**2
        unc_BR = BR * math.sqrt(rel2)

        print(f"Final branching ratio: ({BR:.3e} ± {unc_BR:.2e})")
    except Exception as e:
        print(f"ERROR during BR calculation: {e}")
        sys.exit(1)
