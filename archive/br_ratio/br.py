#!/usr/bin/env python3
"""
A script to calculate branching ratio from fit results and efficiencies.
"""
import yaml
import json
import math
import sys
import os
from pathlib import Path
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate branching ratio from fit results')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--fit-results', default='results/fit_results.json', help='Path to fit results file')
    parser.add_argument('--eff-file', default='results/eff.json', help='Path to efficiency file')
    parser.add_argument('--output', default='results/br_results.json', help='Output file for BR results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--ignore-fit-quality', action='store_true', help='Ignore fit quality issues')
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    # Load inputs
    cfg = load_yaml(Path(args.config))
    fits = load_json(Path(args.fit_results))
    
    # Try to load efficiency file, but continue with default values if it doesn't exist
    eff_path = Path(args.eff_file)
    if eff_path.exists():
        effs = load_json(eff_path)
    else:
        print(f"WARNING: Efficiency file {eff_path} not found. Using default efficiency of 1.0")
        effs = {"sig": {}, "norm": {}}
    
    years = cfg.get('years', [])
    tracks = cfg.get('tracks', [])
    br_norm = cfg.get('br_norm_pdg')
    br_norm_unc = cfg.get('br_norm_pdg_unc', 0)
    
    # Check for required configuration values
    if not years or not tracks or not br_norm:
        print("ERROR: Missing 'years', 'tracks', or 'br_norm_pdg' in config.yml")
        sys.exit(1)
    
    # Constants for fit quality
    GOOD_FIT_STATUS = 0
    MIN_GOOD_COV_QUAL = 2
    
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
    valid_categories = []
    
    print("Calculating branching fraction...")
    print(f"Using BR(norm) = {br_norm:.3e} +/- {br_norm_unc:.3e}\n")
    
    # Process each year/track combination
    for year in years:
        for track in tracks:
            key = f"{year}_{track}"
            sig_key = f"sig_{key}"
            norm_key = f"norm_{key}"
            print(f"Category: {year}, {track}")
            
            # Get fit results
            sig_res = fits.get(sig_key)
            norm_res = fits.get(norm_key)
            
            if not sig_res or not norm_res:
                print(f"  SKIPPING: Missing fit results for {sig_key} or {norm_key}")
                continue
            
            # Check fit quality but don't skip if it's poor (just warn)
            sig_status = sig_res.get('_fit_status', -1)
            sig_cov_qual = sig_res.get('_cov_qual', -1)
            norm_status = norm_res.get('_fit_status', -1)
            norm_cov_qual = norm_res.get('_cov_qual', -1)
            
            fit_quality_warning = False
            
            if sig_status != GOOD_FIT_STATUS or sig_cov_qual < MIN_GOOD_COV_QUAL:
                print(f"  WARNING: Signal fit quality poor (status={sig_status}, cov={sig_cov_qual})")
                fit_quality_warning = True
                
            if norm_status != GOOD_FIT_STATUS or norm_cov_qual < MIN_GOOD_COV_QUAL:
                print(f"  WARNING: Norm fit quality poor (status={norm_status}, cov={norm_cov_qual})")
                fit_quality_warning = True
            
            # Only skip if explicitly told not to ignore fit quality
            if fit_quality_warning and not args.ignore_fit_quality:
                print(f"  SKIPPING: Poor fit quality and --ignore-fit-quality not set")
                continue
            
            # Extract yields and errors
            Nsig = safe_get(sig_res, 'nsig', 'value')
            dNsig = safe_get(sig_res, 'nsig', 'error')
            Nnorm = safe_get(norm_res, 'nsig', 'value')
            dNnorm = safe_get(norm_res, 'nsig', 'error')
            
            if None in (Nsig, dNsig, Nnorm, dNnorm) or Nsig <= 0 or Nnorm <= 0:
                print(f"  SKIPPING: Missing or invalid yield values")
                continue
            
            # Get efficiency data - include defaults if not found
            sig_eff_data = effs.get('sig', {}).get(key, {})
            norm_eff_data = effs.get('norm', {}).get(key, {})
            
            nsp = sig_eff_data.get('n_pass')
            nst = sig_eff_data.get('n_tot')
            nnp = norm_eff_data.get('n_pass')
            nnt = norm_eff_data.get('n_tot')
            
            # If efficiency data is missing, use default values (but warn)
            if None in (nsp, nst, nnp, nnt) or nst <= 0 or nnt <= 0:
                print(f"  WARNING: Missing efficiency data, using default efficiency of 1.0")
                # Default efficiency: assume 1.0 with reasonable n_pass/n_tot values
                nsp = nst = 1000
                nnp = nnt = 1000
            
            # Accumulate values
            sum_Nsig += Nsig
            sum_dNsig2 += dNsig**2
            sum_Nnorm += Nnorm
            sum_dNnorm2 += dNnorm**2
            
            sum_sig_pass += nsp
            sum_sig_tot += nst
            sum_norm_pass += nnp
            sum_norm_tot += nnt
            
            count_categories += 1
            valid_categories.append(key)
            
            # Calculate individual BR for this category
            sig_eff = nsp / nst
            norm_eff = nnp / nnt
            eff_ratio = norm_eff / sig_eff
            yield_ratio = Nsig / Nnorm
            
            cat_br = yield_ratio * eff_ratio * br_norm
            
            # Print detailed info for this category
            print(f"  Nsig = {Nsig:.1f} ± {dNsig:.1f}, Nnorm = {Nnorm:.1f} ± {dNnorm:.1f}")
            print(f"  Sig eff = {sig_eff:.6f}, Norm eff = {norm_eff:.6f}, Ratio = {eff_ratio:.6f}")
            print(f"  Category BR = {cat_br:.3e}")
    
    # Check if we have any valid categories
    if count_categories == 0:
        print("\nERROR: No valid categories processed. Try using --ignore-fit-quality flag.")
        sys.exit(1)
    
    print(f"\nProcessed {count_categories} valid categories: {', '.join(valid_categories)}")
    
    # Calculate combined values
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
    print(f"Eff_sig_comb  = {eff_sig_comb:.6f} ± {unc_eff_sig:.6f}")
    print(f"Eff_norm_comb = {eff_norm_comb:.6f} ± {unc_eff_norm:.6f}\n")
    
    # Calculate final branching ratio
    try:
        # Yield ratio and uncertainty
        R_yield = sum_Nsig / sum_Nnorm
        unc_R_yield = R_yield * math.sqrt((total_dNsig / sum_Nsig)**2 + (total_dNnorm / sum_Nnorm)**2)
        
        # Efficiency ratio and uncertainty
        R_eff = eff_norm_comb / eff_sig_comb
        unc_R_eff = R_eff * math.sqrt((unc_eff_norm / eff_norm_comb)**2 + (unc_eff_sig / eff_sig_comb)**2)
        
        # Final branching ratio calculation
        BR = R_yield * R_eff * br_norm
        
        # Combined uncertainty (quadrature sum of relative uncertainties)
        rel2 = (unc_R_yield / R_yield)**2 + (unc_R_eff / R_eff)**2
        if br_norm_unc and br_norm > 0:
            rel2 += (br_norm_unc / br_norm)**2
        unc_BR = BR * math.sqrt(rel2)
        
        print(f"Yield ratio: {R_yield:.6f} ± {unc_R_yield:.6f}")
        print(f"Efficiency ratio: {R_eff:.6f} ± {unc_R_eff:.6f}")
        
        # Print final result with box around it
        print("\n" + "="*60)
        print(f"RESULT: B(B+ → Λ̄⁰pK⁺K⁻) = ({BR:.3e} ± {unc_BR:.3e})")
        print("="*60)
        
        # Save results to JSON
        results = {
            "branching_ratio": {
                "value": BR,
                "error": unc_BR
            },
            "yield_ratio": {
                "value": R_yield,
                "error": unc_R_yield
            },
            "efficiency_ratio": {
                "value": R_eff,
                "error": unc_R_eff
            },
            "yields": {
                "signal": {
                    "value": sum_Nsig,
                    "error": total_dNsig
                },
                "norm": {
                    "value": sum_Nnorm,
                    "error": total_dNnorm
                }
            },
            "efficiencies": {
                "signal": {
                    "value": eff_sig_comb,
                    "error": unc_eff_sig,
                    "n_pass": sum_sig_pass,
                    "n_tot": sum_sig_tot
                },
                "norm": {
                    "value": eff_norm_comb,
                    "error": unc_eff_norm,
                    "n_pass": sum_norm_pass,
                    "n_tot": sum_norm_tot
                }
            },
            "config": {
                "br_norm": br_norm,
                "br_norm_unc": br_norm_unc,
                "years": years,
                "tracks": tracks,
                "valid_categories": valid_categories
            }
        }
        
        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR during BR calculation: {e}")
        sys.exit(1)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())