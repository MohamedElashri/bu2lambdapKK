#!/usr/bin/env python3
"""
Check which hadron (h1 or h2) is K+ and which is K-
Based on their ID field (PDG codes)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "modules"))

from data_handler import TOMLConfig, DataManager
import awkward as ak
import numpy as np

def main():
    print("\n" + "="*80)
    print("CHECKING HADRON CHARGES (h1 vs h2)")
    print("="*80)
    
    config = TOMLConfig("./config")
    dm = DataManager(config)
    
    # Load some data to check
    print("\nLoading 2016 MD LL data...")
    data = dm.load_tree("data", 2016, "MD", "LL")
    
    print(f"Total events: {len(data):,}")
    
    # Check if ID fields exist
    print("\nChecking available fields:")
    if "h1_ID" in data.fields:
        print("  ✓ h1_ID found")
    else:
        print("  ✗ h1_ID NOT found")
    
    if "h2_ID" in data.fields:
        print("  ✓ h2_ID found")
    else:
        print("  ✗ h2_ID NOT found")
    
    # PDG codes: K+ = 321, K- = -321
    if "h1_ID" in data.fields and "h2_ID" in data.fields:
        print("\nPDG ID statistics:")
        print(f"\nh1_ID values:")
        h1_ids, h1_counts = np.unique(ak.to_numpy(data["h1_ID"]), return_counts=True)
        for pid, count in zip(h1_ids[:10], h1_counts[:10]):  # Show first 10
            print(f"  {pid:6d}: {count:8d} events ({100*count/len(data):5.2f}%)")
        
        print(f"\nh2_ID values:")
        h2_ids, h2_counts = np.unique(ak.to_numpy(data["h2_ID"]), return_counts=True)
        for pid, count in zip(h2_ids[:10], h2_counts[:10]):  # Show first 10
            print(f"  {pid:6d}: {count:8d} events ({100*count/len(data):5.2f}%)")
        
        # Check specific combinations
        print("\nParticle identification:")
        n_h1_kplus = ak.sum(data["h1_ID"] == 321)
        n_h1_kminus = ak.sum(data["h1_ID"] == -321)
        n_h2_kplus = ak.sum(data["h2_ID"] == 321)
        n_h2_kminus = ak.sum(data["h2_ID"] == -321)
        
        print(f"\n  h1 = K+ (321):   {n_h1_kplus:8d} events ({100*n_h1_kplus/len(data):5.2f}%)")
        print(f"  h1 = K- (-321):  {n_h1_kminus:8d} events ({100*n_h1_kminus/len(data):5.2f}%)")
        print(f"  h2 = K+ (321):   {n_h2_kplus:8d} events ({100*n_h2_kplus/len(data):5.2f}%)")
        print(f"  h2 = K- (-321):  {n_h2_kminus:8d} events ({100*n_h2_kminus/len(data):5.2f}%)")
        
        # Determine which is which
        print("\n" + "="*80)
        print("CONCLUSION:")
        print("="*80)
        
        if n_h1_kplus > n_h1_kminus:
            print("  h1 is predominantly K+ (positive kaon)")
        elif n_h1_kminus > n_h1_kplus:
            print("  h1 is predominantly K- (negative kaon)")
        else:
            print("  h1 charge is ambiguous")
        
        if n_h2_kplus > n_h2_kminus:
            print("  h2 is predominantly K+ (positive kaon)")
        elif n_h2_kminus > n_h2_kplus:
            print("  h2 is predominantly K- (negative kaon)")
        else:
            print("  h2 charge is ambiguous")
        
        print("\nFor charmonium analysis:")
        if n_h1_kminus > n_h1_kplus:
            print("  → Use M_LpKm_h1 for M(Λ̄pK⁻) charmonium mass")
            print("  → h1 is K-, h2 is K+")
        elif n_h2_kminus > n_h2_kplus:
            print("  → Use M_LpKm_h2 for M(Λ̄pK⁻) charmonium mass")
            print("  → h2 is K-, h1 is K+")
        else:
            print("  → Cannot determine - check MC truth matching")
    
    else:
        print("\n⚠️  ID fields not available in data")
        print("   Checking MC for truth information...")
        
        # Try MC
        print("\nLoading J/psi MC...")
        mc = dm.load_tree("Jpsi", 2016, "MD", "LL")
        
        if "h1_TRUEID" in mc.fields and "h2_TRUEID" in mc.fields:
            print("\nMC Truth IDs:")
            print(f"\nh1_TRUEID:")
            h1_true_ids, h1_true_counts = np.unique(ak.to_numpy(mc["h1_TRUEID"]), return_counts=True)
            for pid, count in zip(h1_true_ids[:10], h1_true_counts[:10]):
                print(f"  {pid:6d}: {count:8d} events ({100*count/len(mc):5.2f}%)")
            
            print(f"\nh2_TRUEID:")
            h2_true_ids, h2_true_counts = np.unique(ak.to_numpy(mc["h2_TRUEID"]), return_counts=True)
            for pid, count in zip(h2_true_ids[:10], h2_true_counts[:10]):
                print(f"  {pid:6d}: {count:8d} events ({100*count/len(mc):5.2f}%)")
            
            # Count K+ and K-
            n_h1_kplus = ak.sum(mc["h1_TRUEID"] == 321)
            n_h1_kminus = ak.sum(mc["h1_TRUEID"] == -321)
            n_h2_kplus = ak.sum(mc["h2_TRUEID"] == 321)
            n_h2_kminus = ak.sum(mc["h2_TRUEID"] == -321)
            
            print("\nMC Truth identification:")
            print(f"  h1 = K+: {n_h1_kplus:6d} ({100*n_h1_kplus/len(mc):5.2f}%)")
            print(f"  h1 = K-: {n_h1_kminus:6d} ({100*n_h1_kminus/len(mc):5.2f}%)")
            print(f"  h2 = K+: {n_h2_kplus:6d} ({100*n_h2_kplus/len(mc):5.2f}%)")
            print(f"  h2 = K-: {n_h2_kminus:6d} ({100*n_h2_kminus/len(mc):5.2f}%)")
            
            print("\n" + "="*80)
            print("CONCLUSION (from MC truth):")
            print("="*80)
            
            if n_h1_kminus > n_h1_kplus:
                print("  h1 is K- (use M_LpKm_h1 for charmonium)")
            elif n_h2_kminus > n_h2_kplus:
                print("  h2 is K- (use M_LpKm_h2 for charmonium)")

if __name__ == "__main__":
    main()
