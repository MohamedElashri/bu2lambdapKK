"""
Example script demonstrating MC data loading

This script shows:
1. Loading reconstructed MC (similar to data but with truth info)
2. Loading MC truth data
3. Comparing MC vs Data structure
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_loader import MCLoader
from data_loader import DataLoader
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 80)
    print("MC DATA LOADING EXAMPLE")
    print("=" * 80)
    
    # Paths
    mc_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/mc"
    data_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/data"
    
    # Initialize loaders
    mc_loader = MCLoader(mc_dir)
    data_loader = DataLoader(data_dir)
    
    # Configuration
    years = ['16']
    polarities = ['MD']
    track_types = ['LL']
    channel_name = 'B2L0barPKpKm'  # Signal channel
    
    # Branches to load (subset for demonstration)
    # Particle names: Bu (B+), L0 (Lambda), Lp/Lpi (Lambda daughters), p (bachelor p), h1/h2 (K+K-)
    common_branches = [
        'Bu_M', 'Bu_PT', 'Bu_P', 'Bu_ETA',
        'L0_M', 'L0_PT', 'L0_P',
        'h1_PT', 'h1_P', 'h1_PIDmu',
        'h2_PT', 'h2_P', 'h2_PIDmu',
    ]
    
    print("\n" + "=" * 80)
    print("1. LOADING SIGNAL MC - RECONSTRUCTED DATA")
    print("=" * 80)
    
    mc_reco = mc_loader.load_reconstructed(
        sample_name='KpKm',
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name,
        branches=common_branches
    )
    
    # Display results
    for key, data in mc_reco.items():
        print(f"\n{key}:")
        print(f"  Events: {len(data)}")
        print(f"  Branches: {list(data.fields)}")
        print(f"  Sample Bu_M: {data['Bu_M'][:5]}")
    
    print("\n" + "=" * 80)
    print("2. LOADING SIGNAL MC - TRUTH DATA")
    print("=" * 80)
    
    truth_branches = [
        'Bplus_TRUEP_E', 'Bplus_TRUEP_X', 'Bplus_TRUEP_Y', 'Bplus_TRUEP_Z',
        'Bplus_TRUEPT',
        'Lambda~0_TRUEP_E', 'Lambda~0_TRUEPT',
        'Kplus_TRUEPT', 'Kminus_TRUEPT', 'pplus_TRUEPT'
    ]
    
    mc_truth = mc_loader.load_truth(
        sample_name='KpKm',
        years=years,
        polarities=polarities,
        branches=truth_branches
    )
    
    # Display results
    for key, data in mc_truth.items():
        print(f"\n{key}:")
        print(f"  Events: {len(data)}")
        print(f"  Branches: {list(data.fields)}")
        print(f"  Sample Bplus_TRUEPT: {data['Bplus_TRUEPT'][:5]}")
    
    print("\n" + "=" * 80)
    print("3. LOADING BOTH RECONSTRUCTED AND TRUTH")
    print("=" * 80)
    
    mc_both = mc_loader.load_both(
        sample_name='KpKm',
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name,
        reco_branches=common_branches,
        truth_branches=truth_branches
    )
    
    print(f"\nReconstructed data keys: {list(mc_both['reconstructed'].keys())}")
    print(f"Truth data keys: {list(mc_both['truth'].keys())}")
    
    print("\n" + "=" * 80)
    print("4. COMPARING WITH REAL DATA")
    print("=" * 80)
    
    data = data_loader.load_data(
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name
    )
    
    print("\nReal Data:")
    for key, arrays in data.items():
        print(f"  {key}: {len(arrays)} events")
    
    print("\nMC Reconstructed:")
    for key, arrays in mc_reco.items():
        print(f"  {key}: {len(arrays)} events")
    
    print("\nMC Truth (Generator Level):")
    for key, arrays in mc_truth.items():
        print(f"  {key}: {len(arrays)} events")
    
    print("\n" + "=" * 80)
    print("5. MC TRUTH BRANCHES IN RECONSTRUCTED DATA")
    print("=" * 80)
    print("\nMC reconstructed data contains additional truth branches:")
    print("These branches link reconstructed to generator-level info:")
    
    # Load a few MC truth branches from reconstructed tree
    mc_reco_with_truth = mc_loader.load_reconstructed(
        sample_name='KpKm',
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name,
        branches=['Bu_M', 'Bu_TRUEPT', 'Bu_TRUEID', 'Bu_TRUETAU', 'h1_TRUEPT', 'h2_TRUEPT']
    )
    
    for key, data in mc_reco_with_truth.items():
        print(f"\n{key}:")
        print(f"  Bu_M (reconstructed): {data['Bu_M'][:3]}")
        print(f"  Bu_TRUEPT (true): {data['Bu_TRUEPT'][:3]}")
        print(f"  Bu_TRUEID (true PDG ID): {data['Bu_TRUEID'][:3]}")
        print(f"  h1_TRUEPT (true K+ PT): {data['h1_TRUEPT'][:3]}")
        print(f"  h2_TRUEPT (true K- PT): {data['h2_TRUEPT'][:3]}")
    
    print("\n" + "=" * 80)
    print("6. AVAILABLE MC SAMPLES")
    print("=" * 80)
    
    samples = mc_loader.list_available_samples()
    for sample in samples:
        mc_tree = mc_loader.get_mc_tree_name(sample)
        print(f"  - {sample}: {mc_tree}")
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES BETWEEN MC AND DATA")
    print("=" * 80)
    print("""
1. MC Files Structure:
   - Reconstructed tree (DecayTree): Similar to data + MC truth branches
   - MC Truth tree (MCDecayTree): Pure generator-level information
   
2. Reconstructed MC vs Real Data:
   - MC has ~1284 branches, Data has ~197 branches
   - MC includes: TRUE*, MC*, BC* branches
   - Common branches are identical in structure   
    """)

if __name__ == "__main__":
    main()
