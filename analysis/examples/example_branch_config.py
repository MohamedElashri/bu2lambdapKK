"""
Example demonstrating branch configuration usage

This shows how to use the branches_config.toml file to control
which branches are loaded from data and MC files.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import DataLoader
from mc_loader import MCLoader
from branch_config import BranchConfig
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 80)
    print("BRANCH CONFIGURATION EXAMPLES")
    print("=" * 80)
    
    # Paths
    mc_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/mc"
    data_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/data"
    
    # Initialize branch config
    branch_config = BranchConfig()
    
    # Display available configuration
    print("\nAvailable branch sets:")
    for set_name in branch_config.list_available_sets():
        print(f"  - {set_name}")
    
    print("\nAvailable presets:")
    for preset in branch_config.list_available_presets():
        print(f"  - {preset}")
    
    # Initialize loaders
    data_loader = DataLoader(data_dir, branch_config)
    mc_loader = MCLoader(mc_dir, branch_config)
    
    # Configuration
    years = ['16']
    polarities = ['MD']
    track_types = ['LL']
    channel_name = 'B2L0barPKpKm'
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Using 'minimal' preset for quick analysis")
    print("=" * 80)
    
    data_minimal = data_loader.load_data(
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name,
        preset='minimal'
    )
    
    for key, arrays in data_minimal.items():
        print(f"\n{key}:")
        print(f"  Events: {len(arrays)}")
        print(f"  Branches: {list(arrays.fields)}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Using 'standard' preset (kinematics + PID)")
    print("=" * 80)
    
    data_standard = data_loader.load_data(
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name,
        preset='standard'
    )
    
    for key, arrays in data_standard.items():
        print(f"\n{key}:")
        print(f"  Events: {len(arrays)}")
        print(f"  Number of branches: {len(arrays.fields)}")
        print(f"  Sample branches: {list(arrays.fields)[:10]}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom branch sets")
    print("=" * 80)
    
    data_custom = data_loader.load_data(
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name,
        branch_sets=['essential', 'pid']  # Just essential + PID
    )
    
    for key, arrays in data_custom.items():
        print(f"\n{key}:")
        print(f"  Events: {len(arrays)}")
        print(f"  Number of branches: {len(arrays.fields)}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: MC with truth info (using 'mc_reco' preset)")
    print("=" * 80)
    
    mc_with_truth = mc_loader.load_reconstructed(
        sample_name='KpKm',
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name,
        preset='mc_reco',  # Includes MC truth branches
        include_mc_truth=True
    )
    
    for key, arrays in mc_with_truth.items():
        print(f"\n{key}:")
        print(f"  Events: {len(arrays)}")
        print(f"  Number of branches: {len(arrays.fields)}")
        # Check if truth branches are present
        truth_branches = [b for b in arrays.fields if 'TRUE' in b]
        print(f"  MC truth branches: {len(truth_branches)}")
        if truth_branches:
            print(f"  Sample truth branches: {truth_branches[:5]}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 5: MC truth tree (generator level)")
    print("=" * 80)
    
    mc_truth = mc_loader.load_truth(
        sample_name='KpKm',
        years=years,
        polarities=polarities,
        use_config=True  # Use truth_branches from config
    )
    
    for key, arrays in mc_truth.items():
        print(f"\n{key}:")
        print(f"  Events: {len(arrays)}")
        print(f"  Branches: {list(arrays.fields)}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Explicit branch list (overrides config)")
    print("=" * 80)
    
    my_branches = ['Bu_M', 'Bu_PT', 'L0_M', 'L0_PT', 'h1_PT', 'h2_PT']
    
    data_explicit = data_loader.load_data(
        years=years,
        polarities=polarities,
        track_types=track_types,
        channel_name=channel_name,
        branches=my_branches
    )
    
    for key, arrays in data_explicit.items():
        print(f"\n{key}:")
        print(f"  Events: {len(arrays)}")
        print(f"  Branches: {list(arrays.fields)}")
    
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    
    print("\nHow to use:")
    print("1. Edit analysis/branches_config.toml to define your branch sets")
    print("2. Use 'preset' for quick configurations (minimal, standard, mc_reco)")
    print("3. Use 'branch_sets' to mix and match sets (['essential', 'pid'])")
    print("4. Use 'branches' to explicitly specify branches (overrides config)")
    print("5. For MC, set include_mc_truth=True to get TRUE* branches")
    
    print("\nPreset descriptions:")
    presets = {
        'minimal': 'Just essential branches for quick fits',
        'standard': 'Essential + kinematics + PID (recommended for analysis)',
        'full_data': 'Everything including DTF and triggers',
        'mc_reco': 'Standard + MC truth branches (for MC studies)',
        'mc_truth': 'For MCDecayTree (generator-level studies)'
    }
    for preset, desc in presets.items():
        print(f"  - {preset}: {desc}")

if __name__ == "__main__":
    main()
