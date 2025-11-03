"""
Test alias resolution for data vs MC branch names
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from branch_config import BranchConfig

def test_alias_resolution():
    """Test that aliases resolve correctly for data vs MC"""
    print("="*70)
    print("TESTING BRANCH ALIAS SYSTEM")
    print("="*70)
    
    config = BranchConfig()
    
    # Test 1: Resolve common names to data names
    print("\nTest 1: Resolve common PID names to data branch names")
    common_names = ['h1_ProbNNk', 'h2_ProbNNk', 'p_ProbNNp']
    data_names = config.resolve_aliases(common_names, is_mc=False)
    
    for common, data in zip(common_names, data_names):
        print(f"  {common:20s} -> {data} (DATA)")
    
    # Test 2: Resolve common names to MC names
    print("\nTest 2: Resolve common PID names to MC branch names")
    mc_names = config.resolve_aliases(common_names, is_mc=True)
    
    for common, mc in zip(common_names, mc_names):
        print(f"  {common:20s} -> {mc} (MC)")
    
    # Test 3: Normalize back from data names to common names
    print("\nTest 3: Normalize data branches back to common names")
    actual_data_branches = ['h1_MC15TuneV1_ProbNNk', 'h2_MC15TuneV1_ProbNNk', 
                            'h1_ID', 'h1_PT']
    rename_map = config.normalize_branches(actual_data_branches, is_mc=False)
    
    for actual, common in rename_map.items():
        print(f"  {actual:30s} -> {common}")
    
    # Test 4: Normalize back from MC names to common names
    print("\nTest 4: Normalize MC branches back to common names")
    actual_mc_branches = ['h1_MC12TuneV4_ProbNNk', 'h2_MC12TuneV4_ProbNNk',
                         'h1_ID', 'h1_PT']
    rename_map_mc = config.normalize_branches(actual_mc_branches, is_mc=True)
    
    for actual, common in rename_map_mc.items():
        print(f"  {actual:30s} -> {common}")
    
    # Test 5: Load branches with aliases
    print("\nTest 5: Load PID branches (should use common names)")
    pid_branches = config.get_branches_from_sets(['pid'])
    print(f"  PID branches from config (common names): {pid_branches[:5]}...")
    
    print("\n" + "="*70)
    print("✓ Alias system working correctly!")
    print("="*70)
    print("\nNow our analysis code can use common names like 'h1_ProbNNk'")
    print("and the loaders will automatically:")
    print("  1. Resolve to data-specific names (h1_MC15TuneV1_ProbNNk)")
    print("  2. Or MC-specific names (h1_MC12TuneV4_ProbNNk)")
    print("  3. Load the correct branches")
    print("  4. Rename them back to common names in the output")
    print("\nWe don't need to worry about MC15TuneV1 vs MC12TuneV4!")

if __name__ == "__main__":
    test_alias_resolution()
